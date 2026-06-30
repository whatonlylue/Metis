"""Tool specs the agent loop can offer to a model.

``build_sandbox_tools`` wraps the existing M0 sandbox layer (``metis.sandbox``)
so the agent can read/write/list inside a project — lockbox enforcement is
inherited for free since these tools call straight through to it.

``build_define_tool`` is the DEFINE-step tool: it reuses ``ProjectSpec``'s own
JSON schema as the tool's input schema, so the model is constrained to exactly
the fields the harness understands, and reuses the M0 project store
(``create_project``/``write_project_yaml``) to persist the result.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from metis.projects import create_project, write_project_yaml
from metis.projects.schema import ProjectSpec
from metis.sandbox import format_listing, list_dir, read_file, run_python, write_file
from metis.sandbox.runlog import log_action


@dataclass
class ToolSpec:
    """A tool offered to the model: a name/schema the model sees, a handler we run."""

    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[[dict[str, Any]], str]


def build_sandbox_tools(project_root: Path) -> list[ToolSpec]:
    """Generic file tools backed by the lockbox-enforced sandbox layer."""
    return [
        ToolSpec(
            name="read_file",
            description=(
                "Read a text file inside the project. Cannot read benchmark/. Long files are "
                "returned in windows of up to 400 lines / 50k chars; page through a big file "
                "with offset (0-based line) and limit instead of re-reading the whole thing."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "offset": {
                        "type": "integer",
                        "description": "0-based line to start at (default 0).",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max lines to return (default 400, max 400).",
                    },
                },
                "required": ["path"],
            },
            handler=lambda args: read_file(
                project_root,
                args["path"],
                offset=int(args.get("offset", 0)),
                limit=int(args.get("limit", 400)),
            ),
        ),
        ToolSpec(
            name="write_file",
            description="Write a text file inside the project. Cannot write benchmark/.",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
            handler=lambda args: _write_file_tool(project_root, args),
        ),
        ToolSpec(
            name="list_dir",
            description=(
                "List entry names of a directory inside the project. Large directories (e.g. "
                "dataset folders) are capped at 200 entries and shown as a by-extension "
                "breakdown; pass summary=true to force that breakdown for any directory."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "default": "."},
                    "limit": {
                        "type": "integer",
                        "description": "Max entries to list before summarizing (default 200, max 200).",
                    },
                    "summary": {
                        "type": "boolean",
                        "description": "Force a by-extension count summary instead of listing names.",
                    },
                },
            },
            handler=lambda args: format_listing(
                list_dir(project_root, args.get("path", ".")),
                limit=int(args.get("limit", 200)),
                summary=bool(args.get("summary", False)),
            ),
        ),
    ]


def _write_file_tool(project_root: Path, args: dict[str, Any]) -> str:
    write_file(project_root, args["path"], args["content"])
    return f"wrote {args['path']}"


# Default budgets for agent-invoked scripts. The harness owns these — the agent
# cannot raise them beyond what the tool allows (it can only request lower).
_DEFAULT_RUN_TIMEOUT_S = 120.0
_MAX_RUN_TIMEOUT_S = 600.0


def build_run_python_tool(
    project_root: Path, on_event: Callable[[dict[str, Any]], None] | None = None
) -> ToolSpec:
    """Tool to run a training script in the sandboxed, budgeted subprocess.

    Delegates to ``metis.sandbox.run_python``, which confines the script to the
    project dir, blocks ``benchmark/`` at runtime, and enforces time/memory caps.
    If ``on_event`` is given, each stdout line is emitted as a ``train_output``
    event so the UI can show live epoch/training progress.
    """

    def _on_output(script: str) -> Callable[[str], None]:
        def emit(line: str) -> None:
            if on_event is not None and line.strip():
                on_event({"type": "train_output", "script": script, "line": line})

        return emit

    def handler(args: dict[str, Any]) -> str:
        # Harness-enforced budget gate: refuse to start a run once any declared
        # budget is exhausted. The agent cannot bypass this — it lives here, not
        # in the prompt.
        try:
            from metis.benchmark.budget import compute_budget_status
            from metis.benchmark.store import record_usage

            budget = compute_budget_status(project_root)
            if budget.should_stop:
                return "STOP: resource budget exhausted — " + "; ".join(budget.reasons)
        except Exception:
            record_usage = None  # type: ignore[assignment]

        # Anti-gaming guard: before ANY agent code runs against data/processed, make
        # sure the harness has carved and locked a test holdout. This covers the case
        # where the human dropped pre-processed X.npy/y.npy straight into processed/
        # (skipping ingest_dataset) — the holdout is still removed from training data
        # before the model can see it. Idempotent: a no-op once sealed.
        seal_note = ""
        try:
            from metis.benchmark import ensure_holdout_sealed

            seal_marker = project_root / "benchmark" / "holdout" / "seal.yaml"
            was_sealed = seal_marker.exists()
            ensure_holdout_sealed(project_root)
            if not was_sealed and seal_marker.exists():
                seal_note = (
                    "[harness] sealed a test holdout from data/processed before running "
                    "(removed from training data; you can never read it).\n"
                )
        except Exception:
            pass  # never block a run on the seal guard

        timeout = float(args.get("timeout_s", _DEFAULT_RUN_TIMEOUT_S))
        timeout = max(1.0, min(timeout, _MAX_RUN_TIMEOUT_S))
        memory_mb = args.get("memory_mb")
        try:
            result = run_python(
                project_root,
                args["script"],
                timeout_s=timeout,
                memory_mb=int(memory_mb) if memory_mb is not None else None,
                on_output=_on_output(str(args["script"])),
            )
        except Exception as exc:
            return f"error: {exc}"
        if record_usage is not None:
            try:
                bench_dir = project_root / "benchmark"
                bench_dir.mkdir(parents=True, exist_ok=True)
                record_usage(
                    bench_dir,
                    kind="run_python",
                    variant_id=str(args["script"]),
                    wall_clock_s=result.duration_s,
                    detail="agent run_python",
                )
            except Exception:
                pass
        status = "timed out" if result.timed_out else f"exit {result.exit_code}"
        parts = [seal_note + f"{args['script']}: {status} in {result.duration_s:.1f}s"]
        if result.stdout.strip():
            parts.append("stdout:\n" + result.stdout.strip()[-2000:])
        if result.stderr.strip():
            parts.append("stderr:\n" + result.stderr.strip()[-2000:])
        return "\n".join(parts)

    return ToolSpec(
        name="run_python",
        description=(
            "Run a Python script that already exists inside the project (e.g. a train.py "
            "you wrote) in a sandboxed subprocess with a wall-clock timeout and optional "
            "memory cap. Confined to the project directory; cannot read or write benchmark/. "
            "Returns exit status plus captured stdout/stderr."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "script": {
                    "type": "string",
                    "description": "Path to the script inside the project, e.g. models/logreg/train.py.",
                },
                "timeout_s": {
                    "type": "number",
                    "description": f"Wall-clock budget in seconds (default {int(_DEFAULT_RUN_TIMEOUT_S)}, "
                    f"max {int(_MAX_RUN_TIMEOUT_S)}).",
                },
                "memory_mb": {
                    "type": "integer",
                    "description": "Optional address-space cap in MB.",
                },
            },
            "required": ["script"],
        },
        handler=handler,
    )


_PROJECT_SPEC_SCHEMA = ProjectSpec.model_json_schema()


def build_benchmark_tools(project_root: Path) -> list[ToolSpec]:
    """Agent-facing benchmark tools: submit a variant for scoring, read the leaderboard.

    These call the harness-side ``BenchmarkRunner`` directly — the agent submits
    a variant_id and receives only the returned scores. It never touches
    benchmark/ itself (the lockbox blocks that at the sandbox layer).
    """
    from metis.benchmark import (
        BenchmarkRunner,
        compute_budget_status,
        get_failed_variants,
        is_plateaued,
        prune_project,
        ranked_leaderboard,
    )

    def _submit(args: dict[str, Any]) -> str:
        variant_id = args["variant_id"]
        runner = BenchmarkRunner()
        record = runner.run(project_root, variant_id)
        if record.error and record.task_metric_value is None:
            return f"benchmark error for {variant_id!r}: {record.error}"
        parts = [f"benchmarked {variant_id!r}"]
        if record.task_metric_value is not None:
            parts.append(f"{record.task_metric_name}={record.task_metric_value:.4f}")
        if record.model_size_mb is not None:
            parts.append(f"size={record.model_size_mb:.3f} MB")
        if record.param_count is not None:
            parts.append(f"params={record.param_count:,}")
        if record.latency_ms_p50 is not None:
            parts.append(f"p50={record.latency_ms_p50:.3f} ms")
        if record.throughput_sps is not None:
            parts.append(f"throughput={record.throughput_sps:,.0f} samp/s")
        if record.error:
            parts.append(f"warning: {record.error}")
        return ", ".join(parts)

    def _leaderboard(args: dict[str, Any]) -> str:
        n = int(args.get("n", 10))
        include_pruned = bool(args.get("include_pruned", False))
        try:
            rows = ranked_leaderboard(project_root, n=n, include_pruned=include_pruned)
        except Exception as exc:
            return f"error fetching leaderboard: {exc}"
        if not rows:
            return "No benchmark results yet."
        metric_col = rows[0]["task_metric_name"]
        header = (
            f"{'Rank':>4}  {'Variant':<20}  {str(metric_col):>10}  {'Params':>10}  "
            f"{'Size MB':>8}  {'p50 ms':>8}  {'p95 ms':>8}  {'samp/s':>10}  "
            f"{'Pareto':>6}  {'Status':>7}"
        )
        lines = [header, "-" * len(header)]
        for i, r in enumerate(rows, 1):
            mv = f"{r['task_metric_value']:.4f}" if r["task_metric_value"] is not None else "N/A"
            sz = f"{r['model_size_mb']:.3f}" if r["model_size_mb"] is not None else "N/A"
            pc = f"{r['param_count']:,}" if r["param_count"] is not None else "N/A"
            p50 = f"{r['latency_ms_p50']:.3f}" if r["latency_ms_p50"] is not None else "N/A"
            p95 = f"{r['latency_ms_p95']:.3f}" if r["latency_ms_p95"] is not None else "N/A"
            tp = f"{r['throughput_sps']:,.0f}" if r["throughput_sps"] is not None else "N/A"
            pr = str(r.get("pareto_rank", "N/A"))
            status = "pruned" if r.get("pruned") else "active"
            lines.append(
                f"{i:>4}  {str(r['variant_id']):<20}  {mv:>10}  {pc:>10}  "
                f"{sz:>8}  {p50:>8}  {p95:>8}  {tp:>10}  {pr:>6}  {status:>7}"
            )
        return "\n".join(lines)

    def _prune(args: dict[str, Any]) -> str:
        keep_top_k = args.get("keep_top_k")
        drop_bottom_fraction = args.get("drop_bottom_fraction")
        try:
            pruned = prune_project(
                project_root,
                keep_top_k=int(keep_top_k) if keep_top_k is not None else None,
                drop_bottom_fraction=(
                    float(drop_bottom_fraction) if drop_bottom_fraction is not None else None
                ),
                reason="requested by agent",
            )
        except Exception as exc:
            return f"error pruning: {exc}"
        if not pruned:
            return "Nothing pruned (no policy configured, or only the top variant remains)."
        return f"Pruned {len(pruned)} variant(s): {', '.join(pruned)}"

    def _budget(_args: dict[str, Any]) -> str:
        try:
            status = compute_budget_status(project_root)
        except Exception as exc:
            return f"error reading budget: {exc}"

        def rem(v: object) -> str:
            return "unlimited" if v is None else f"{v}"

        lines = [
            f"wall-clock: {status.wall_clock_minutes_used:.2f} min used, "
            f"remaining {rem(status.wall_clock_minutes_remaining)}",
            f"variants trained: {status.variants_trained}, "
            f"remaining {rem(status.variants_remaining)}",
            f"dollars: ${status.dollars_used:.2f} used, remaining {rem(status.dollars_remaining)}",
            f"STOP: {status.should_stop}",
        ]
        if status.reasons:
            lines.append("reasons: " + "; ".join(status.reasons))
        return "\n".join(lines)

    def _failed(args: dict[str, Any]) -> str:
        n = int(args.get("n", 25))
        try:
            rows = get_failed_variants(project_root / "benchmark", n=n)
        except Exception as exc:
            return f"error fetching failed variants: {exc}"
        if not rows:
            return "No failed/errored variants recorded."
        lines = ["Failed variants (benchmark errored — not on the leaderboard):"]
        for r in rows:
            err = str(r.get("error") or "").splitlines()
            lines.append(f"  {r['variant_id']}: {err[0] if err else 'unknown error'}")
        return "\n".join(lines)

    def _plateau(_args: dict[str, Any]) -> str:
        try:
            plateaued = is_plateaued(project_root)
        except Exception as exc:
            return f"error checking plateau: {exc}"
        return (
            "Leaderboard has PLATEAUED — branch out (mutate top performers / try new families)."
            if plateaued
            else "Still improving — keep refining the current leaders."
        )

    return [
        ToolSpec(
            name="submit_for_benchmark",
            description=(
                "Submit a trained model variant for harness-side evaluation on the sealed holdout. "
                "Pass variant_id (the folder name under models/). "
                "The harness evaluates the model — you receive only the scores back. "
                "The agent never sees holdout data, ensuring the benchmark cannot be gamed."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "variant_id": {
                        "type": "string",
                        "description": "Folder name under models/ for this trained variant.",
                    }
                },
                "required": ["variant_id"],
            },
            handler=_submit,
        ),
        ToolSpec(
            name="get_leaderboard",
            description=(
                "Get the current ranked leaderboard from results.db. Ranking follows the "
                "project's rank_objective (single accuracy, weighted sum, or Pareto frontier). "
                "Each row shows accuracy + efficiency columns plus its Pareto rank and "
                "active/pruned status. Pruned variants are hidden unless include_pruned is set."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "n": {
                        "type": "integer",
                        "description": "Maximum rows to return (default 10).",
                    },
                    "include_pruned": {
                        "type": "boolean",
                        "description": "Include variants previously pruned (default false).",
                    },
                },
            },
            handler=_leaderboard,
        ),
        ToolSpec(
            name="request_prune",
            description=(
                "Ask the harness to prune the weakest variants from the active search. "
                "Pruning only MARKS variants (reversible; recipes/weights are preserved) and "
                "removes them from the default leaderboard. Pass keep_top_k to keep only the "
                "best k, or drop_bottom_fraction to drop the worst fraction; omit both to use "
                "the project's configured prune_policy."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "keep_top_k": {
                        "type": "integer",
                        "description": "Keep only the top-k ranked variants; prune the rest.",
                    },
                    "drop_bottom_fraction": {
                        "type": "number",
                        "description": "Fraction (0-1) of the worst variants to prune.",
                    },
                },
            },
            handler=_prune,
        ),
        ToolSpec(
            name="get_budget_status",
            description=(
                "Get cumulative resource usage (wall-clock time, variants trained, estimated $) "
                "versus the project's declared budgets, and whether the search must STOP. "
                "Budgets are enforced by the harness, not by you."
            ),
            input_schema={"type": "object", "properties": {}},
            handler=_budget,
        ),
        ToolSpec(
            name="get_failed_variants",
            description=(
                "List variants whose benchmark run errored (e.g. training crashed, no model.py). "
                "These are NOT shown on the leaderboard because they have no score. Check this if "
                "a variant you trained never appears in get_leaderboard, to see why it failed."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "n": {"type": "integer", "description": "Max failures to return (default 25)."}
                },
            },
            handler=_failed,
        ),
        ToolSpec(
            name="check_plateau",
            description=(
                "Ask the harness whether the leaderboard's best objective has plateaued over "
                "the last N benchmark rounds (per the project's plateau policy). When it has, "
                "branch out: mutate top performers or introduce new model families."
            ),
            input_schema={"type": "object", "properties": {}},
            handler=_plateau,
        ),
    ]


def build_data_tools(project_root: Path) -> list[ToolSpec]:
    """Agent-facing DATA-step tool: ingest the **human-provided** dataset.

    Metis does not auto-source or scrape data — the human places the dataset under
    ``data/raw/<dataset>/``. ``ingest_dataset`` de-dups, validates and splits it; the
    TEST split is sealed harness-side into benchmark/holdout, so the agent only ever
    sees train/val and the returned summary never contains holdout samples.
    """

    def _ingest(args: dict[str, Any]) -> str:
        dataset_id = str(args["dataset"])
        dataset_dir = project_root / "data" / "raw" / dataset_id
        try:
            from metis.data_sources import ingest_dataset
            from metis.projects import load_project

            spec = load_project(project_root)
            split = spec.data.split
            result = ingest_dataset(
                project_root,
                dataset_dir,
                train=split.train,
                val=split.val,
                test=split.test,
                seed=spec.data.split_seed,
                classes=spec.classes,
            )
        except Exception as exc:
            log_action(project_root, "ingest_dataset", args, ok=False, error=str(exc))
            return f"error: {exc}"
        log_action(project_root, "ingest_dataset", args, ok=True)
        # NOTE: only sizes + the validation report are returned. The test split is
        # sealed in benchmark/holdout and is never included here.
        return (
            f"ingested {dataset_id!r}: train={result.train_size}, val={result.val_size}, "
            f"test sealed={result.test_size} (in lockbox, not returned)\n" + result.report.summary()
        )

    return [
        ToolSpec(
            name="ingest_dataset",
            description=(
                "De-dup, validate and auto train/val/test split the HUMAN-PROVIDED dataset "
                "the human placed under data/raw/<dataset>/. Train/val land in "
                "data/processed/; the harness seals the TEST split into the locked "
                "benchmark/holdout/ that you can never read. Returns split sizes + a "
                "validation/class-balance report only. (Metis does not source or scrape "
                "data — if data/raw/ is empty, ask the human to provide their dataset.)"
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "dataset": {
                        "type": "string",
                        "description": "Dataset folder name under data/raw/.",
                    }
                },
                "required": ["dataset"],
            },
            handler=_ingest,
        ),
    ]


def build_template_tools(project_root: Path) -> list[ToolSpec]:
    """PROPOSE/TRAIN templates: let the agent pick a proven model family to scaffold.

    Instead of hand-writing train.py/model.py (slow, token-heavy, and easy to get
    wrong — e.g. importing torch when it isn't installed), the agent lists the
    prebuilt families and instantiates one. Scaffolding routes through the
    lockbox-enforced ``scaffold_candidate`` (toy.py), so safety/audit are inherited.
    """
    from metis.training import toy, zoo

    def _list(_args: dict[str, Any]) -> str:
        lines = ["Prebuilt model templates. instantiate_template one of these by key:"]
        lines.append("\n[sklearn — flat/tabular feature data, X.npy shaped [N, features]]")
        for key, spec in toy.FAMILIES.items():
            grid = ", ".join(f"{k}={v}" for k, v in spec.hparam_grid.items())
            lines.append(f"  {key}: {spec.family} — tunable: {grid}")
        lines.append(
            "\n[torch image models — X.npy shaped [N, H, W, C]; require the 'ml' extra "
            "(torch+torchvision) installed; give run_python a higher memory_mb]"
        )
        for key, tspec in zoo.TORCH_FAMILIES.items():
            grid = ", ".join(f"{k}={v}" for k, v in tspec.hparam_grid.items())
            dl = " (downloads pretrained weights)" if tspec.needs_download else ""
            lines.append(
                f"  {key}: {tspec.description}{dl}\n      tunable: {grid}; "
                f"suggested memory_mb≥{tspec.min_memory_mb}"
            )
        return "\n".join(lines)

    def _instantiate(args: dict[str, Any]) -> str:
        template = str(args["template"])
        variant_id = str(args["variant_id"])
        overrides = dict(args.get("hparams") or {})
        try:
            if template in toy.FAMILIES:
                spec = toy.FAMILIES[template]
                hp = {**spec.default_hparams, **overrides}
                candidate = toy.build_candidate(spec, hp, variant_id)
                toy.scaffold_candidate(project_root, candidate)
                note = "run it with run_python (numpy + scikit-learn)."
            elif template in zoo.TORCH_FAMILIES:
                tspec = zoo.TORCH_FAMILIES[template]
                hp = {**tspec.default_hparams, **overrides}
                candidate = zoo.build_torch_candidate(tspec, hp, variant_id)
                toy.scaffold_candidate(project_root, candidate)
                note = (
                    f"run it with run_python (needs torch+torchvision from the 'ml' extra; "
                    f"pass memory_mb≥{tspec.min_memory_mb}). Expects image data shaped "
                    f"[N, H, W, C] in data/processed/X.npy."
                )
            else:
                known = ", ".join([*toy.FAMILIES, *zoo.TORCH_FAMILIES])
                return f"error: unknown template {template!r}. Available: {known}"
        except Exception as exc:
            return f"error scaffolding template: {exc}"
        return f"scaffolded models/{variant_id}/ (train.py + model.py) from {template}; {note}"

    return [
        ToolSpec(
            name="list_model_templates",
            description=(
                "List the prebuilt model families you can scaffold instead of hand-writing "
                "train.py. Covers light sklearn families (tabular/flat data) and torch image "
                "models (CNNs). Prefer these over writing model code from scratch."
            ),
            input_schema={"type": "object", "properties": {}},
            handler=_list,
        ),
        ToolSpec(
            name="instantiate_template",
            description=(
                "Scaffold a chosen model template into models/<variant_id>/ (writes train.py "
                "and model.py for you). Pass template (a key from list_model_templates), "
                "variant_id, and optional hparams overrides. Then run it with run_python and "
                "submit_for_benchmark — no need to write the architecture yourself."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "template": {
                        "type": "string",
                        "description": "Template key from list_model_templates (e.g. logreg, tiny_cnn).",
                    },
                    "variant_id": {
                        "type": "string",
                        "description": "Folder name to create under models/ for this variant.",
                    },
                    "hparams": {
                        "type": "object",
                        "description": "Optional hyperparameter overrides (defaults used otherwise).",
                    },
                },
                "required": ["template", "variant_id"],
            },
            handler=_instantiate,
        ),
    ]


def build_define_tool(project_root: Path) -> ToolSpec:
    """The DEFINE-step tool: turns structured arguments into a validated project.yaml."""

    def handler(args: dict[str, Any]) -> str:
        try:
            spec = ProjectSpec.model_validate(args)
        except Exception as exc:  # surfaced to the model so it can retry
            return f"error: invalid project spec: {exc}"
        if project_root.exists():
            write_project_yaml(project_root, spec)
        else:
            create_project(project_root, spec)
        return f"saved project.yaml for {spec.name!r}"

    return ToolSpec(
        name="save_project_spec",
        description=(
            "Save the project definition as project.yaml. Call this once you have enough "
            "information from the human's task description to fill out all required fields."
        ),
        input_schema=_PROJECT_SPEC_SCHEMA,
        handler=handler,
    )


def build_agent_tools(
    project_root: Path, on_event: Callable[[dict[str, Any]], None] | None = None
) -> list[ToolSpec]:
    """The full toolset for the interactive agent loop driving a project end-to-end.

    Bundles every capability the agent needs across the loop: scoped file I/O and
    ``run_python`` (lockbox-enforced), ``save_project_spec`` (DEFINE),
    ``ingest_dataset`` (DATA — human-provided only), and the benchmark/leaderboard/
    prune/budget/plateau tools (BENCHMARK→RANK→PRUNE→BRANCH). The ``benchmark/``
    lockbox is enforced underneath, so this set is safe to hand to the model wholesale.

    ``on_event`` is forwarded to ``run_python`` so training stdout streams to the UI.
    """
    return [
        *build_sandbox_tools(project_root),
        build_run_python_tool(project_root, on_event=on_event),
        build_define_tool(project_root),
        *build_data_tools(project_root),
        *build_template_tools(project_root),
        *build_benchmark_tools(project_root),
    ]
