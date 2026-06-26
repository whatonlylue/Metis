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
from metis.sandbox import list_dir, read_file, run_python, write_file


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
            description="Read a text file inside the project. Cannot read benchmark/.",
            input_schema={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
            handler=lambda args: read_file(project_root, args["path"]),
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
            description="List entry names of a directory inside the project.",
            input_schema={
                "type": "object",
                "properties": {"path": {"type": "string", "default": "."}},
            },
            handler=lambda args: "\n".join(list_dir(project_root, args.get("path", "."))),
        ),
    ]


def _write_file_tool(project_root: Path, args: dict[str, Any]) -> str:
    write_file(project_root, args["path"], args["content"])
    return f"wrote {args['path']}"


# Default budgets for agent-invoked scripts. The harness owns these — the agent
# cannot raise them beyond what the tool allows (it can only request lower).
_DEFAULT_RUN_TIMEOUT_S = 120.0
_MAX_RUN_TIMEOUT_S = 600.0


def build_run_python_tool(project_root: Path) -> ToolSpec:
    """Tool to run a training script in the sandboxed, budgeted subprocess.

    Delegates to ``metis.sandbox.run_python``, which confines the script to the
    project dir, blocks ``benchmark/`` at runtime, and enforces time/memory caps.
    """

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

        timeout = float(args.get("timeout_s", _DEFAULT_RUN_TIMEOUT_S))
        timeout = max(1.0, min(timeout, _MAX_RUN_TIMEOUT_S))
        memory_mb = args.get("memory_mb")
        try:
            result = run_python(
                project_root,
                args["script"],
                timeout_s=timeout,
                memory_mb=int(memory_mb) if memory_mb is not None else None,
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
        parts = [f"{args['script']}: {status} in {result.duration_s:.1f}s"]
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
