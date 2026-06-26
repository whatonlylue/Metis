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
from metis.sandbox import list_dir, read_file, write_file


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


_PROJECT_SPEC_SCHEMA = ProjectSpec.model_json_schema()


def build_benchmark_tools(project_root: Path) -> list[ToolSpec]:
    """Agent-facing benchmark tools: submit a variant for scoring, read the leaderboard.

    These call the harness-side ``BenchmarkRunner`` directly — the agent submits
    a variant_id and receives only the returned scores. It never touches
    benchmark/ itself (the lockbox blocks that at the sandbox layer).
    """
    from metis.benchmark import BenchmarkRunner, get_leaderboard
    from metis.projects import load_project

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
        if record.error:
            parts.append(f"warning: {record.error}")
        return ", ".join(parts)

    def _leaderboard(args: dict[str, Any]) -> str:
        n = int(args.get("n", 10))
        try:
            spec = load_project(project_root)
            rows = get_leaderboard(
                project_root / "benchmark",
                task_metric_name=spec.target_metric,
                n=n,
            )
        except Exception as exc:
            return f"error fetching leaderboard: {exc}"
        if not rows:
            return "No benchmark results yet."
        metric_col = rows[0]["task_metric_name"]
        header = (
            f"{'Rank':>4}  {'Variant':<24}  {str(metric_col):>10}  {'Size MB':>8}  {'Params':>10}"
        )
        lines = [header, "-" * len(header)]
        for i, r in enumerate(rows, 1):
            mv = f"{r['task_metric_value']:.4f}" if r["task_metric_value"] is not None else "N/A"
            sz = f"{r['model_size_mb']:.3f}" if r["model_size_mb"] is not None else "N/A"
            pc = f"{r['param_count']:,}" if r["param_count"] is not None else "N/A"
            lines.append(f"{i:>4}  {str(r['variant_id']):<24}  {mv:>10}  {sz:>8}  {pc:>10}")
        return "\n".join(lines)

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
            description="Get the current ranked leaderboard from results.db.",
            input_schema={
                "type": "object",
                "properties": {
                    "n": {
                        "type": "integer",
                        "description": "Maximum rows to return (default 10).",
                    }
                },
            },
            handler=_leaderboard,
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
