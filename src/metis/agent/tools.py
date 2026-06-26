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
