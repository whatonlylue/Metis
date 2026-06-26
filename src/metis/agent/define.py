"""DEFINE step: turn a human's task description into a validated ``project.yaml``.

Drives one ``AgentLoop`` run with the sandbox tools plus ``save_project_spec``
(see ``tools.build_define_tool``). The model must call that tool at least once
before this can succeed — there's no other way for ``project.yaml`` to exist.
"""

from __future__ import annotations

from pathlib import Path

from metis.agent.client import LLMClient
from metis.agent.loop import AgentLoop, EventCallback
from metis.agent.tools import build_define_tool, build_sandbox_tools
from metis.projects import load_project
from metis.projects.schema import ProjectSpec

DEFINE_SYSTEM_PROMPT = """You are the DEFINE step of Metis, an agent harness that trains \
small, efficient task-specific models (not LLMs).

The human will describe what they want to classify or predict. Turn that \
description into a complete project definition and save it by calling \
save_project_spec exactly once. Pick sensible defaults for fields the human \
didn't specify (e.g. rank_objective, target_metric). Do not ask the human \
follow-up questions — infer reasonable values and proceed."""


def run_define_step(
    client: LLMClient,
    project_root: Path,
    task_description: str,
    *,
    on_event: EventCallback | None = None,
) -> ProjectSpec:
    """Run the DEFINE step, returning the validated, persisted ``ProjectSpec``."""
    tools = [*build_sandbox_tools(project_root), build_define_tool(project_root)]
    loop = AgentLoop(client=client, system=DEFINE_SYSTEM_PROMPT, tools=tools, on_event=on_event)
    loop.run(task_description)

    if not (project_root / "project.yaml").exists():
        raise RuntimeError("Agent finished without saving a project.yaml")
    return load_project(project_root)
