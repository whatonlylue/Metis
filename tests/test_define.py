"""Tests for the DEFINE step: a fake LLMClient drives run_define_step."""

from __future__ import annotations

from pathlib import Path

import pytest

from metis.agent.client import AgentMessage, LLMClient, ToolCall
from metis.agent.define import run_define_step


class OneShotDefineClient(LLMClient):
    """Calls save_project_spec on its first turn, then stops."""

    def __init__(self, spec_args: dict) -> None:
        self._spec_args = spec_args
        self._calls = 0

    def send(self, messages, tools, system) -> AgentMessage:
        self._calls += 1
        if self._calls == 1:
            call = ToolCall(id="call-1", name="save_project_spec", input=self._spec_args)
            return AgentMessage(text="saving now", tool_calls=[call])
        return AgentMessage(text="done", tool_calls=[])


class NeverSavesClient(LLMClient):
    def send(self, messages, tools, system) -> AgentMessage:
        return AgentMessage(text="thinking but never saving", tool_calls=[])


def test_run_define_step_returns_validated_spec(tmp_path: Path) -> None:
    root = tmp_path / "flowers-5"
    client = OneShotDefineClient(
        {
            "name": "flowers-5",
            "description": "Classify a photo as one of five flower species.",
            "task_type": "image_classification",
            "classes": ["daisy", "dandelion", "rose", "sunflower", "tulip"],
        }
    )

    spec = run_define_step(client, root, "I want to classify flower photos.")

    assert spec.name == "flowers-5"
    assert spec.classes == ["daisy", "dandelion", "rose", "sunflower", "tulip"]
    assert (root / "project.yaml").exists()


def test_run_define_step_raises_if_never_saved(tmp_path: Path) -> None:
    root = tmp_path / "ghost-project"
    client = NeverSavesClient()

    with pytest.raises(RuntimeError):
        run_define_step(client, root, "I want to classify something.")


def test_run_define_step_emits_events(tmp_path: Path) -> None:
    root = tmp_path / "flowers-5"
    client = OneShotDefineClient(
        {
            "name": "flowers-5",
            "description": "x",
            "task_type": "regression",
        }
    )
    events = []

    run_define_step(client, root, "go", on_event=events.append)

    tool_calls = [e for e in events if e["type"] == "tool_call"]
    assert any(e["name"] == "save_project_spec" for e in tool_calls)
