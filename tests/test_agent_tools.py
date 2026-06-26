"""Tests for agent tool specs: sandbox wrappers + the DEFINE save_project_spec tool."""

from __future__ import annotations

from pathlib import Path

import pytest

from metis.agent.tools import build_define_tool, build_sandbox_tools
from metis.projects import load_project


@pytest.fixture
def project_root(tmp_path: Path) -> Path:
    (tmp_path / "benchmark" / "holdout").mkdir(parents=True)
    (tmp_path / "benchmark" / "results.db").write_text("secret")
    (tmp_path / "models").mkdir()
    return tmp_path


def _tool(tools, name: str):
    return next(t for t in tools if t.name == name)


def test_write_then_read_round_trip(project_root: Path) -> None:
    tools = build_sandbox_tools(project_root)
    write_result = _tool(tools, "write_file").handler({"path": "models/x.txt", "content": "hi"})
    assert "wrote" in write_result
    assert _tool(tools, "read_file").handler({"path": "models/x.txt"}) == "hi"


def test_list_dir_tool_lists_entries(project_root: Path) -> None:
    tools = build_sandbox_tools(project_root)
    (project_root / "models" / "variant-1").mkdir()
    assert _tool(tools, "list_dir").handler({"path": "models"}) == "variant-1"


def test_sandbox_tool_lockbox_violation_raises(project_root: Path) -> None:
    tools = build_sandbox_tools(project_root)
    with pytest.raises(PermissionError):
        _tool(tools, "read_file").handler({"path": "benchmark/results.db"})


def test_define_tool_saves_valid_spec(project_root: Path) -> None:
    tool = build_define_tool(project_root)
    result = tool.handler(
        {
            "name": "flowers",
            "description": "Classify flower photos.",
            "task_type": "image_classification",
            "classes": ["daisy", "rose"],
        }
    )
    assert "saved project.yaml" in result
    spec = load_project(project_root)
    assert spec.name == "flowers"
    assert spec.classes == ["daisy", "rose"]


def test_define_tool_reports_validation_error_without_raising(project_root: Path) -> None:
    tool = build_define_tool(project_root)
    result = tool.handler({"name": "bad", "task_type": "not_a_real_type"})
    assert result.startswith("error:")
    assert not (project_root / "project.yaml").exists()


def test_define_tool_scaffolds_project_if_missing(tmp_path: Path) -> None:
    root = tmp_path / "new-project"
    tool = build_define_tool(root)
    tool.handler(
        {
            "name": "new-project",
            "description": "x",
            "task_type": "regression",
        }
    )
    assert (root / "models").is_dir()
    assert (root / "benchmark" / "holdout").is_dir()
