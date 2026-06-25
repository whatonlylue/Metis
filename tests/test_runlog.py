"""Tests for runs/ action logging: every sandbox tool call is recorded."""

from __future__ import annotations

from pathlib import Path

import pytest

from metis.sandbox import LockboxViolation, list_dir, read_actions, read_file, write_file


@pytest.fixture
def project_root(tmp_path: Path) -> Path:
    (tmp_path / "benchmark" / "holdout").mkdir(parents=True)
    (tmp_path / "benchmark" / "results.db").write_text("secret-results")
    (tmp_path / "models").mkdir()
    return tmp_path


def test_successful_write_and_read_are_logged(project_root: Path) -> None:
    write_file(project_root, "models/variant-1/recipe.yaml", "arch: cnn\n")
    read_file(project_root, "models/variant-1/recipe.yaml")

    actions = read_actions(project_root)
    assert [a["tool"] for a in actions] == ["write_file", "read_file"]
    assert all(a["ok"] for a in actions)
    assert all(a["args"]["path"] == "models/variant-1/recipe.yaml" for a in actions)
    assert all("timestamp" in a for a in actions)


def test_list_dir_is_logged(project_root: Path) -> None:
    list_dir(project_root, "models")

    actions = read_actions(project_root)
    assert actions[0]["tool"] == "list_dir"
    assert actions[0]["ok"] is True


def test_lockbox_violations_are_logged_as_failures(project_root: Path) -> None:
    with pytest.raises(LockboxViolation):
        read_file(project_root, "benchmark/results.db")

    actions = read_actions(project_root)
    assert actions[0]["tool"] == "read_file"
    assert actions[0]["ok"] is False
    assert "sealed" in actions[0]["error"]


def test_read_actions_empty_when_no_log(project_root: Path) -> None:
    assert read_actions(project_root) == []
