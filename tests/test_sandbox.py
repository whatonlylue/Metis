"""Tests for the scoped file tools (read_file/write_file/list_dir)."""

from __future__ import annotations

from pathlib import Path

import pytest

from metis.sandbox import LockboxViolation, list_dir, read_file, write_file


@pytest.fixture
def project_root(tmp_path: Path) -> Path:
    (tmp_path / "benchmark" / "holdout").mkdir(parents=True)
    (tmp_path / "benchmark" / "results.db").write_text("secret-results")
    (tmp_path / "benchmark" / "holdout" / "test.npy").write_text("secret-holdout")
    (tmp_path / "models").mkdir()
    return tmp_path


def test_write_then_read_roundtrip(project_root: Path) -> None:
    write_file(project_root, "models/variant-1/recipe.yaml", "arch: cnn\n")
    assert read_file(project_root, "models/variant-1/recipe.yaml") == "arch: cnn\n"


def test_write_creates_parent_dirs(project_root: Path) -> None:
    write_file(project_root, "models/variant-2/weights/model.bin", "weights")
    assert (project_root / "models" / "variant-2" / "weights" / "model.bin").exists()


def test_list_dir_lists_entries(project_root: Path) -> None:
    write_file(project_root, "models/variant-1/recipe.yaml", "arch: cnn\n")
    entries = list_dir(project_root, "models")
    assert entries == ["variant-1"]


def test_list_dir_root_shows_benchmark_name_but_not_contents(project_root: Path) -> None:
    # The agent can see that benchmark/ exists as a name at the root...
    entries = list_dir(project_root, ".")
    assert "benchmark" in entries
    # ...but cannot list into it.
    with pytest.raises(LockboxViolation):
        list_dir(project_root, "benchmark")


def test_read_file_blocked_inside_benchmark(project_root: Path) -> None:
    with pytest.raises(LockboxViolation):
        read_file(project_root, "benchmark/results.db")
    with pytest.raises(LockboxViolation):
        read_file(project_root, "benchmark/holdout/test.npy")


def test_write_file_blocked_inside_benchmark(project_root: Path) -> None:
    with pytest.raises(LockboxViolation):
        write_file(project_root, "benchmark/results.db", "tampered")
    # The original content must be untouched.
    assert (project_root / "benchmark" / "results.db").read_text() == "secret-results"


def test_write_file_blocked_via_traversal_into_benchmark(project_root: Path) -> None:
    with pytest.raises(LockboxViolation):
        write_file(project_root, "models/../benchmark/results.db", "tampered")
