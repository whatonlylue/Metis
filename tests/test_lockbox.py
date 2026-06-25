"""Tests proving the lockbox blocks reads/writes/traversal into benchmark/."""

from __future__ import annotations

from pathlib import Path

import pytest

from metis.sandbox.lockbox import LockboxViolation, resolve_within_project


@pytest.fixture
def project_root(tmp_path: Path) -> Path:
    (tmp_path / "benchmark" / "holdout").mkdir(parents=True)
    (tmp_path / "benchmark" / "results.db").write_text("secret")
    (tmp_path / "models").mkdir()
    return tmp_path


def test_resolves_normal_path(project_root: Path) -> None:
    target = resolve_within_project(project_root, "models/foo.txt")
    assert target == (project_root / "models" / "foo.txt").resolve()


def test_resolves_project_root_itself(project_root: Path) -> None:
    assert resolve_within_project(project_root, ".") == project_root.resolve()


def test_rejects_benchmark_dir(project_root: Path) -> None:
    with pytest.raises(LockboxViolation):
        resolve_within_project(project_root, "benchmark")


def test_rejects_benchmark_subpath(project_root: Path) -> None:
    with pytest.raises(LockboxViolation):
        resolve_within_project(project_root, "benchmark/results.db")
    with pytest.raises(LockboxViolation):
        resolve_within_project(project_root, "benchmark/holdout/secret.npy")


def test_rejects_traversal_escape(project_root: Path) -> None:
    with pytest.raises(LockboxViolation):
        resolve_within_project(project_root, "../outside.txt")


def test_rejects_traversal_into_benchmark(project_root: Path) -> None:
    with pytest.raises(LockboxViolation):
        resolve_within_project(project_root, "models/../benchmark/results.db")


def test_rejects_symlink_escape(project_root: Path, tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside_secret.txt"
    outside.write_text("nope")
    link = project_root / "escape_link"
    link.symlink_to(outside)
    with pytest.raises(LockboxViolation):
        resolve_within_project(project_root, "escape_link")


def test_rejects_symlink_into_benchmark(project_root: Path) -> None:
    link = project_root / "sneaky"
    link.symlink_to(project_root / "benchmark")
    with pytest.raises(LockboxViolation):
        resolve_within_project(project_root, "sneaky/results.db")
