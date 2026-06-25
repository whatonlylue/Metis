"""Tests for the project store: scaffolding + project.yaml validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from metis.projects import SUBDIRS, create_project, load_project
from metis.projects.schema import ProjectSpec, TaskType


def test_create_project_scaffolds_tree_and_yaml(tmp_path: Path) -> None:
    root = tmp_path / "flowers-5"
    spec = ProjectSpec(
        name="flowers-5",
        description="Classify a photo as one of five flower species.",
        task_type=TaskType.image_classification,
        classes=["daisy", "dandelion", "rose", "sunflower", "tulip"],
    )

    create_project(root, spec)

    for sub in SUBDIRS:
        assert (root / sub).is_dir()
    assert (root / "project.yaml").exists()

    loaded = load_project(root)
    assert loaded == spec


def test_create_project_rejects_existing_dir(tmp_path: Path) -> None:
    root = tmp_path / "dup"
    root.mkdir()
    spec = ProjectSpec(name="dup", description="x", task_type=TaskType.regression)

    with pytest.raises(FileExistsError):
        create_project(root, spec)


def test_load_project_validates_schema(tmp_path: Path) -> None:
    root = tmp_path / "bad"
    root.mkdir()
    (root / "project.yaml").write_text("name: bad\ndescription: x\ntask_type: not_a_real_type\n")

    with pytest.raises(ValueError):
        load_project(root)
