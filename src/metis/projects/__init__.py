"""Project store: scaffold ``projects/<name>/`` trees and validate ``project.yaml``."""

from __future__ import annotations

from pathlib import Path

import yaml

from metis.projects.schema import ProjectSpec

SUBDIRS = [
    "data/raw",
    "data/processed",
    "data/labels",
    "models",
    "benchmark/holdout",
    "runs",
]


def create_project(root: Path, spec: ProjectSpec) -> Path:
    """Scaffold the project directory tree and write a validated ``project.yaml``.

    Raises ``FileExistsError`` if ``root`` already exists.
    """
    if root.exists():
        raise FileExistsError(f"Project already exists: {root}")
    for sub in SUBDIRS:
        (root / sub).mkdir(parents=True, exist_ok=True)
    write_project_yaml(root, spec)
    return root


def write_project_yaml(root: Path, spec: ProjectSpec) -> None:
    """Serialize ``spec`` to ``<root>/project.yaml``."""
    (root / "project.yaml").write_text(
        yaml.safe_dump(spec.model_dump(mode="json"), sort_keys=False)
    )


def load_project(root: Path) -> ProjectSpec:
    """Read and validate ``<root>/project.yaml``."""
    data = yaml.safe_load((root / "project.yaml").read_text())
    return ProjectSpec.model_validate(data)
