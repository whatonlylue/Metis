"""Scoped file tools exposed to the agent.

Every path passed in here is resolved through ``lockbox.resolve_within_project``
first, so these functions inherit the sandbox and lockbox guarantees: no escaping
the project root, no touching the sealed ``benchmark/`` directory.
"""

from __future__ import annotations

from pathlib import Path

from metis.sandbox.lockbox import resolve_within_project


def read_file(project_root: Path, path: str | Path) -> str:
    """Read a text file scoped to ``project_root``."""
    target = resolve_within_project(project_root, path)
    return target.read_text()


def write_file(project_root: Path, path: str | Path, content: str) -> None:
    """Write a text file scoped to ``project_root``, creating parent dirs as needed."""
    target = resolve_within_project(project_root, path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)


def list_dir(project_root: Path, path: str | Path = ".") -> list[str]:
    """List entry names of a directory scoped to ``project_root``."""
    target = resolve_within_project(project_root, path)
    if not target.is_dir():
        raise NotADirectoryError(f"Not a directory: {path!r}")
    return sorted(entry.name for entry in target.iterdir())
