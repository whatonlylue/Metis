"""Scoped file tools exposed to the agent.

Every path passed in here is resolved through ``lockbox.resolve_within_project``
first, so these functions inherit the sandbox and lockbox guarantees: no escaping
the project root, no touching the sealed ``benchmark/`` directory.

Every call (success or failure) is also appended to ``runs/actions.jsonl`` via
``runlog.log_action``, so the human/harness has a full audit trail of agent activity.
"""

from __future__ import annotations

from pathlib import Path

from metis.sandbox.lockbox import resolve_within_project
from metis.sandbox.runlog import log_action


def read_file(project_root: Path, path: str | Path) -> str:
    """Read a text file scoped to ``project_root``."""
    try:
        target = resolve_within_project(project_root, path)
        content = target.read_text()
    except Exception as exc:
        log_action(project_root, "read_file", {"path": str(path)}, ok=False, error=str(exc))
        raise
    log_action(project_root, "read_file", {"path": str(path)}, ok=True)
    return content


def write_file(project_root: Path, path: str | Path, content: str) -> None:
    """Write a text file scoped to ``project_root``, creating parent dirs as needed."""
    try:
        target = resolve_within_project(project_root, path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    except Exception as exc:
        log_action(project_root, "write_file", {"path": str(path)}, ok=False, error=str(exc))
        raise
    log_action(project_root, "write_file", {"path": str(path)}, ok=True)


def list_dir(project_root: Path, path: str | Path = ".") -> list[str]:
    """List entry names of a directory scoped to ``project_root``."""
    try:
        target = resolve_within_project(project_root, path)
        if not target.is_dir():
            raise NotADirectoryError(f"Not a directory: {path!r}")
        entries = sorted(entry.name for entry in target.iterdir())
    except Exception as exc:
        log_action(project_root, "list_dir", {"path": str(path)}, ok=False, error=str(exc))
        raise
    log_action(project_root, "list_dir", {"path": str(path)}, ok=True)
    return entries
