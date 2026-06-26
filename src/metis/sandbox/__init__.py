"""Sandbox tool layer: the only surface the agent uses to touch a project.

Re-exports the scoped file tools and the lockbox violation type so callers can
``from metis.sandbox import read_file, write_file, list_dir, LockboxViolation``.
"""

from __future__ import annotations

from metis.sandbox.lockbox import LockboxViolation
from metis.sandbox.pyrunner import RunResult, run_python
from metis.sandbox.runlog import read_actions
from metis.sandbox.tools import list_dir, read_file, write_file

__all__ = [
    "LockboxViolation",
    "read_file",
    "write_file",
    "list_dir",
    "read_actions",
    "run_python",
    "RunResult",
]
