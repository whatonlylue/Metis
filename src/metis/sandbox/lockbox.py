"""Lockbox enforcement — the load-bearing safety property of Metis.

The agent only ever touches a project through the sandbox tool layer. This module
is the gate: it canonicalizes every path the agent asks for and refuses anything
that resolves inside ``<project>/benchmark/``. The agent can therefore never read
the sealed holdout, read the scoring code, or edit recorded results.

Enforced here in code — NOT by prompt instructions to the agent.
"""

from __future__ import annotations

from pathlib import Path

SEALED_DIRNAME = "benchmark"


class LockboxViolation(PermissionError):
    """Raised when the agent attempts to touch the sealed benchmark area."""


def resolve_within_project(project_root: Path, requested: str | Path) -> Path:
    """Resolve ``requested`` against the project root and enforce two invariants:

    1. The path stays inside the project (no ``..``/symlink escape).
    2. The path is not inside the sealed ``benchmark/`` lockbox.

    Returns the canonical path on success; raises on violation.
    """
    root = project_root.resolve()
    # Resolve symlinks and ``..`` so traversal tricks can't escape the checks.
    target = (root / requested).resolve()

    if root not in target.parents and target != root:
        raise LockboxViolation(f"Path escapes project sandbox: {requested!r}")

    sealed = (root / SEALED_DIRNAME).resolve()
    if target == sealed or sealed in target.parents:
        raise LockboxViolation(
            f"{SEALED_DIRNAME}/ is sealed; the agent cannot read or write it: {requested!r}"
        )

    return target
