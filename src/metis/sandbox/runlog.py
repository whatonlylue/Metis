"""Action logging: every sandbox tool call is appended to ``runs/actions.jsonl``.

This is an audit trail, not a safety boundary — the lockbox (see ``lockbox.py``)
is what actually stops the agent from touching ``benchmark/``. The log exists so
a human (or the harness) can later answer "what did the agent do and when."
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

RUNS_DIRNAME = "runs"
LOG_FILENAME = "actions.jsonl"


def log_action(
    project_root: Path,
    tool: str,
    args: dict[str, Any],
    *,
    ok: bool,
    error: str | None = None,
) -> None:
    """Append one JSON line recording a sandbox tool call.

    Logging is best-effort scaffolding around the lockbox, so it writes directly
    to ``<project_root>/runs/`` rather than going through ``resolve_within_project``
    — the tool call being logged has already been through that check.
    """
    log_path = project_root / RUNS_DIRNAME / LOG_FILENAME
    log_path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tool": tool,
        "args": args,
        "ok": ok,
        "error": error,
    }
    with log_path.open("a") as f:
        f.write(json.dumps(entry) + "\n")


def read_actions(project_root: Path) -> list[dict[str, Any]]:
    """Read back the logged actions for a project, oldest first."""
    log_path = project_root / RUNS_DIRNAME / LOG_FILENAME
    if not log_path.exists():
        return []
    return [json.loads(line) for line in log_path.read_text().splitlines() if line]
