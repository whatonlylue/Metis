"""Scoped file tools exposed to the agent.

Every path passed in here is resolved through ``lockbox.resolve_within_project``
first, so these functions inherit the sandbox and lockbox guarantees: no escaping
the project root, no touching the sealed ``benchmark/`` directory.

Every call (success or failure) is also appended to ``runs/actions.jsonl`` via
``runlog.log_action``, so the human/harness has a full audit trail of agent activity.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from metis.sandbox.lockbox import resolve_within_project
from metis.sandbox.runlog import log_action

# Harness-owned caps so a single read/list cannot flood the agent's context window
# (and stay in its history for the rest of the session). The agent can page through
# larger files/dirs via the tool arguments, but never raise these ceilings.
READ_MAX_LINES = 400
READ_MAX_CHARS = 50_000
LIST_MAX_ENTRIES = 200


def read_file(
    project_root: Path,
    path: str | Path,
    *,
    offset: int = 0,
    limit: int = READ_MAX_LINES,
) -> str:
    """Read a text file scoped to ``project_root``, windowed to keep context small.

    Returns the ``offset:offset+limit`` line window, capped at ``READ_MAX_CHARS``.
    When the whole file fits within the caps and ``offset`` is 0, the exact original
    text is returned unchanged; otherwise a one-line footer notes what was shown.
    """
    try:
        target = resolve_within_project(project_root, path)
        content = target.read_text()
    except Exception as exc:
        log_action(project_root, "read_file", {"path": str(path)}, ok=False, error=str(exc))
        raise
    log_action(project_root, "read_file", {"path": str(path)}, ok=True)

    lines = content.splitlines(keepends=True)
    total = len(lines)
    limit = max(1, min(int(limit), READ_MAX_LINES))
    offset = max(0, int(offset))
    if offset == 0 and total <= limit and len(content) <= READ_MAX_CHARS:
        return content

    window = lines[offset : offset + limit]
    body = "".join(window)
    char_capped = len(body) > READ_MAX_CHARS
    if char_capped:
        body = body[:READ_MAX_CHARS]
    shown_end = offset + len(window)
    note = "; char-capped" if char_capped else ""
    footer = f"\n\n[showing lines {offset + 1}-{shown_end} of {total}{note}]"
    return body + footer


def format_listing(
    entries: list[str], *, limit: int = LIST_MAX_ENTRIES, summary: bool = False
) -> str:
    """Render directory entries for the agent, bounded so large dirs stay cheap.

    Small dirs render as a plain newline-separated list. When ``summary`` is set or
    the entry count exceeds ``limit``, render a by-extension breakdown plus the first
    ``limit`` names and a ``showing N of M`` footer instead of every filename.
    """
    limit = max(1, min(int(limit), LIST_MAX_ENTRIES))
    total = len(entries)
    if not (summary or total > limit):
        return "\n".join(entries)

    counts: Counter[str] = Counter((Path(name).suffix.lower() or "<no-ext>") for name in entries)
    breakdown = ", ".join(f"{ext}: {n}" for ext, n in counts.most_common())
    head = entries[:limit]
    parts = [
        f"{total} entries — by extension: {breakdown}",
        "\n".join(head),
        f"[showing {len(head)} of {total} entries; pass summary=false with a higher limit "
        f"to see more, max {LIST_MAX_ENTRIES}]",
    ]
    return "\n".join(parts)


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
