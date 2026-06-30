"""Tests for AgentSession context hygiene (stale tool_result elision) and
resumable session persistence (history + display feed)."""

from __future__ import annotations

from pathlib import Path

from metis.agent.session import (
    _ELIDE_STUB,
    AgentSession,
    _elide_stale_tool_results,
    load_feed,
    load_history,
    save_feed,
    save_history,
)


def _tr_turn(tool_use_id: str, content: str) -> dict:
    return {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": tool_use_id, "content": content}],
    }


def test_recent_results_kept_intact() -> None:
    big = "x" * 1000
    history = [_tr_turn(str(i), big) for i in range(3)]
    out = _elide_stale_tool_results(history, keep_recent=6)
    # Fewer turns than keep_recent — nothing elided.
    assert all(m["content"][0]["content"] == big for m in out)


def test_old_large_results_elided_recent_preserved() -> None:
    big = "x" * 1000
    history = [_tr_turn(str(i), big) for i in range(10)]
    out = _elide_stale_tool_results(history, keep_recent=3)
    elided = [m["content"][0]["content"] for m in out]
    # First 7 elided, last 3 preserved.
    assert elided[:7] == [_ELIDE_STUB] * 7
    assert elided[7:] == [big] * 3


def test_small_results_not_elided() -> None:
    small = "ok"
    history = [_tr_turn(str(i), small) for i in range(10)]
    out = _elide_stale_tool_results(history, keep_recent=3)
    # Short results are left alone even when old (cheap to keep).
    assert all(m["content"][0]["content"] == small for m in out)


def test_non_tool_result_turns_ignored() -> None:
    history = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": [{"type": "text", "text": "hi"}]},
        _tr_turn("a", "y" * 1000),
    ]
    out = _elide_stale_tool_results(history, keep_recent=6)
    assert out[0]["content"] == "hello"
    assert out[2]["content"][0]["content"] == "y" * 1000


# ----------------------------------------------------------------- persistence


def test_history_round_trips_through_session_dir(tmp_path: Path) -> None:
    history = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]
    save_history(tmp_path, history)
    assert (tmp_path / "session" / "history.json").exists()
    assert load_history(tmp_path) == history


def test_load_history_missing_or_corrupt_returns_empty(tmp_path: Path) -> None:
    assert load_history(tmp_path) == []
    (tmp_path / "session").mkdir()
    (tmp_path / "session" / "history.json").write_text("{ not json")
    assert load_history(tmp_path) == []


def test_feed_round_trips_and_caps_length(tmp_path: Path) -> None:
    lines = [f"line {i}" for i in range(600)]
    save_feed(tmp_path, lines)
    loaded = load_feed(tmp_path)
    # Capped to the last 500 lines so the on-disk transcript can't grow unbounded.
    assert len(loaded) == 500
    assert loaded[-1] == "line 599"


def test_session_resumes_persisted_history(tmp_path: Path) -> None:
    history = [{"role": "user", "content": "earlier"}]
    save_history(tmp_path, history)
    # A fresh session with no seeded history should pick up the persisted transcript,
    # so quitting and re-entering doesn't make the agent start from scratch.
    session = AgentSession(project_root=tmp_path, client=object())  # type: ignore[arg-type]
    assert session.history == history
