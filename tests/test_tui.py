"""Smoke + behaviour tests for the redesigned TUI."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from textual.widgets import DataTable, ListView, RichLog

from metis.benchmark.store import BenchmarkRecord, append_result
from metis.projects import create_project
from metis.projects.schema import ProjectSpec, TaskType
from metis.tui import MetisApp
from metis.tui.app import (
    ChatInput,
    CredentialsScreen,
    ModelScreen,
    _format_tool_result,
    _tail,
)


def test_app_mounts_rail_leaderboard_feed_and_chat(tmp_path: Path) -> None:
    (tmp_path / "demo-project").mkdir()

    async def _check() -> None:
        app = MetisApp(tmp_path)
        async with app.run_test():
            assert app.query_one("#project-list", ListView) is not None
            assert app.query_one("#thinking", RichLog) is not None
            assert app.query_one("#chat", ChatInput) is not None
            table = app.query_one("#leaderboard", DataTable)
            assert len(table.columns) == 12

    asyncio.run(_check())


def test_projects_listed_as_chat_boxes(tmp_path: Path) -> None:
    (tmp_path / "alpha").mkdir()
    (tmp_path / "beta").mkdir()
    (tmp_path / ".hidden").mkdir()  # dotfiles are not projects

    async def _check() -> None:
        app = MetisApp(tmp_path)
        async with app.run_test():
            names = {item.project_name for item in app._project_items()}
            assert names == {"alpha", "beta"}
            # Each card carries a changing status line.
            app._tick_statuses()

    asyncio.run(_check())


def test_leaderboard_populates_from_results_db(tmp_path: Path) -> None:
    spec = ProjectSpec(
        name="lb",
        description="leaderboard test",
        task_type=TaskType.image_classification,
        target_metric="accuracy",
    )
    root = create_project(tmp_path / "lb", spec)
    append_result(
        root / "benchmark",
        BenchmarkRecord(
            variant_id="logreg",
            task_metric_name="accuracy",
            task_metric_value=0.95,
            param_count=650,
            model_size_mb=0.01,
            latency_ms_p50=0.12,
            latency_ms_p95=0.20,
            throughput_sps=50000.0,
        ),
    )

    async def _check() -> None:
        app = MetisApp(tmp_path)
        async with app.run_test():
            app._load_leaderboard(root)
            table = app.query_one("#leaderboard", DataTable)
            assert table.row_count == 1
            cells = table.get_row_at(0)
            assert "logreg" in cells
            assert "0.9500" in cells

    asyncio.run(_check())


def test_chat_without_api_key_opens_token_manager(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pasting a token / chatting with no key used to do nothing; now it surfaces
    the token manager and remembers the pending message to resume."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("METIS_CREDENTIALS_FILE", str(tmp_path / "creds.json"))
    # A model must be chosen for the credentials gate to fire (no default provider).
    monkeypatch.setenv("METIS_PROVIDER", "anthropic")
    monkeypatch.setenv("METIS_MODEL", "claude-haiku-4-5")
    (tmp_path / "proj").mkdir()

    async def _check() -> None:
        app = MetisApp(tmp_path)
        async with app.run_test():
            app._selected = "proj"
            app._submit_to_agent("proj", "start working")
            await app.workers.wait_for_complete()
            assert app._pending_send == ("proj", "start working")
            assert isinstance(app.screen, CredentialsScreen)

    asyncio.run(_check())


def test_chat_without_model_opens_model_picker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no provider preferred, the first message prompts an explicit model
    pick (not the token manager) and queues the message to resume after."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("METIS_PROVIDER", raising=False)
    monkeypatch.delenv("METIS_MODEL", raising=False)
    monkeypatch.setenv("METIS_CREDENTIALS_FILE", str(tmp_path / "creds.json"))
    monkeypatch.setenv("METIS_MODEL_CONFIG", str(tmp_path / "model.json"))
    (tmp_path / "proj").mkdir()

    async def _check() -> None:
        app = MetisApp(tmp_path)
        async with app.run_test():
            assert app._selection is None
            app._selected = "proj"
            app._submit_to_agent("proj", "start working")
            await app.workers.wait_for_complete()
            assert app._pending_send == ("proj", "start working")
            assert isinstance(app.screen, ModelScreen)

    asyncio.run(_check())


def test_new_project_creates_card(tmp_path: Path) -> None:
    async def _check() -> None:
        app = MetisApp(tmp_path)
        async with app.run_test():
            app._on_new_project(("widgets", ""))
            assert (tmp_path / "widgets" / "project.yaml").exists()
            names = {item.project_name for item in app._project_items()}
            assert "widgets" in names

    asyncio.run(_check())


def test_reselecting_project_restores_its_feed(tmp_path: Path) -> None:
    """Clicking back onto a project must re-render its conversation, not wipe it —
    even while its agent is still working in a background worker."""
    (tmp_path / "alpha").mkdir()
    (tmp_path / "beta").mkdir()

    async def _check() -> None:
        app = MetisApp(tmp_path)
        async with app.run_test():
            items = {i.project_name: i for i in app._project_items()}
            app._select_project(items["alpha"])
            app._feed_write("alpha", "agent says hello")
            # Pretend alpha's agent is mid-run, then switch away and back.
            app._busy.add("alpha")
            app._select_project(items["beta"])
            app._select_project(items["alpha"])
            log = app.query_one("#thinking", RichLog)
            rendered = "\n".join(str(line) for line in log.lines)
            assert "agent says hello" in rendered
            assert "still working in the background" in rendered
            # And it persisted, so a fresh app reloads it.
            assert "agent says hello" in app._get_feed("alpha")

    asyncio.run(_check())


def test_format_tool_result_parses_json_message() -> None:
    assert _format_tool_result('{"message": "trained logreg"}') == "trained logreg"
    # Plain (non-JSON) strings pass through untouched.
    assert _format_tool_result("benchmarked logreg, accuracy=0.95") == (
        "benchmarked logreg, accuracy=0.95"
    )
    # No natural message field → compact key=value rendering rather than raw JSON.
    assert _format_tool_result('{"a": 1, "b": 2}') == "a=1, b=2"


def test_tail_keeps_end_and_collapses_whitespace() -> None:
    out = _tail("x" * 50 + "\n\n   END", limit=10)
    assert out.startswith("…")
    assert out.endswith("END")
    assert len(out) == 10
