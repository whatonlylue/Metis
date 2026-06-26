"""Smoke test for the TUI: it mounts and shows the picker + feed widgets."""

from __future__ import annotations

import asyncio
from pathlib import Path

from textual.widgets import DirectoryTree, Log

from metis.tui import MetisApp


def test_app_mounts_picker_and_feed(tmp_path: Path) -> None:
    (tmp_path / "demo-project").mkdir()

    async def _check() -> None:
        app = MetisApp(tmp_path)
        async with app.run_test():
            assert app.query_one("#picker", DirectoryTree) is not None
            assert app.query_one("#feed", Log) is not None

    asyncio.run(_check())
