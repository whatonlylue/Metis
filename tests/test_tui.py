"""Smoke + leaderboard tests for the TUI."""

from __future__ import annotations

import asyncio
from pathlib import Path

from textual.widgets import DataTable, DirectoryTree, Log

from metis.benchmark.store import BenchmarkRecord, append_result
from metis.projects import create_project
from metis.projects.schema import ProjectSpec, TaskType
from metis.tui import MetisApp


def test_app_mounts_picker_feed_and_leaderboard(tmp_path: Path) -> None:
    (tmp_path / "demo-project").mkdir()

    async def _check() -> None:
        app = MetisApp(tmp_path)
        async with app.run_test():
            assert app.query_one("#picker", DirectoryTree) is not None
            assert app.query_one("#feed", Log) is not None
            table = app.query_one("#leaderboard", DataTable)
            assert table is not None
            assert len(table.columns) == 12

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
