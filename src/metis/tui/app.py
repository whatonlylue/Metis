"""M3 TUI: project picker, a leaderboard table, and the live agent feed.

The leaderboard reads ``benchmark/results.db`` via the harness-side store (the
TUI is part of the harness, not the agent, so reading sealed results is allowed)
and shows accuracy/task-metric alongside the efficiency columns: parameter
count, on-disk size, single-sample latency (p50/p95), and throughput.

The feed shows the most recent ``runs/actions.jsonl`` entries for the selected
project.
"""

from __future__ import annotations

from pathlib import Path

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import DataTable, DirectoryTree, Footer, Header, Log

from metis.benchmark import compute_budget_status, ranked_leaderboard
from metis.sandbox import read_actions

DEFAULT_PROJECTS_DIR = Path("projects")

_LEADERBOARD_COLUMNS = (
    "Rank",
    "Variant",
    "Metric",
    "Score",
    "Params",
    "Size MB",
    "p50 ms",
    "p95 ms",
    "samp/s",
    "Pareto",
    "Status",
)


class MetisApp(App[None]):
    """Pick a project, watch its leaderboard and agent action feed."""

    CSS = """
    Horizontal { height: 1fr; }
    #picker { width: 30%; border: solid green; }
    #right { width: 1fr; }
    #leaderboard { height: 60%; border: solid magenta; }
    #feed { height: 1fr; border: solid blue; }
    """
    BINDINGS = [("q", "quit", "Quit")]

    def __init__(self, projects_dir: Path = DEFAULT_PROJECTS_DIR) -> None:
        super().__init__()
        self._projects_dir = projects_dir

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal():
            yield DirectoryTree(str(self._projects_dir), id="picker")
            with Vertical(id="right"):
                yield DataTable(id="leaderboard")
                yield Log(id="feed", auto_scroll=True)
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#leaderboard", DataTable)
        table.add_columns(*_LEADERBOARD_COLUMNS)
        self.query_one("#feed", Log).write_line(
            "Select a project directory to see its leaderboard and agent action feed."
        )

    def on_directory_tree_directory_selected(self, event: DirectoryTree.DirectorySelected) -> None:
        project_root = Path(event.path)
        self._load_leaderboard(project_root)
        self._load_feed(project_root)

    def _load_leaderboard(self, project_root: Path) -> None:
        table = self.query_one("#leaderboard", DataTable)
        table.clear()
        benchmark_dir = project_root / "benchmark"
        if not (benchmark_dir / "results.db").exists():
            return
        try:
            # Project-aware ranking (Pareto / weighted / single), including pruned
            # variants so the human can see what dropped out of the active search.
            rows = ranked_leaderboard(project_root, n=25, include_pruned=True)
        except Exception:
            return
        for i, r in enumerate(rows, 1):
            table.add_row(
                str(i),
                str(r["variant_id"]),
                str(r["task_metric_name"]),
                _fmt(r["task_metric_value"], ".4f"),
                _fmt(r["param_count"], ",d"),
                _fmt(r["model_size_mb"], ".3f"),
                _fmt(r["latency_ms_p50"], ".3f"),
                _fmt(r["latency_ms_p95"], ".3f"),
                _fmt(r["throughput_sps"], ",.0f"),
                str(r.get("pareto_rank", "N/A")),
                "pruned" if r.get("pruned") else "active",
            )

    def _load_feed(self, project_root: Path) -> None:
        feed = self.query_one("#feed", Log)
        feed.clear()
        feed.write_line(f"Project: {project_root}")
        try:
            b = compute_budget_status(project_root)
            stop = " [STOP]" if b.should_stop else ""
            feed.write_line(
                f"Budget: {b.wall_clock_minutes_used:.1f} min, {b.variants_trained} variants, "
                f"${b.dollars_used:.2f}{stop}"
            )
        except Exception:
            pass
        try:
            actions = read_actions(project_root)
        except Exception as exc:
            feed.write_line(f"(could not read action log: {exc})")
            return
        if not actions:
            feed.write_line("(no actions logged yet)")
            return
        for action in actions[-100:]:
            status = "ok" if action["ok"] else f"error: {action['error']}"
            feed.write_line(f"[{action['timestamp']}] {action['tool']} {action['args']} {status}")


def _fmt(value: object, spec: str) -> str:
    """Format a possibly-None numeric cell, falling back to ``N/A``."""
    if value is None:
        return "N/A"
    try:
        return format(value, spec)
    except (ValueError, TypeError):
        return str(value)


def run_tui(projects_dir: Path = DEFAULT_PROJECTS_DIR) -> None:
    MetisApp(projects_dir).run()
