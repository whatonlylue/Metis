"""Minimal M1 TUI: project picker on the left, live agent feed on the right.

"Live" in M1 means the most recent ``runs/actions.jsonl`` entries for whichever
project is selected — wiring an in-process ``AgentLoop`` run to stream
``on_event`` callbacks straight into this feed is the natural next step once
later milestones give the agent something to actually do here.
"""

from __future__ import annotations

from pathlib import Path

from textual.app import App, ComposeResult
from textual.containers import Horizontal
from textual.widgets import DirectoryTree, Footer, Header, Log

from metis.sandbox import read_actions

DEFAULT_PROJECTS_DIR = Path("projects")


class MetisApp(App[None]):
    """Pick a project, watch its agent action feed."""

    CSS = """
    Horizontal { height: 1fr; }
    #picker { width: 30%; border: solid green; }
    #feed { width: 1fr; border: solid blue; }
    """
    BINDINGS = [("q", "quit", "Quit")]

    def __init__(self, projects_dir: Path = DEFAULT_PROJECTS_DIR) -> None:
        super().__init__()
        self._projects_dir = projects_dir

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal():
            yield DirectoryTree(str(self._projects_dir), id="picker")
            yield Log(id="feed", auto_scroll=True)
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#feed", Log).write_line(
            "Select a project directory to see its agent action feed."
        )

    def on_directory_tree_directory_selected(
        self, event: DirectoryTree.DirectorySelected
    ) -> None:
        self._load_feed(Path(event.path))

    def _load_feed(self, project_root: Path) -> None:
        feed = self.query_one("#feed", Log)
        feed.clear()
        feed.write_line(f"Project: {project_root}")
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


def run_tui(projects_dir: Path = DEFAULT_PROJECTS_DIR) -> None:
    MetisApp(projects_dir).run()
