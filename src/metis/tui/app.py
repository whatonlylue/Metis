"""Metis TUI.

Layout (the M-series redesign):

  ┌ projects ─┬─ leaderboard (the models the agent has tried + scores) ─┐
  │ + New     │                                                          │
  │ ▸ projA   ├─ agent thinking / planning (claude-code-style feed) ─────┤
  │   status… │                                                          │
  │ ▸ projB   │  > chat box — talk to the agent, steer it                │
  └───────────┴──────────────────────────────────────────────────────────┘

The left rail replaces the old folder tree with one "chat box" per project: a
name plus a short, changing status line. Pick a project to load its leaderboard
(top, harness-read from the sealed ``results.db``) and its agent conversation
(bottom). The chat box at the very bottom sends guidance to the agent, whose
thinking/tool-use streams into the feed above it.

The TUI is part of the harness, not the agent, so it may read the sealed
benchmark results directly. The agent itself only ever talks through its tools.
"""

from __future__ import annotations

import itertools
import json
import os
import textwrap
from pathlib import Path
from typing import Any

from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.events import Key as _Key
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    ListItem,
    ListView,
    RichLog,
    TextArea,
)

from metis.agent.credentials import (
    FileCredentialStore,
    looks_like_api_key,
    mask_key,
    resolve_api_key,
)
from metis.agent.model_config import ModelSelection, load_selection, save_selection
from metis.benchmark import (
    compute_budget_status,
    get_failed_variants,
    get_latest_robustness,
    ranked_leaderboard,
)
from metis.agent.client import Usage
from metis.agent.pricing import cost_usd
from metis.agent.session import (
    _FEED_MAX_LINES,
    _TRAIN_MAX_LINES,
    load_feed,
    load_train,
    save_feed,
    save_train,
)
from metis.paths import projects_dir as _default_projects_dir
from metis.paths import ui_config_path as _default_ui_config_path
from metis.projects import create_project
from metis.projects.schema import ProjectSpec, TaskType

DEFAULT_PROJECTS_DIR = _default_projects_dir()

#: Default theme — the ANSI themes render with the user's own terminal palette, so
#: out of the box Metis "matches the terminal" until they pick something explicit.
DEFAULT_THEME = "ansi-dark"

#: Braille spinner frames cycled next to the active task so the user can see the
#: agent is working (and on what), rather than a frozen static line.
_SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

#: Wrap width for a failed variant's error text inside the leaderboard Status cell,
#: so long tracebacks wrap within the row instead of forcing horizontal scroll.
_ERROR_WRAP_WIDTH = 40


def _ui_config_path() -> Path:
    """Where the persisted UI prefs (theme) live; override with ``METIS_UI_CONFIG``."""
    override = os.environ.get("METIS_UI_CONFIG")
    if override:
        return Path(override)
    return _default_ui_config_path()


def load_ui_theme() -> str | None:
    """Return the saved theme name, or ``None`` if unset/unreadable."""
    try:
        data = json.loads(_ui_config_path().read_text())
    except (OSError, ValueError):
        return None
    theme = data.get("theme") if isinstance(data, dict) else None
    return theme if isinstance(theme, str) else None


def save_ui_theme(theme: str) -> None:
    """Persist the chosen theme so it survives a restart (best-effort)."""
    try:
        path = _ui_config_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"theme": theme}))
    except (OSError, TypeError):
        pass

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
    "Robust",
    "Status",
)

#: Hints cycled in a project's status line while the agent is idle.
_IDLE_HINTS = (
    "idle — type below to brief the agent",
    "waiting for your guidance",
    "ready when you are",
)


class ChatInput(TextArea):
    """TextArea chat input: Enter submits, Shift+Enter inserts a newline."""

    class Submitted(Message):
        def __init__(self, textarea: "ChatInput", value: str) -> None:
            super().__init__()
            self.value = value
            self.textarea = textarea

    async def _on_key(self, event: _Key) -> None:
        if event.key == "enter":
            event.prevent_default()
            self.post_message(ChatInput.Submitted(self, self.text))
        elif event.key == "shift+enter":
            self.insert("\n")
            event.prevent_default()
        else:
            await super()._on_key(event)


def _script_to_label(script: str) -> str:
    """Convert a training/data script path to a friendly 2-4 word description."""
    path = Path(script)
    stem = path.stem.lower()
    parts = path.parts

    # Model training scripts live at models/<variant>/train.py
    if stem == "train" and len(parts) >= 2:
        variant = parts[-2]
        words = variant.replace("-", "_").split("_")
        clean = " ".join(w.capitalize() for w in words if w)
        return f"Training {clean}"

    _KNOWN: dict[str, str] = {
        "setup_processed": "Preparing training data",
        "combine_data": "Combining data splits",
        "preprocess_retinal": "Preprocessing retinal images",
        "preprocess": "Preprocessing dataset",
        "inspect_data": "Inspecting dataset",
        "setup_check": "Checking environment",
    }
    if stem in _KNOWN:
        return _KNOWN[stem]

    words = stem.replace("-", "_").split("_")
    return " ".join(w.capitalize() for w in words if w)


def _tool_feed_label(name: str, args: dict[str, Any] | None) -> str:
    """Friendly 2-4 word description of a tool call for the chat feed."""
    args = args or {}
    match name:
        case "read_file":
            return f"Reading {Path(args.get('path', 'file')).name}"
        case "write_file":
            return f"Writing {Path(args.get('path', 'file')).name}"
        case "list_dir":
            return f"Exploring {args.get('path', '.')}"
        case "run_python":
            return _script_to_label(str(args.get("script", "script")))
        case "ingest_dataset":
            return f"Ingesting {args.get('dataset', '')} dataset"
        case "instantiate_template":
            return f"Scaffolding {args.get('template', '')} template"
        case "submit_for_benchmark":
            return f"Benchmarking {args.get('variant_id', 'variant')}"
        case "get_leaderboard":
            return "Checking leaderboard"
        case "request_prune":
            return "Pruning weak models"
        case "get_budget_status":
            return "Checking budget status"
        case "get_failed_variants":
            return "Reviewing failed runs"
        case "check_plateau":
            return "Checking training plateau"
        case "save_project_spec":
            return "Saving project spec"
        case "list_model_templates":
            return "Listing model templates"
        case _:
            return name.replace("_", " ").title()


def _is_project_dir(path: Path) -> bool:
    """A project is any direct subdirectory (``project.yaml`` may not exist yet)."""
    return path.is_dir() and not path.name.startswith(".")


class ProjectItem(ListItem):
    """One project "chat box" in the left rail: a name and a changing status."""

    def __init__(self, project_name: str, project_root: Path) -> None:
        super().__init__()
        self.project_name = project_name
        self.project_root = project_root

    def compose(self) -> ComposeResult:
        with Vertical(classes="proj-box"):
            yield Label(self.project_name, classes="proj-name")
            yield Label("idle", classes="proj-status")

    def set_status(self, text: str) -> None:
        try:
            self.query_one(".proj-status", Label).update(text)
        except Exception:
            pass


class CredentialsScreen(ModalScreen[bool]):
    """Modal to set / validate / clear the agent's API key.

    Metis is litellm-native: one key drives whichever provider the chosen model
    belongs to. Dismisses ``True`` when a usable key is present on close (so the
    caller can resume whatever it was blocked on), ``False`` otherwise. The raw
    secret is never rendered, logged, or echoed — only a presence/length indicator.
    """

    CSS = """
    CredentialsScreen { align: center middle; }
    #dialog {
        width: 64; height: auto; padding: 1 2;
        border: thick $accent; background: $surface;
    }
    #cred-status { margin-bottom: 1; }
    #cred-msg { margin-top: 1; color: $text-muted; }
    """
    BINDINGS = [("escape", "close", "Close")]

    def __init__(self, store: FileCredentialStore) -> None:
        super().__init__()
        self._store = store

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Label("Agent API token", id="cred-title")
            yield Label(self._status_text(), id="cred-status")
            yield Input(password=True, placeholder="Paste API key (hidden)", id="cred-input")
            with Horizontal():
                yield Button("Save", id="cred-save", variant="primary")
                yield Button("Validate", id="cred-validate")
                yield Button("Clear", id="cred-clear", variant="error")
                yield Button("Close", id="cred-close")
            yield Label(
                "Or export METIS_API_KEY, or your provider's key "
                "(e.g. ANTHROPIC_API_KEY / OPENAI_API_KEY).",
                id="cred-msg",
            )

    def _status_text(self) -> str:
        return f"Stored key: {mask_key(self._store.get())}"

    def _refresh_status(self) -> None:
        self.query_one("#cred-status", Label).update(self._status_text())

    def on_input_submitted(self, event: Input.Submitted) -> None:
        # Enter in the field saves, so a paste-and-Enter actually persists.
        self._save()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cred-save":
            self._save()
        elif event.button.id == "cred-validate":
            field = self.query_one("#cred-input", Input)
            candidate = field.value or self._store.get()
            ok = looks_like_api_key(candidate)
            self.query_one("#cred-msg", Label).update(
                "Key looks valid (format check)." if ok else "No valid key present."
            )
        elif event.button.id == "cred-clear":
            removed = self._store.clear()
            self.query_one("#cred-input", Input).value = ""
            self._refresh_status()
            self.query_one("#cred-msg", Label).update("Cleared." if removed else "No stored key.")
        elif event.button.id == "cred-close":
            self.action_close()

    def _save(self) -> None:
        msg = self.query_one("#cred-msg", Label)
        field = self.query_one("#cred-input", Input)
        value = field.value
        if not value.strip():
            msg.update("Nothing to save: enter a key first.")
            return
        try:
            self._store.set(value)
        except ValueError as exc:
            msg.update(str(exc))
            return
        field.value = ""  # never keep the secret on screen
        self._refresh_status()
        msg.update("Saved (key hidden). Close to continue.")

    def action_close(self) -> None:
        self.dismiss(self._store.has())


class NewProjectScreen(ModalScreen[tuple[str, str] | None]):
    """Modal collecting a new project's name + task description.

    Returns ``(name, description)`` on create, or ``None`` if cancelled.
    """

    CSS = """
    NewProjectScreen { align: center middle; }
    #np-dialog {
        width: 72; height: auto; padding: 1 2;
        border: thick $accent; background: $surface;
    }
    #np-msg { margin-top: 1; color: $text-muted; }
    """
    BINDINGS = [("escape", "cancel", "Cancel")]

    def __init__(self, existing: set[str]) -> None:
        super().__init__()
        self._existing = existing

    def compose(self) -> ComposeResult:
        with Vertical(id="np-dialog"):
            yield Label("New project", id="np-title")
            yield Label("Name (folder-safe, e.g. fundus-grading)")
            yield Input(placeholder="project name", id="np-name")
            yield Label("What should the model classify or predict?")
            yield Input(
                placeholder="e.g. grade diabetic retinopathy in fundus images", id="np-desc"
            )
            with Horizontal():
                yield Button("Create", id="np-create", variant="primary")
                yield Button("Cancel", id="np-cancel")
            yield Label("", id="np-msg")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "np-create":
            self._create()
        elif event.button.id == "np-cancel":
            self.dismiss(None)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self._create()

    def _create(self) -> None:
        msg = self.query_one("#np-msg", Label)
        name = self.query_one("#np-name", Input).value.strip()
        desc = self.query_one("#np-desc", Input).value.strip()
        if not name:
            msg.update("Enter a project name.")
            return
        safe = name.replace(" ", "-")
        if any(c in safe for c in "/\\") or safe.startswith("."):
            msg.update("Name can't contain slashes or start with a dot.")
            return
        if safe in self._existing:
            msg.update(f"Project {safe!r} already exists.")
            return
        self.dismiss((safe, desc))

    def action_cancel(self) -> None:
        self.dismiss(None)


class _ThemeItem(ListItem):
    """A theme row that remembers which theme name it stands for."""

    def __init__(self, theme_name: str) -> None:
        super().__init__(Label(theme_name))
        self.theme_name = theme_name


class ThemeScreen(ModalScreen[str | None]):
    """Pick a color theme. Dismisses the chosen theme name, or ``None`` if cancelled.

    The list previews live — focusing a row applies that theme immediately so the
    user can see it before committing; cancelling restores the theme they started on.
    """

    CSS = """
    ThemeScreen { align: center middle; }
    #theme-dialog {
        width: 52; height: auto; max-height: 90%; padding: 1 2;
        border: round $accent; background: $surface;
    }
    #theme-list { height: auto; max-height: 24; scrollbar-size: 0 0; }
    """
    BINDINGS = [("escape", "cancel", "Close")]

    def __init__(self, themes: list[str], current: str) -> None:
        super().__init__()
        self._themes = themes
        self._current = current

    def compose(self) -> ComposeResult:
        with Vertical(id="theme-dialog"):
            yield Label("Color theme — ↑/↓ to preview, Enter to keep", id="theme-title")
            yield ListView(id="theme-list")

    def on_mount(self) -> None:
        listview = self.query_one("#theme-list", ListView)
        for name in self._themes:
            listview.append(_ThemeItem(name))
        if self._current in self._themes:
            listview.index = self._themes.index(self._current)

    def on_list_view_highlighted(self, event: ListView.Highlighted) -> None:
        # Live preview: apply the focused theme to the running app behind the modal.
        item = event.item
        if isinstance(item, _ThemeItem):
            self.app.theme = item.theme_name

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        item = event.item
        self.dismiss(item.theme_name if isinstance(item, _ThemeItem) else None)

    def action_cancel(self) -> None:
        # Restore whatever was active when we opened, since previews mutated it.
        self.app.theme = self._current
        self.dismiss(None)


class _ModelItem(ListItem):
    """A suggested-model row that remembers its litellm model string."""

    def __init__(self, model: str, label: str) -> None:
        super().__init__(Label(label))
        self.model = model


#: Fallback click-to-fill list when litellm can't infer the user's providers (no
#: key detectable). Not an exhaustive registry — any litellm model string works.
_STATIC_SUGGESTED_MODELS = [
    "anthropic/claude-haiku-4-5",
    "anthropic/claude-sonnet-4-6",
    "anthropic/claude-opus-4-8",
    "openai/gpt-4o-mini",
    "openai/gpt-4o",
    "gemini/gemini-1.5-flash",
]


def _suggested_models() -> list[str]:
    """Model strings to offer as click-to-fill suggestions.

    Prefer litellm's live view of the chat models the user's configured keys can
    actually drive — ``get_valid_models`` inspects the provider env vars — so the
    list reflects whichever provider key is present. Falls back to a small static
    list when no provider key is detectable (or litellm errors).
    """
    try:
        import litellm

        valid = litellm.get_valid_models(check_provider_endpoint=False)
        chat: list[str] = []
        for model in valid:
            try:
                if litellm.get_model_info(model).get("mode") == "chat":
                    chat.append(model)
            except Exception:
                continue
        if chat:
            return sorted(chat)
    except Exception:
        pass
    return list(_STATIC_SUGGESTED_MODELS)


class ModelScreen(ModalScreen["str | None"]):
    """Choose which model drives the agent — any litellm model string.

    Metis is litellm-native, so this is a free-form field: type a litellm model
    id (``anthropic/claude-opus-4-8``, ``gpt-4o``, ``gemini/gemini-1.5-pro``, …)
    and press Enter, or click a suggestion to fill the field. There is no default
    and no preferred provider. Dismisses the chosen model string, or ``None`` if
    cancelled.
    """

    CSS = """
    ModelScreen { align: center middle; }
    #model-dialog {
        width: 60; height: auto; max-height: 90%; padding: 1 2;
        border: round $accent; background: $surface;
    }
    #model-list { height: auto; max-height: 24; scrollbar-size: 0 0; }
    #model-hint { color: $text-muted; margin-top: 1; }
    """
    BINDINGS = [("escape", "cancel", "Close")]

    def __init__(self, selection: ModelSelection | None) -> None:
        super().__init__()
        self._selection = selection

    def compose(self) -> ComposeResult:
        with Vertical(id="model-dialog"):
            yield Label("Agent model — type a litellm id, Enter to select", id="model-title")
            yield Input(
                value=self._selection.model if self._selection is not None else "",
                placeholder="e.g. anthropic/claude-opus-4-8 or gpt-4o",
                id="model-input",
            )
            yield Label("Suggestions (click to fill):", id="model-suggest-label")
            yield ListView(id="model-list")
            yield Label(
                "The model picked here drives the Metis agent (not the models it trains).",
                id="model-hint",
            )

    def on_mount(self) -> None:
        # Populate after the first refresh so the ListView is guaranteed mounted.
        self.call_after_refresh(self._populate)

    def _populate(self) -> None:
        listview = self.query_one("#model-list", ListView)
        for model in _suggested_models():
            marker = "●" if self._selection is not None and self._selection.model == model else " "
            listview.append(_ModelItem(model, f"{marker} {model}"))

    def on_input_submitted(self, event: Input.Submitted) -> None:
        model = event.value.strip()
        self.dismiss(model or None)

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        # Clicking a suggestion fills the field so it can still be edited before Enter.
        item = event.item
        if isinstance(item, _ModelItem):
            self.query_one("#model-input", Input).value = item.model

    def action_cancel(self) -> None:
        self.dismiss(None)


class MetisApp(App[None]):
    """Pick a project, watch its leaderboard, and chat with the agent driving it."""

    #: One consistent panel look everywhere — a rounded accent border carrying the
    #: panel's title (the "Talk to Metis" style applied uniformly), and scrollbars
    #: hidden (``scrollbar-size: 0 0``) while wheel/keyboard scrolling still works.
    CSS = """
    #body { height: 1fr; padding: 0 1 0 0; }
    #sidebar {
        width: 34; border: round $accent; padding: 0 1;
        border-title-align: left; margin: 0 1;
    }
    #new-project { width: 1fr; margin: 0 0 1 0; }
    #project-list { height: 1fr; scrollbar-size: 0 0; background: transparent; }
    .proj-box { height: auto; padding: 0 1; }
    .proj-name { text-style: bold; }
    .proj-status { color: $text-muted; }
    #main { width: 1fr; }
    #leaderboard {
        height: 45%; border: round $accent; border-title-align: left;
        scrollbar-size: 0 0; padding: 0 1;
    }
    #agent-cols { height: 1fr; }
    #thinking {
        width: 2fr; border: round $accent; border-title-align: left;
        scrollbar-size: 0 0; padding: 0 1; margin: 0 1 0 0;
    }
    #training {
        width: 1fr; border: round $accent; border-title-align: left;
        scrollbar-size: 0 0; padding: 0 1;
    }
    #chat {
        dock: bottom; border: round $accent; border-title-align: left;
        height: 5; padding: 0 1;
    }
    """
    BINDINGS = [
        ("q", "quit", "Quit"),
        ("k", "manage_credentials", "Token"),
        ("m", "choose_model", "Model"),
        ("n", "new_project", "New project"),
        ("t", "choose_theme", "Theme"),
    ]

    def __init__(self, projects_dir: Path = DEFAULT_PROJECTS_DIR) -> None:
        super().__init__()
        self._projects_dir = projects_dir
        self._credential_store = FileCredentialStore()
        self._selected: str | None = None
        self._sessions: dict[str, Any] = {}  # project_name -> AgentSession
        self._busy: set[str] = set()
        self._idle_cycle = itertools.cycle(_IDLE_HINTS)
        self._spinner = itertools.cycle(_SPINNER_FRAMES)
        # Current short activity label per busy project (what the spinner annotates).
        self._activity: dict[str, str] = {}
        self._pending_send: tuple[str, str] | None = None  # (project_name, text)
        # Per-project display feed (rendered markup lines). Kept so re-selecting a
        # project re-renders its conversation instead of wiping it, and persisted to
        # session/ so it survives a restart. Lazily loaded from disk on first access.
        self._feeds: dict[str, list[str]] = {}
        # Per-project training-output box, persisted like the feed so switching away
        # and back (or restarting) doesn't wipe the live epoch/score history.
        self._train_feeds: dict[str, list[str]] = {}
        # Cumulative token usage per project (this TUI session), for the live
        # token/cost readout in the agent pane's title.
        self._usage: dict[str, Usage] = {}
        # Which provider + model drives the agent (persisted across restarts).
        # ``None`` until the user picks — Metis prefers no provider, so the first
        # chat or token action prompts for an explicit choice.
        self._selection: ModelSelection | None = load_selection()

    # ------------------------------------------------------------------ layout
    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="body"):
            with Vertical(id="sidebar"):
                yield Button("+ New project", id="new-project", variant="primary")
                yield ListView(id="project-list")
            with Vertical(id="main"):
                yield DataTable(id="leaderboard")
                with Horizontal(id="agent-cols"):
                    yield RichLog(id="thinking", wrap=True, markup=True)
                    yield RichLog(id="training", wrap=True, markup=True)
                yield ChatInput(id="chat", show_line_numbers=False)
        yield Footer()

    def on_mount(self) -> None:
        self.title = "Metis"
        self.sub_title = "train efficient task-specific models"
        # Restore the saved theme; default to an ANSI theme so we match the terminal.
        saved = load_ui_theme()
        self.theme = saved if (saved and saved in self.available_themes) else DEFAULT_THEME
        # Persist whenever the theme changes — including via the command palette.
        self.theme_changed_signal.subscribe(self, self._persist_theme)

        self.query_one("#leaderboard", DataTable).add_columns(*_LEADERBOARD_COLUMNS)
        # Uniform bordered, titled panels (the "Talk to Metis" look applied everywhere).
        self.query_one("#sidebar", Vertical).border_title = "Projects"
        self.query_one("#leaderboard", DataTable).border_title = "Leaderboard — models tried"
        self.query_one("#thinking", RichLog).border_title = self._agent_title_text(None)
        self.query_one("#training", RichLog).border_title = "Training output"
        self.query_one("#chat", ChatInput).border_title = "Talk to Metis"

        self._reload_projects()
        self.query_one("#thinking", RichLog).write(
            "[dim]Pick a project on the left, or press [b]n[/b] to start a new one.[/dim]"
        )
        self.set_interval(2.0, self._tick_statuses)
        # Faster cadence drives the spinner on whatever the agent is actively doing.
        self.set_interval(0.12, self._tick_spinner)

    # ----------------------------------------------------------------- theme
    def _persist_theme(self, theme: Any) -> None:
        name = getattr(theme, "name", None)
        if isinstance(name, str):
            save_ui_theme(name)

    def action_choose_theme(self) -> None:
        themes = sorted(self.available_themes.keys())
        self.push_screen(ThemeScreen(themes, self.theme), self._on_theme_chosen)

    def _on_theme_chosen(self, name: str | None) -> None:
        if name:
            self.theme = name  # fires theme_changed_signal → _persist_theme

    # ----------------------------------------------------------------- project rail
    def _reload_projects(self) -> None:
        listview = self.query_one("#project-list", ListView)
        listview.clear()
        self._projects_dir.mkdir(parents=True, exist_ok=True)
        for path in sorted(p for p in self._projects_dir.iterdir() if _is_project_dir(p)):
            listview.append(ProjectItem(path.name, path))

    def _project_items(self) -> list[ProjectItem]:
        return list(self.query("#project-list ProjectItem").results(ProjectItem))

    def _existing_names(self) -> set[str]:
        return {item.project_name for item in self._project_items()}

    def _tick_statuses(self) -> None:
        """Refresh the changing status line on each idle project's chat box."""
        hint = next(self._idle_cycle)
        for item in self._project_items():
            if item.project_name in self._busy:
                continue
            item.set_status(hint)

    def _tick_spinner(self) -> None:
        """Animate a spinner next to whatever each busy project is currently doing."""
        if not self._busy:
            return
        frame = next(self._spinner)
        for item in self._project_items():
            if item.project_name in self._busy:
                activity = self._activity.get(item.project_name, "working")
                item.set_status(f"{frame} {activity}")
        # Mirror the spinner into the selected agent panel's title bar.
        if self._selected in self._busy:
            self._refresh_agent_title(frame)

    def _set_status(self, project_name: str, text: str) -> None:
        for item in self._project_items():
            if item.project_name == project_name:
                item.set_status(text)
                return

    # ------------------------------------------------------------------ feed
    def _get_feed(self, project_name: str) -> list[str]:
        """The project's display feed, loaded from session/ on first access."""
        if project_name not in self._feeds:
            self._feeds[project_name] = load_feed(self._projects_dir / project_name)
        return self._feeds[project_name]

    def _feed_write(self, project_name: str, markup: str) -> None:
        """Append a line to a project's feed: persist it, and render it if selected.

        Routing every project-scoped line through here (instead of writing straight
        to the RichLog) is what lets a different project's conversation survive being
        switched away from and back to — and what makes it reload after a restart.
        """
        feed = self._get_feed(project_name)
        feed.append(markup)
        # Keep the in-memory buffer bounded in lockstep with the on-disk cap, so a
        # long-running project's transcript doesn't grow without limit or flood the
        # pane when re-selected.
        if len(feed) > _FEED_MAX_LINES:
            del feed[:-_FEED_MAX_LINES]
        save_feed(self._projects_dir / project_name, feed)
        if self._selected == project_name:
            self.query_one("#thinking", RichLog).write(markup)

    # ------------------------------------------------------------------ training box
    def _get_train(self, project_name: str) -> list[str]:
        """The project's training-output lines, loaded from session/ on first access."""
        if project_name not in self._train_feeds:
            self._train_feeds[project_name] = load_train(self._projects_dir / project_name)
        return self._train_feeds[project_name]

    def _train_write(self, project_name: str, markup: str) -> None:
        """Append a line to a project's training box: persist it, render if selected.

        Persisting (instead of writing straight to the RichLog) is what lets the
        training output survive switching projects and restarting the TUI.
        """
        feed = self._get_train(project_name)
        feed.append(markup)
        if len(feed) > _TRAIN_MAX_LINES:
            del feed[:-_TRAIN_MAX_LINES]
        save_train(self._projects_dir / project_name, feed)
        if self._selected == project_name:
            self.query_one("#training", RichLog).write(markup)

    # ------------------------------------------------------------------ usage/cost
    def _model_for(self, project_name: str) -> str:
        """The model id driving a project's session (for pricing), or the selected one."""
        session = self._sessions.get(project_name)
        model = getattr(getattr(session, "client", None), "model", None)
        if model:
            return model
        return self._selection.model if self._selection is not None else ""

    def _agent_title_text(self, project_name: str | None, spinner: str = "") -> str:
        """Render the agent pane title with a live token + session-cost readout.

        The readout makes prompt caching visible: ``reused`` is the running sum of
        ``cache_read_input_tokens`` (tokens served from cache at ~0.1x), so the user
        can confirm the cache is working rather than wondering. ``cost`` already
        prices cache reads/writes at their discounted/premium rates.
        """
        prefix = f"{spinner} " if spinner else ""
        base = f"{prefix}Agent — session & plan"
        if project_name is None:
            return base
        usage = self._usage.get(project_name)
        if usage is None:
            return base
        total_tokens = (
            usage.input_tokens
            + usage.output_tokens
            + usage.cache_creation_input_tokens
            + usage.cache_read_input_tokens
        )
        cost = cost_usd(usage, self._model_for(project_name))
        reused = usage.cache_read_input_tokens
        return (
            f"{base}   ·   {total_tokens:,} tok  ·  {reused:,} reused (cached)  ·  ${cost:,.4f}"
        )

    def _refresh_agent_title(self, spinner: str = "") -> None:
        try:
            self.query_one("#thinking", RichLog).border_title = self._agent_title_text(
                self._selected, spinner
            )
        except Exception:
            pass

    def _accumulate_usage(self, project_name: str, usage: Usage) -> None:
        self._usage[project_name] = self._usage.get(project_name, Usage()) + usage
        if project_name == self._selected:
            self._refresh_agent_title()

    # ---------------------------------------------------------------- selection
    def on_list_view_selected(self, event: ListView.Selected) -> None:
        item = event.item
        if isinstance(item, ProjectItem):
            self._select_project(item)

    def _select_project(self, item: ProjectItem) -> None:
        self._selected = item.project_name
        self._load_leaderboard(item.project_root)
        self._refresh_agent_title()
        # Re-render the project's persisted training output (epochs/scores) rather than
        # wiping it — it now survives switching projects and restarts.
        training = self.query_one("#training", RichLog)
        training.clear()
        for line in self._get_train(item.project_name):
            training.write(line)
        chat = self.query_one("#chat", ChatInput)
        chat.border_title = f"Talk to Metis — {item.project_name}"
        log = self.query_one("#thinking", RichLog)
        log.clear()
        feed = self._get_feed(item.project_name)
        if feed:
            # Re-render the prior conversation rather than wiping it — including for a
            # project whose agent is still working in a background worker.
            for line in feed:
                log.write(line)
            if item.project_name in self._busy:
                log.write("[dim]— agent is still working in the background; output continues below —[/dim]")
        else:
            log.write(f"[b]{item.project_name}[/b] selected.")
            self._write_budget(item.project_root, log)
        chat.focus()

    # --------------------------------------------------------------- leaderboard
    def _load_leaderboard(self, project_root: Path) -> None:
        table = self.query_one("#leaderboard", DataTable)
        table.clear()
        benchmark_dir = project_root / "benchmark"
        if not (benchmark_dir / "results.db").exists():
            return
        try:
            rows = ranked_leaderboard(project_root, n=25, include_pruned=True)
        except Exception:
            return
        try:
            robustness = get_latest_robustness(benchmark_dir)
        except Exception:
            robustness = {}
        for i, r in enumerate(rows, 1):
            rob = robustness.get(str(r["variant_id"]), {})
            pruned = bool(r.get("pruned"))
            status = (
                Text("pruned", style="yellow") if pruned else Text("active", style="green")
            )
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
                _fmt(rob.get("aggregate_robustness"), ".3f"),
                status,
            )

        # Errored variants have no score, so they're excluded from the ranked query
        # above; append them at the bottom so a crashed model is still visible. The
        # full error wraps inside the Status cell (no horizontal scroll) and is tagged
        # red so a failed run reads as failed at a glance.
        try:
            failed = get_failed_variants(benchmark_dir, n=25)
        except Exception:
            failed = []
        for r in failed:
            err = " ".join(str(r.get("error") or "unknown error").split())
            wrapped = textwrap.wrap(err, width=_ERROR_WRAP_WIDTH) or ["unknown error"]
            status_cell = Text("\n".join(["✗ FAILED"] + wrapped), style="bold red")
            table.add_row(
                Text("—"),
                Text(str(r["variant_id"]), style="red"),
                Text("—"),
                "N/A",
                "N/A",
                "N/A",
                "N/A",
                "N/A",
                "N/A",
                "N/A",
                "N/A",
                status_cell,
                height=len(wrapped) + 1,
            )

    def _write_budget(self, project_root: Path, log: RichLog) -> None:
        try:
            b = compute_budget_status(project_root)
        except Exception:
            return
        stop = " [b red][STOP][/b red]" if b.should_stop else ""
        log.write(
            f"[dim]budget: {b.wall_clock_minutes_used:.1f} min, {b.variants_trained} variants, "
            f"${b.dollars_used:.2f}{stop}[/dim]"
        )

    # --------------------------------------------------------------- quit
    def action_quit(self) -> None:  # type: ignore[override]
        """Quit cleanly: stop any in-flight training first.

        Training runs in sandboxed subprocesses on worker threads; if we just exit,
        they keep churning (or get orphaned). The agent transcript is already
        persisted to session/history.json after every step (see AgentSession), so
        once training is stopped there's nothing more to flush here.
        """
        from metis.sandbox import terminate_all

        try:
            killed = terminate_all()
            if killed:
                self._feed_write(
                    self._selected or "",
                    f"[yellow]Stopped {killed} running training process(es) on quit.[/yellow]",
                )
        except Exception:
            pass
        self.exit()

    # --------------------------------------------------------------- credentials
    def _has_agent_key(self) -> bool:
        """True if the agent can authenticate: a stored/generic key, or the selected
        model's provider env var is already set (litellm resolves that itself)."""
        if resolve_api_key() is not None:
            return True
        model = self._selection.model if self._selection is not None else ""
        if not model:
            return False
        try:
            import litellm

            return bool(litellm.validate_environment(model).get("keys_in_environment"))
        except Exception:
            return False

    def action_manage_credentials(self) -> None:
        # One generic key drives whichever provider the model belongs to.
        self.push_screen(CredentialsScreen(self._credential_store))

    # --------------------------------------------------------------- model picker
    def action_choose_model(self) -> None:
        self.push_screen(ModelScreen(self._selection), self._on_model_chosen)

    def _on_model_chosen(self, choice: str | None) -> None:
        if choice is None:
            # Cancelled without a prior selection: a queued message can't proceed.
            if self._selection is None and self._pending_send is not None:
                name, _ = self._pending_send
                self._pending_send = None
                self._feed_write(
                    name, "[yellow]No model chosen — pick one (press m) to let the agent run.[/yellow]"
                )
            return
        self._selection = save_selection(choice)
        # Drop cached sessions so the next turn is driven by the newly-chosen model
        # (the conversation transcript itself is persisted and reloads transparently).
        self._sessions.clear()
        self._refresh_agent_title()
        if self._selected is not None:
            self._feed_write(self._selected, f"[green]Model set to [b]{choice}[/b].[/green]")
        if not self._has_agent_key():
            # No key yet — collect it, then resume any queued message on close.
            self.push_screen(CredentialsScreen(self._credential_store), self._on_creds_closed)
            return
        # Model picked and key present — run anything that was waiting on the choice.
        self._resume_pending()

    def action_new_project(self) -> None:
        self.push_screen(NewProjectScreen(self._existing_names()), self._on_new_project)

    def _on_new_project(self, result: tuple[str, str] | None) -> None:
        if result is None:
            return
        name, desc = result
        root = self._projects_dir / name
        spec = ProjectSpec(
            name=name,
            description=desc or "TODO: refine in chat",
            task_type=TaskType.image_classification,
        )
        try:
            create_project(root, spec)
        except FileExistsError:
            pass
        self._reload_projects()
        for item in self._project_items():
            if item.project_name == name:
                self.query_one("#project-list", ListView).index = self._project_items().index(item)
                self._select_project(item)
                break
        self._feed_write(name, f"[green]Created project [b]{name}[/b].[/green]")
        if desc:
            # Kick the agent off with the human's task description (DEFINE step).
            self._submit_to_agent(name, desc)
        else:
            self._feed_write(name, "[dim]Describe the task in the chat box to brief the agent.[/dim]")

    # ----------------------------------------------------------------- chat / agent
    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "new-project":
            self.action_new_project()

    def on_chat_input_submitted(self, event: ChatInput.Submitted) -> None:
        text = event.value.strip()
        event.textarea.clear()
        if not text:
            return
        if self._selected is None:
            self.query_one("#thinking", RichLog).write(
                "[yellow]Pick a project first (or press n to create one).[/yellow]"
            )
            return
        self._submit_to_agent(self._selected, text)

    def _submit_to_agent(self, project_name: str, text: str) -> None:
        self._feed_write(project_name, f"[b cyan]you ▸[/b cyan] {text}")

        # Model gate — Metis has no default provider, so the very first message
        # prompts the user to choose which LLM drives the agent. The message is
        # queued and resumes automatically once a model (and key) are set.
        if self._selection is None:
            self._feed_write(
                project_name,
                "[yellow]⚠ No model chosen yet. Opening the model picker — pick which LLM "
                "should drive Metis (no provider is preferred).[/yellow]",
            )
            self._pending_send = (project_name, text)
            self.action_choose_model()
            return

        # Credentials gate — this is what made "paste a token, nothing happens" feel
        # broken before: we now surface the missing key and resume once it's set.
        if not self._has_agent_key():
            self._feed_write(
                project_name,
                "[yellow]⚠ No API key set. Opening the token manager — "
                "paste your key, then close to continue.[/yellow]",
            )
            self._pending_send = (project_name, text)
            self.push_screen(CredentialsScreen(self._credential_store), self._on_creds_closed)
            return

        if project_name in self._busy:
            self._feed_write(
                project_name, "[yellow]Agent is still working on the previous turn…[/yellow]"
            )
            return

        self._run_agent_turn(project_name, text)

    def _on_creds_closed(self, has_key: bool | None) -> None:
        if not has_key:
            pending = self._pending_send
            self._pending_send = None
            msg = "[yellow]Still no API key — the agent can't run until one is set.[/yellow]"
            if pending is not None:
                self._feed_write(pending[0], msg)
            else:
                self.query_one("#thinking", RichLog).write(msg)
            return
        if self._pending_send is not None:
            self._feed_write(self._pending_send[0], "[green]Token saved.[/green]")
        else:
            self.query_one("#thinking", RichLog).write("[green]Token saved.[/green]")
        self._resume_pending()

    def _resume_pending(self) -> None:
        """Run a message that was queued behind the model / credentials gates."""
        pending = self._pending_send
        self._pending_send = None
        if pending is None:
            return
        name, text = pending
        if name in self._busy:
            self._feed_write(
                name, "[yellow]Agent is still working on the previous turn…[/yellow]"
            )
            return
        self._run_agent_turn(name, text)

    def _run_agent_turn(self, project_name: str, text: str) -> None:
        self._busy.add(project_name)
        self._activity[project_name] = "thinking…"
        self.run_worker(
            lambda: self._agent_worker(project_name, text),
            thread=True,
            exclusive=False,
            group=f"agent-{project_name}",
        )

    def _agent_worker(self, project_name: str, text: str) -> None:
        """Runs in a worker thread; all UI access goes through call_from_thread."""
        try:
            session = self._get_session(project_name)
        except Exception as exc:
            self.call_from_thread(self._agent_error, project_name, str(exc))
            return

        def on_event(event: dict[str, Any]) -> None:
            self.call_from_thread(self._on_agent_event, project_name, event)

        try:
            session.send(text, on_event=on_event)
        except Exception as exc:
            self.call_from_thread(self._agent_error, project_name, str(exc))
            return
        self.call_from_thread(self._agent_done, project_name)

    def _get_session(self, project_name: str) -> Any:
        session = self._sessions.get(project_name)
        if session is not None:
            return session
        from metis.agent.litellm_client import build_client
        from metis.agent.session import AgentSession

        if self._selection is None:  # guarded by the model gate in _submit_to_agent
            raise RuntimeError("No model selected — pick one before running the agent.")
        root = self._projects_dir / project_name
        client = build_client(self._selection.model)
        session = AgentSession(project_root=root, client=client)
        self._sessions[project_name] = session
        return session

    # ---- callbacks marshalled back onto the UI thread -----------------------
    def _on_agent_event(self, project_name: str, event: dict[str, Any]) -> None:
        etype = event.get("type")
        if etype == "text":
            raw = (event.get("text") or "").strip()
            if not raw:
                return
            # Show the first sentence as a concise goal summary.
            first = raw.split("\n")[0].strip()
            for sep in ".!?:":
                idx = first.find(sep)
                if 0 < idx < 110:
                    first = first[: idx + 1]
                    break
            if len(first) > 110:
                first = first[:107] + "…"
            elif len(raw) > len(first) + 1:
                first += "…"
            self._feed_write(project_name, f"[italic dim]{first}[/italic dim]")
            self._activity[project_name] = "planning…"
        elif etype == "tool_call":
            name = event.get("name", "?")
            args = event.get("input") or {}
            label = _tool_feed_label(name, args)
            self._activity[project_name] = f"{label}…"
            self._feed_write(project_name, f"[magenta]⚙[/magenta] [white]{label}[/white]")
            if name == "run_python":
                script = args.get("script", "?")
                self._train_write(
                    project_name, f"[b green]▶ {_script_to_label(str(script))}[/b green]"
                )
        elif etype == "usage":
            usage = event.get("usage")
            if isinstance(usage, Usage):
                self._accumulate_usage(project_name, usage)
        elif etype == "train_output":
            # Live epoch / training stdout — route to the dedicated (persisted) box.
            line = event.get("line", "")
            low = line.lower()
            # Highlight epoch progress and score lines; dim everything else.
            is_progress = (
                ("epoch" in low and any(k in low for k in ("loss", "acc", "auc")))
                or "best" in low
                or "✓" in line
            )
            style = "green" if is_progress else "dim"
            self._train_write(project_name, f"[{style}]{line}[/{style}]")
        elif etype == "tool_result":
            result_text = _format_tool_result(event.get("result"))
            self._feed_write(project_name, f"[dim]  ↳ {result_text}[/dim]")
            if event.get("name") in {"submit_for_benchmark", "request_prune"}:
                self._load_leaderboard(self._projects_dir / project_name)

    def _agent_error(self, project_name: str, message: str) -> None:
        self._feed_write(project_name, f"[red]agent error: {message}[/red]")
        self._busy.discard(project_name)
        self._activity.pop(project_name, None)
        self._set_status(project_name, "error — see feed")
        self._refresh_agent_title()

    def _agent_done(self, project_name: str) -> None:
        self._busy.discard(project_name)
        self._activity.pop(project_name, None)
        self._set_status(project_name, "done — awaiting guidance")
        self._refresh_agent_title()
        self._load_leaderboard(self._projects_dir / project_name)


def _tail(text: str, limit: int) -> str:
    """Collapse whitespace and keep the LAST ``limit`` chars (where errors usually are).

    Tool stdout/stderr is most informative at the end — a traceback's final line, the
    last log message — so we tail rather than head-truncate to avoid flooding the pane.
    """
    text = " ".join(text.split())
    return text if len(text) <= limit else "…" + text[-(limit - 1) :]


def _format_tool_result(result: object, limit: int = 300) -> str:
    """Render a tool result for the feed: pull a human message out of JSON, then tail it.

    Handlers may hand back a JSON blob; showing the raw JSON is noise. Surface the
    natural message field if there is one, fall back to compact key=value pairs, and
    tail-truncate so a long payload doesn't overload the pane.
    """
    text = str(result).strip()
    try:
        data = json.loads(text)
    except (ValueError, TypeError):
        return _tail(text, limit)
    if isinstance(data, dict):
        for key in ("message", "summary", "text", "detail", "result", "error"):
            value = data.get(key)
            if isinstance(value, str) and value:
                return _tail(value, limit)
        return _tail(", ".join(f"{k}={v}" for k, v in data.items()), limit)
    if isinstance(data, list):
        return _tail("; ".join(str(item) for item in data), limit)
    return _tail(str(data), limit)


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
