"""A multi-turn agent session driving one project end-to-end.

``AgentSession`` is the conversational front-end the TUI talks to. It owns:

  * the full toolset for a project (``build_agent_tools``),
  * the running message transcript (so the chat is multi-turn — the agent keeps
    context across the human's guidance messages),
  * the system prompt describing the whole Metis loop.

Each call to :meth:`send` runs the tool-use loop to completion for that turn,
streaming events to an optional callback, and folds the resulting transcript back
into ``history`` so the next message continues the same conversation.

The session never touches credentials directly — it is handed a constructed
``LLMClient`` (the secret lives behind the credentials boundary).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from metis.agent.client import LLMClient
from metis.agent.hardware import describe_hardware
from metis.agent.loop import AgentLoop, EventCallback
from metis.agent.tools import build_agent_tools

#: Per-project directory holding the resumable session state (history + display feed).
#: Lives alongside data/, models/, runs/ — NOT inside the benchmark/ lockbox.
SESSION_DIRNAME = "session"
_HISTORY_FILE = "history.json"
_FEED_FILE = "feed.json"
_TRAIN_FILE = "train.json"
#: Cap the persisted display feed so a long-running project's transcript can't grow
#: without bound on disk or flood the pane on reload.
_FEED_MAX_LINES = 500
#: The live training box is noisier (per-epoch stdout); cap it tighter so reloading
#: a project doesn't flood the pane with thousands of historical epoch lines.
_TRAIN_MAX_LINES = 300


def _session_dir(project_root: Path) -> Path:
    return project_root / SESSION_DIRNAME


def load_history(project_root: Path) -> list[dict[str, Any]]:
    """Load a previously persisted conversation transcript, or [] if none/corrupt."""
    path = _session_dir(project_root) / _HISTORY_FILE
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
    except (ValueError, OSError):
        return []
    return data if isinstance(data, list) else []


def save_history(project_root: Path, history: list[dict[str, Any]]) -> None:
    """Persist the conversation transcript so the agent resumes instead of restarting.

    Best-effort: a write failure must never crash an agent turn, so errors are swallowed.
    """
    try:
        directory = _session_dir(project_root)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / _HISTORY_FILE).write_text(json.dumps(history))
    except (OSError, TypeError):
        pass


def load_feed(project_root: Path) -> list[str]:
    """Load the persisted display feed (rendered chat/tool lines), or [] if none."""
    path = _session_dir(project_root) / _FEED_FILE
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
    except (ValueError, OSError):
        return []
    return [str(line) for line in data] if isinstance(data, list) else []


def save_feed(project_root: Path, feed: list[str]) -> None:
    """Persist the (tail-capped) display feed so re-entering a project restores the view."""
    try:
        directory = _session_dir(project_root)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / _FEED_FILE).write_text(json.dumps(feed[-_FEED_MAX_LINES:]))
    except (OSError, TypeError):
        pass


def load_train(project_root: Path) -> list[str]:
    """Load the persisted training-output lines (rendered markup), or [] if none."""
    path = _session_dir(project_root) / _TRAIN_FILE
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
    except (ValueError, OSError):
        return []
    return [str(line) for line in data] if isinstance(data, list) else []


def save_train(project_root: Path, lines: list[str]) -> None:
    """Persist the (tail-capped) training-output box so it survives a restart.

    Mirrors the display feed: the training pane was previously ephemeral (wiped on
    every project switch), which lost the live epoch/score history the moment the
    user looked at another project.
    """
    try:
        directory = _session_dir(project_root)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / _TRAIN_FILE).write_text(json.dumps(lines[-_TRAIN_MAX_LINES:]))
    except (OSError, TypeError):
        pass

MAIN_SYSTEM_PROMPT = """You are Metis, an agent that designs, trains, benchmarks \
and refines small, efficient, task-specific models — NOT large language models. \
The models you produce are compact classifiers/regressors (small CNNs, \
gradient-boosted trees, classical ML, tiny transformers).

You operate inside a single project directory through your tools. You drive the \
whole loop, and the human watches and steers you via chat:

1. DEFINE — If project.yaml is missing or incomplete, turn the human's task \
description into a complete definition and save it with save_project_spec.
2. DATA — The human PROVIDES their own data; you do NOT source or scrape it. \
The human places raw data under data/raw/<dataset>/. Call ingest_dataset to \
de-dup, validate and split it (the harness seals the test holdout you can never \
see). If data/raw/ is empty, tell the human you need them to add their dataset.
3. PROPOSE — Propose a BREADTH of candidate model families suited to the data. \
FIRST call list_model_templates and prefer instantiate_template to scaffold proven \
families (sklearn for tabular/flat data, torch CNNs for image data) — do not browse \
the dataset file-by-file or write architectures from scratch when a template fits.
4. TRAIN — Prefer instantiate_template + run_python. Only hand-write a train.py if no \
template fits. The base runtime has numpy + scikit-learn; torch/torchvision are only \
present if the 'ml' extra is installed, so do NOT write torch/CUDA code unless a torch \
template is available (the torch image templates already handle this).
5. BENCHMARK — Call submit_for_benchmark per trained variant. The harness scores \
it on the sealed holdout and returns accuracy + efficiency metrics. You never \
see the holdout and cannot grade yourself — this is the anti-gaming guarantee.
6. RANK — Read get_leaderboard; ranking follows the project's objective.
7. PRUNE — request_prune the weakest variants.
8. BRANCH — check_plateau; when plateaued, mutate top performers or try new \
families. Watch get_budget_status and STOP when budgets are exhausted.

DATA PIPELINE — READ THIS BEFORE TOUCHING DATA:
• Check data/processed/X.npy FIRST. If it already exists (from user pre-processing or \
a prior run), skip ingest_dataset entirely and go straight to PROPOSE → TRAIN. \
The harness automatically seals the holdout on the first run_python call — you \
do not need to call ingest_dataset to trigger sealing.
• Call ingest_dataset only if data/processed/ is empty. It reads X.npy + y.npy from \
data/raw/<dataset>/, de-dups, validates, splits, and seals the holdout.
• If ingest_dataset errors (e.g. the user ran their own pre-processing scripts that \
placed data directly in data/processed/), check whether data/processed/X.npy now \
exists. If it does, the data is ready — proceed to PROPOSE → TRAIN.

MULTI-LABEL CLASSIFICATION (y has shape [N, C] with C > 1):
• ingest_dataset handles multi-label y correctly. If it errors for another reason, \
check data/processed/ for existing data as above.
• Template train.py files generated by instantiate_template use nn.CrossEntropyLoss, \
which is for SINGLE-LABEL tasks only. For multi-label data (y.npy shape [N, C]):
  - Read the generated train.py immediately after scaffolding.
  - Replace nn.CrossEntropyLoss() → nn.BCEWithLogitsLoss().
  - Keep labels as float32 — remove any .long() cast on y.
  - At inference time use sigmoid + threshold (not argmax).
  Perform this edit BEFORE calling run_python on any template-generated train.py.
• Verify the number of output classes matches y.shape[1], not len(project classes).

BE EFFICIENT — each tool call costs a turn:
• Do not re-read the same file twice; do not re-run data inspection scripts.
• Check what already exists (list_dir, read_file) before recreating it.
• One run_python per model variant — do not retry a failed training script by \
re-running the same file; diagnose the error, fix train.py, then run again.

Efficiency is first-class: a smaller/faster model that is slightly less accurate \
often wins. Keep the human informed in plain language: say what you are about to \
do and why before doing it. When the human sends guidance, incorporate it. Do \
not ask permission for every step — proceed autonomously, but honour the human's \
direction and budgets."""


def build_main_system_prompt() -> str:
    """The main loop prompt with a live description of the host hardware appended.

    Detecting the chip/RAM/accelerator at runtime (rather than baking a static
    string) lets the agent pick a training device and architecture that actually
    fit the machine — e.g. reach for the Apple-Silicon GPU via MPS instead of
    wrongly writing off the CPU as 'too slow'.
    """
    return f"{MAIN_SYSTEM_PROMPT}\n\n{describe_hardware()}"


@dataclass
class AgentSession:
    """One conversational, multi-turn agent run over a single project."""

    project_root: Path
    client: LLMClient
    system: str = field(default_factory=build_main_system_prompt)
    max_turns: int = 40
    history: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        # Resume a prior run: if the caller didn't seed history, load it from session/
        # so quitting and re-entering doesn't make the agent start from scratch.
        if not self.history:
            self.history = load_history(self.project_root)

    def send(self, user_message: str, *, on_event: EventCallback | None = None) -> str:
        """Run one chat turn to completion; return the agent's final text.

        The transcript is appended to ``history`` so the next call continues the
        same conversation with full context, and persisted to session/ so the
        conversation survives a restart of the TUI.
        """
        loop = AgentLoop(
            client=self.client,
            system=self.system,
            tools=build_agent_tools(self.project_root, on_event=on_event),
            max_turns=self.max_turns,
            on_event=on_event,
            # Persist after every step so quitting mid-training (which kills the
            # worker before send() returns) still leaves a recent transcript on disk.
            on_step=lambda messages: save_history(self.project_root, messages),
        )
        self.history = _elide_stale_tool_results(loop.run(user_message, history=self.history))
        save_history(self.project_root, self.history)
        return _last_assistant_text(self.history)


# Stale large tool outputs (file dumps, dir listings) bloat the rolling history
# that re-feeds the model every turn. Once a tool result is several turns old it is
# rarely needed verbatim, so elide its body to a stub while keeping recent ones intact.
_ELIDE_KEEP_RECENT = 6
_ELIDE_MIN_CHARS = 500
_ELIDE_STUB = "[older tool result elided to save context]"


def _elide_stale_tool_results(
    history: list[dict[str, Any]],
    *,
    keep_recent: int = _ELIDE_KEEP_RECENT,
) -> list[dict[str, Any]]:
    """Replace the bodies of tool_result blocks older than the last ``keep_recent`` turns."""
    tr_indices = [
        i
        for i, msg in enumerate(history)
        if msg.get("role") == "user"
        and isinstance(msg.get("content"), list)
        and any(isinstance(b, dict) and b.get("type") == "tool_result" for b in msg["content"])
    ]
    if len(tr_indices) <= keep_recent:
        return history
    for i in tr_indices[:-keep_recent]:
        for block in history[i]["content"]:
            if not (isinstance(block, dict) and block.get("type") == "tool_result"):
                continue
            content = block.get("content")
            if isinstance(content, str) and len(content) > _ELIDE_MIN_CHARS:
                block["content"] = _ELIDE_STUB
    return history


def _last_assistant_text(messages: list[dict[str, Any]]) -> str:
    """Extract the final assistant text block from a transcript (best-effort)."""
    for msg in reversed(messages):
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            texts = [
                b.get("text", "")
                for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            ]
            joined = "".join(texts).strip()
            if joined:
                return joined
    return ""
