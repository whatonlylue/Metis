"""Persisted choice of which model drives the agent.

Metis is litellm-native: the agent is driven by a single free-form model string
(``anthropic/claude-opus-4-8``, ``gpt-4o``, ``gemini/gemini-1.5-pro``, …) that
litellm routes to the right provider. Unlike credentials, this choice is not a
secret — it lives in a plain JSON file (``~/.config/metis/model.json``,
overridable with ``METIS_MODEL_CONFIG``). Resolution order:

  1. ``METIS_MODEL`` environment variable (handy for CI / scripts).
  2. The saved selection in the config file.

There is deliberately no default model: if nothing is configured,
``load_selection`` returns ``None`` and the caller (the TUI) prompts the user to
pick one. A user who picks once has it remembered across restarts.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from metis.paths import model_config_path as _default_model_config_path

ENV_MODEL = "METIS_MODEL"
ENV_MODEL_CONFIG = "METIS_MODEL_CONFIG"


@dataclass(frozen=True)
class ModelSelection:
    """The resolved model string the agent should be driven with."""

    model: str


def model_config_path() -> Path:
    """Where the model selection is persisted, honouring ``METIS_MODEL_CONFIG``."""
    override = os.environ.get(ENV_MODEL_CONFIG)
    return Path(override).expanduser() if override else _default_model_config_path()


def _read() -> dict[str, str]:
    try:
        data = json.loads(model_config_path().read_text())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def save_selection(model: str) -> ModelSelection:
    """Persist the chosen model string, returning the resolved selection."""
    chosen = model.strip()
    if not chosen:
        raise ValueError("refusing to save an empty model string")
    path = model_config_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"model": chosen}))
    except OSError:
        pass
    return ModelSelection(chosen)


def load_selection() -> ModelSelection | None:
    """Resolve the active selection: ``METIS_MODEL`` env > saved file, else ``None``.

    Returns ``None`` when nothing has been configured so the caller can prompt for
    an explicit pick. There is deliberately no default model.
    """
    model = os.environ.get(ENV_MODEL) or _read().get("model")
    if not model or not model.strip():
        return None
    return ModelSelection(model.strip())
