"""Persisted choice of which provider + model drives the agent.

Unlike credentials, the model choice is not a secret — it lives in a plain JSON
file (``~/.config/metis/model.json``, overridable with ``METIS_MODEL_CONFIG``).
Resolution order:

  1. ``METIS_PROVIDER`` / ``METIS_MODEL`` environment variables (handy for CI / scripts).
  2. The saved selection in the config file.

Crucially there is **no provider fallback**: if nothing is configured,
``load_selection`` returns ``None`` and the caller (the TUI) prompts the user to
pick a provider + model. Metis never silently picks a provider — neither
Anthropic nor OpenAI is preferred. Once a provider is chosen but no specific
model is, the provider's small ``default_model`` fills in (a model-size default,
not a provider preference). A user who picks once has it remembered across restarts.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from metis.agent.providers import PROVIDERS, provider_spec, resolve_model

ENV_PROVIDER = "METIS_PROVIDER"
ENV_MODEL = "METIS_MODEL"
ENV_MODEL_CONFIG = "METIS_MODEL_CONFIG"

_DEFAULT_FILE = Path.home() / ".config" / "metis" / "model.json"


@dataclass(frozen=True)
class ModelSelection:
    """A resolved (provider, model) pair the agent should be driven with."""

    provider: str
    model: str


def model_config_path() -> Path:
    """Where the model selection is persisted, honouring ``METIS_MODEL_CONFIG``."""
    override = os.environ.get(ENV_MODEL_CONFIG)
    return Path(override).expanduser() if override else _DEFAULT_FILE


def _read() -> dict[str, str]:
    try:
        data = json.loads(model_config_path().read_text())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def save_selection(provider: str, model: str | None = None) -> ModelSelection:
    """Persist the chosen provider + model (validating the provider exists)."""
    spec = provider_spec(provider)
    chosen = resolve_model(provider, model)
    path = model_config_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"provider": spec.name, "model": chosen}))
    except OSError:
        pass
    return ModelSelection(spec.name, chosen)


def load_selection() -> ModelSelection | None:
    """Resolve the active selection: env vars > saved file, else ``None``.

    Returns ``None`` when no provider has been configured (no env var, nothing
    saved, or a saved provider that no longer exists) so the caller can prompt
    for an explicit pick. There is deliberately no provider fallback.
    """
    provider = os.environ.get(ENV_PROVIDER) or _read().get("provider")
    if not provider or provider not in PROVIDERS:
        return None
    model = os.environ.get(ENV_MODEL) or _read().get("model")
    # Drop a saved model that doesn't belong to the resolved provider.
    spec = provider_spec(provider)
    if model and all(m.id != model for m in spec.models):
        model = None
    return ModelSelection(provider, resolve_model(provider, model))
