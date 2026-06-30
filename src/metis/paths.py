"""Central resolution of Metis's per-user home directory.

Everything personal to a user lives under a single ``~/.metis/`` folder:

  * ``~/.metis/projects/``     — every project the agent works on.
  * ``~/.metis/credentials.json`` — API keys (``0600``, owner-only).
  * ``~/.metis/model.json``    — the chosen provider + model.
  * ``~/.metis/ui.json``       — TUI preferences (theme, …).

Consolidating under one home means an *installed* ``metis`` (e.g. via ``pipx``)
keeps a user's data in one predictable place no matter which directory they
launch it from — and it works identically on macOS, Linux, and Windows because
``Path.home()`` resolves the right location on each.

Override the whole root with the ``METIS_HOME`` environment variable (useful for
tests, sandboxes, or running multiple isolated instances). Individual files keep
their own finer-grained overrides (``METIS_CREDENTIALS_FILE``,
``METIS_MODEL_CONFIG``, ``METIS_UI_CONFIG``) which take precedence over this root.
"""

from __future__ import annotations

import os
from pathlib import Path

#: Environment variable to relocate the entire Metis home directory.
ENV_HOME = "METIS_HOME"


def metis_home() -> Path:
    """The root of all per-user Metis data, honouring ``METIS_HOME``.

    Defaults to ``~/.metis`` (cross-platform via :func:`Path.home`).
    """
    override = os.environ.get(ENV_HOME)
    return Path(override).expanduser() if override else Path.home() / ".metis"


def projects_dir() -> Path:
    """Directory holding every project tree (``~/.metis/projects``)."""
    return metis_home() / "projects"


def credentials_path() -> Path:
    """Default path for the API-key credentials file."""
    return metis_home() / "credentials.json"


def model_config_path() -> Path:
    """Default path for the persisted provider + model selection."""
    return metis_home() / "model.json"


def ui_config_path() -> Path:
    """Default path for persisted TUI preferences (theme, …)."""
    return metis_home() / "ui.json"
