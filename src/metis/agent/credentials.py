"""Credentials boundary: where the agent client obtains its API key.

Metis is litellm-native, so the agent is driven by a single API key plus a
free-form model string (``anthropic/claude-opus-4-8``, ``gpt-4o``, …). litellm
routes the key to whichever provider the model belongs to, so we no longer keep a
separate key per provider — one generic key is stored here.

This is a thin, auditable auth boundary:

  * The secret is resolved through a chain of ``CredentialProvider``s
    (explicit > environment > local file). The client only depends on this
    interface, so a future OAuth flow can slot in behind ``OAuthCredentialProvider``
    without touching the loop or tools.
  * ``FileCredentialStore`` persists the key to a JSON file created with ``0600``
    permissions (owner read/write only). The secret is *never* logged, echoed,
    returned in masked-display helpers, or written anywhere the agent can read
    (``results.db``, ``runs/``, ``project.yaml``).
  * ``mask_key`` deliberately reveals no characters of the key — only whether one
    is present and its length — so it is safe to render in the TUI/logs.

If no key is stored here, ``resolve_api_key`` returns ``None`` and litellm falls
back to the provider's own env var (``ANTHROPIC_API_KEY``, ``OPENAI_API_KEY``, …),
so users who prefer environment variables keep working with no extra setup.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

from metis.paths import credentials_path

#: Generic env-var override for the agent's API key. Provider-specific env vars
#: (``ANTHROPIC_API_KEY`` etc.) are still honoured by litellm itself as a fallback.
ENV_VAR = "METIS_API_KEY"
#: Optional override for the on-disk credentials file location (handy in tests).
ENV_CREDENTIALS_FILE = "METIS_CREDENTIALS_FILE"
#: The single JSON field the credentials file stores the key under.
_FIELD = "api_key"


def default_credentials_path() -> Path:
    """Resolve the credentials-file path, honouring ``METIS_CREDENTIALS_FILE``."""
    override = os.environ.get(ENV_CREDENTIALS_FILE)
    return Path(override).expanduser() if override else credentials_path()


def mask_key(key: str | None) -> str:
    """Render a key for display WITHOUT revealing any of its characters.

    Returns a presence/length indicator only — safe to print to the TUI, the
    action log, or stdout. Never reproduce the raw secret anywhere.
    """
    if not key or not key.strip():
        return "(not set)"
    return f"present (••••, {len(key.strip())} chars)"


def looks_like_api_key(key: str | None) -> bool:
    """Cheap, offline sanity check used by the 'validate' UI action.

    A real validation requires a network round-trip; this only rejects
    obviously-empty/too-short values so the UI can give fast feedback without
    ever transmitting the secret.
    """
    if not key:
        return False
    return len(key.strip()) >= 8


@dataclass
class FileCredentialStore:
    """Read/write the agent's API key in a ``0600`` JSON file owned by the user."""

    path: Path = field(default_factory=default_credentials_path)

    def _read(self) -> dict[str, str]:
        try:
            data = json.loads(self.path.read_text())
        except (FileNotFoundError, ValueError, OSError):
            return {}
        return data if isinstance(data, dict) else {}

    def get(self) -> str | None:
        return self._read().get(_FIELD) or None

    def set(self, api_key: str) -> None:
        """Persist *api_key* with owner-only permissions.

        The file is created via ``os.open`` with mode ``0600`` so the secret is
        never momentarily world-readable, and existing files are re-``chmod``'d.
        """
        key = (api_key or "").strip()
        if not key:
            raise ValueError("refusing to store an empty API key")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(self.path.parent, 0o700)
        except OSError:
            pass  # best-effort dir tightening
        payload = json.dumps({_FIELD: key})
        fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(payload)
        try:
            os.chmod(self.path, 0o600)
        except OSError:
            pass

    def clear(self) -> bool:
        """Delete the stored key. Returns True if a key (or file) was removed."""
        existed = self.path.exists()
        self.path.unlink(missing_ok=True)
        return existed

    def has(self) -> bool:
        return self.get() is not None


@runtime_checkable
class CredentialProvider(Protocol):
    """The auth boundary the agent client depends on: hand me a key or None."""

    def get_api_key(self) -> str | None: ...


class EnvCredentialProvider:
    """Reads the key from an environment variable."""

    def __init__(self, var: str = ENV_VAR) -> None:
        self._var = var

    def get_api_key(self) -> str | None:
        return os.environ.get(self._var) or None


class StoredCredentialProvider:
    """Reads the key from a ``FileCredentialStore``."""

    def __init__(self, store: FileCredentialStore | None = None) -> None:
        self._store = store or FileCredentialStore()

    def get_api_key(self) -> str | None:
        return self._store.get()


class OAuthCredentialProvider:
    """Placeholder for a future OAuth flow — conforms to the interface only.

    It never yields a key today (so it is harmless in a provider chain) and the
    interactive handshake is explicitly not implemented yet.
    """

    def get_api_key(self) -> str | None:
        return None

    def begin_authorization(self) -> str:  # pragma: no cover - stub
        raise NotImplementedError(
            "OAuth flow is not implemented yet; use API-key auth (set a token in the "
            "TUI, or export METIS_API_KEY / your provider's key env var)."
        )


class ChainedCredentialProvider:
    """Tries each provider in order and returns the first non-empty key."""

    def __init__(self, providers: list[CredentialProvider]) -> None:
        self._providers = list(providers)

    def get_api_key(self) -> str | None:
        for provider in self._providers:
            key = provider.get_api_key()
            if key:
                return key
        return None


def default_credential_provider() -> CredentialProvider:
    """Resolution order for the agent key: ``METIS_API_KEY`` env, then local file."""
    return ChainedCredentialProvider(
        [EnvCredentialProvider(ENV_VAR), StoredCredentialProvider()]
    )


def resolve_api_key(
    explicit: str | None = None,
    provider: CredentialProvider | None = None,
) -> str | None:
    """Resolve the agent's API key: explicit argument > credential chain > ``None``.

    Returns ``None`` (never raises) when nothing is configured, letting litellm
    fall back to the provider's own env var. The value is never logged.
    """
    return explicit or (provider or default_credential_provider()).get_api_key()
