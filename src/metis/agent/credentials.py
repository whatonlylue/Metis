"""Credentials boundary: where the agent client obtains its API key.

This is the M6 replacement for the stubbed credentials handling in
``anthropic_client``. It is a thin, auditable auth boundary:

  * The secret is resolved through a chain of ``CredentialProvider``s
    (explicit > environment > local file). The agent client only depends on
    this interface, so a future real OAuth flow can be slotted in behind
    ``OAuthCredentialProvider`` without touching the loop or tools.
  * ``FileCredentialStore`` persists the key to a JSON file created with
    ``0600`` permissions (owner read/write only). The secret is *never* logged,
    echoed, returned in masked-display helpers, or written anywhere the agent
    can read (``results.db``, ``runs/``, ``project.yaml``).
  * ``mask_key`` deliberately reveals no characters of the key — only whether
    one is present and its length — so it is safe to render in the TUI/logs.

The full OAuth handshake is intentionally stubbed: ``OAuthCredentialProvider``
conforms to the interface but raises ``NotImplementedError`` from
``begin_authorization`` until a real flow is wired up.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

#: Environment variable holding the Anthropic API key (highest-priority fallback
#: after an explicit argument).
ENV_VAR = "ANTHROPIC_API_KEY"
#: Optional override for the on-disk credentials file location (handy in tests).
ENV_CREDENTIALS_FILE = "METIS_CREDENTIALS_FILE"

_DEFAULT_FILE = Path.home() / ".config" / "metis" / "credentials.json"
_KEY_FIELD = "anthropic_api_key"


def default_credentials_path() -> Path:
    """Resolve the credentials-file path, honouring ``METIS_CREDENTIALS_FILE``."""
    override = os.environ.get(ENV_CREDENTIALS_FILE)
    return Path(override).expanduser() if override else _DEFAULT_FILE


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

    A real validation requires a network round-trip (stubbed for now); this only
    rejects obviously-empty/too-short values so the UI can give fast feedback
    without ever transmitting the secret.
    """
    if not key:
        return False
    return len(key.strip()) >= 8


@dataclass
class FileCredentialStore:
    """Read/write the API key in a ``0600`` JSON file owned by the current user."""

    path: Path = field(default_factory=default_credentials_path)

    def get(self) -> str | None:
        try:
            data = json.loads(self.path.read_text())
        except (FileNotFoundError, ValueError, OSError):
            return None
        key = data.get(_KEY_FIELD) if isinstance(data, dict) else None
        return key or None

    def set(self, api_key: str) -> None:
        """Persist *api_key* with owner-only permissions.

        The file is created via ``os.open`` with mode ``0600`` so the secret is
        never momentarily world-readable, and existing files are re-``chmod``'d
        to ``0600`` defensively.
        """
        key = (api_key or "").strip()
        if not key:
            raise ValueError("refusing to store an empty API key")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(self.path.parent, 0o700)
        except OSError:
            pass  # best-effort dir tightening
        payload = json.dumps({_KEY_FIELD: key})
        fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(payload)
        try:
            os.chmod(self.path, 0o600)
        except OSError:
            pass

    def clear(self) -> bool:
        """Delete the stored key. Returns True if a file was removed."""
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
            "OAuth flow is not implemented yet; use API-key auth (set a token in the TUI "
            "or the ANTHROPIC_API_KEY env var)."
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
    """The standard resolution order: environment variable, then the local file."""
    return ChainedCredentialProvider([EnvCredentialProvider(), StoredCredentialProvider()])
