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

from metis.agent.providers import provider_spec
from metis.paths import credentials_path

#: Legacy default env-var name for the bare ``EnvCredentialProvider`` helper. Each
#: provider has its own env var (see ``providers.py``) and ``credential_provider_for``
#: always passes the right one explicitly — this constant is only the fallback for a
#: ``EnvCredentialProvider()`` constructed with no argument, and implies no provider
#: preference.
ENV_VAR = "ANTHROPIC_API_KEY"
#: Optional override for the on-disk credentials file location (handy in tests).
ENV_CREDENTIALS_FILE = "METIS_CREDENTIALS_FILE"


def _field_for(provider: str) -> str:
    """The JSON field a provider's key is stored under in the credentials file."""
    return provider_spec(provider).credential_field


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

    A real validation requires a network round-trip (stubbed for now); this only
    rejects obviously-empty/too-short values so the UI can give fast feedback
    without ever transmitting the secret.
    """
    if not key:
        return False
    return len(key.strip()) >= 8


@dataclass
class FileCredentialStore:
    """Read/write per-provider API keys in a ``0600`` JSON file owned by the user.

    The file maps a provider's credential field (e.g. ``anthropic_api_key``,
    ``openai_api_key``) to its secret, so multiple providers' keys can coexist.
    Every method takes an explicit ``provider`` — the store has no default
    provider, mirroring the harness's neutrality between Anthropic and OpenAI.
    """

    path: Path = field(default_factory=default_credentials_path)

    def _read(self) -> dict[str, str]:
        try:
            data = json.loads(self.path.read_text())
        except (FileNotFoundError, ValueError, OSError):
            return {}
        return data if isinstance(data, dict) else {}

    def get(self, provider: str) -> str | None:
        key = self._read().get(_field_for(provider))
        return key or None

    def set(self, api_key: str, provider: str) -> None:
        """Persist *api_key* for *provider* with owner-only permissions.

        Other providers' stored keys are preserved. The file is created via
        ``os.open`` with mode ``0600`` so the secret is never momentarily
        world-readable, and existing files are re-``chmod``'d to ``0600``.
        """
        key = (api_key or "").strip()
        if not key:
            raise ValueError("refusing to store an empty API key")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(self.path.parent, 0o700)
        except OSError:
            pass  # best-effort dir tightening
        data = self._read()
        data[_field_for(provider)] = key
        payload = json.dumps(data)
        fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(payload)
        try:
            os.chmod(self.path, 0o600)
        except OSError:
            pass

    def clear(self, provider: str) -> bool:
        """Delete *provider*'s stored key. Returns True if a key was removed.

        Removes the whole file once no provider keys remain, so a fully-cleared
        store leaves nothing behind on disk.
        """
        data = self._read()
        existed = data.pop(_field_for(provider), None) is not None
        if not data:
            removed = self.path.exists()
            self.path.unlink(missing_ok=True)
            return existed or removed
        if existed:
            payload = json.dumps(data)
            fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w") as f:
                f.write(payload)
        return existed

    def has(self, provider: str) -> bool:
        return self.get(provider) is not None


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
    """Reads a provider's key from a ``FileCredentialStore``."""

    def __init__(
        self,
        store: FileCredentialStore | None = None,
        provider: str = "anthropic",
    ) -> None:
        self._store = store or FileCredentialStore()
        self._provider = provider

    def get_api_key(self) -> str | None:
        return self._store.get(self._provider)


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


def credential_provider_for(provider: str) -> CredentialProvider:
    """Resolution order for *provider*: its env var, then the local file.

    Each provider has its own env var (``ANTHROPIC_API_KEY``, ``OPENAI_API_KEY``,
    …) and its own field in the credentials file, so keys never collide. The
    *provider* is always explicit — there is no default provider.
    """
    spec = provider_spec(provider)
    return ChainedCredentialProvider(
        [
            EnvCredentialProvider(spec.env_var),
            StoredCredentialProvider(provider=provider),
        ]
    )
