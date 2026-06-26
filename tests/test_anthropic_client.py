"""Tests for the stubbed credentials boundary in the Anthropic client.

No real network calls are made: ``anthropic.Anthropic(api_key=...)``'s
constructor doesn't hit the network, so these only exercise key resolution.
"""

from __future__ import annotations

import pytest

from metis.agent.anthropic_client import AnthropicClient, CredentialError, _resolve_api_key


def test_resolve_api_key_raises_without_key_or_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(CredentialError):
        _resolve_api_key(None)


def test_resolve_api_key_uses_explicit_argument(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert _resolve_api_key("sk-explicit") == "sk-explicit"


def test_resolve_api_key_falls_back_to_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-from-env")
    assert _resolve_api_key(None) == "sk-from-env"


def test_client_construction_raises_without_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(CredentialError):
        AnthropicClient()


def test_client_construction_succeeds_with_explicit_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    client = AnthropicClient(api_key="sk-explicit")
    assert client is not None
