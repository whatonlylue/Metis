"""Tests for the stubbed credentials boundary in the Anthropic client.

No real network calls are made: ``anthropic.Anthropic(api_key=...)``'s
constructor doesn't hit the network, so these only exercise key resolution.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from types import SimpleNamespace

from metis.agent.anthropic_client import (
    AnthropicClient,
    CredentialError,
    _resolve_api_key,
    _with_rolling_cache_breakpoint,
)
from metis.agent.tools import ToolSpec


def _isolate_credentials(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Point credential resolution at an empty temp file so the test never reads
    the developer's real ~/.config/metis/credentials.json."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("METIS_CREDENTIALS_FILE", str(tmp_path / "creds.json"))


def test_resolve_api_key_raises_without_key_or_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _isolate_credentials(monkeypatch, tmp_path)
    with pytest.raises(CredentialError):
        _resolve_api_key(None)


def test_resolve_api_key_uses_explicit_argument(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert _resolve_api_key("sk-explicit") == "sk-explicit"


def test_resolve_api_key_falls_back_to_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-from-env")
    assert _resolve_api_key(None) == "sk-from-env"


def test_client_construction_raises_without_credentials(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _isolate_credentials(monkeypatch, tmp_path)
    with pytest.raises(CredentialError):
        AnthropicClient()


def test_client_construction_succeeds_with_explicit_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    client = AnthropicClient(api_key="sk-explicit")
    assert client is not None


# --------------------------------------------------------------- prompt caching


def test_rolling_breakpoint_promotes_string_without_mutating_caller() -> None:
    messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
    ]
    out = _with_rolling_cache_breakpoint(messages)
    # Caller's list and dicts are untouched.
    assert messages[-1]["content"] == "hi there"
    # Last message's string content is promoted to a cached text block, with the
    # 1-hour TTL we use for long-running training turns.
    assert out[-1]["content"] == [
        {"type": "text", "text": "hi there", "cache_control": {"type": "ephemeral", "ttl": "1h"}}
    ]
    # Earlier messages are passed through by identity (no needless copying).
    assert out[0] is messages[0]


def test_rolling_breakpoint_marks_final_structured_block() -> None:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "a", "content": "x"},
                {"type": "tool_result", "tool_use_id": "b", "content": "y"},
            ],
        }
    ]
    out = _with_rolling_cache_breakpoint(messages)
    assert out[-1]["content"][-1]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
    assert "cache_control" not in out[-1]["content"][0]
    # Caller untouched.
    assert "cache_control" not in messages[-1]["content"][-1]


def test_send_sets_cache_control_and_preserves_messages(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    client = AnthropicClient(api_key="sk-explicit")

    captured: dict[str, object] = {}

    def fake_create(**kwargs: object) -> object:
        captured.update(kwargs)
        return SimpleNamespace(content=[], stop_reason="end_turn")

    monkeypatch.setattr(client._client.messages, "create", fake_create)

    tools = [
        ToolSpec(
            name="t1", description="d1", input_schema={"type": "object"}, handler=lambda _: ""
        ),
        ToolSpec(
            name="t2", description="d2", input_schema={"type": "object"}, handler=lambda _: ""
        ),
    ]
    messages = [{"role": "user", "content": "go"}]
    client.send(messages, tools, system="SYS")

    # (a) system is a list block carrying a 1-hour cache_control breakpoint
    assert captured["system"] == [
        {"type": "text", "text": "SYS", "cache_control": {"type": "ephemeral", "ttl": "1h"}}
    ]
    # (b) last tool carries cache_control, earlier tool does not
    sent_tools = captured["tools"]
    assert sent_tools[-1]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}  # type: ignore[index]
    assert "cache_control" not in sent_tools[0]  # type: ignore[operator]
    # (c) last message's final block carries cache_control
    sent_messages = captured["messages"]
    assert sent_messages[-1]["content"][-1]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}  # type: ignore[index]
    # (d) caller's messages list is not mutated
    assert messages == [{"role": "user", "content": "go"}]
