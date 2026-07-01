"""Tests for the litellm-backed agent client.

No network calls are made: ``litellm.completion`` is monkeypatched to capture the
request and return a canned response, and ``supports_prompt_caching`` is patched to
drive the cache-control branch both ways.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import metis.agent.litellm_client as lc
from metis.agent.litellm_client import LiteLLMClient, _to_openai_messages, build_client
from metis.agent.tools import ToolSpec


def _isolate_credentials(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("METIS_API_KEY", raising=False)
    monkeypatch.setenv("METIS_CREDENTIALS_FILE", str(tmp_path / "creds.json"))


# ------------------------------------------------------------- translation


def test_translation_maps_tool_use_and_results() -> None:
    msgs = [
        {"role": "user", "content": "hello"},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "checking"},
                {"type": "tool_use", "id": "t1", "name": "list_dir", "input": {"path": "."}},
            ],
        },
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}]},
    ]
    out = _to_openai_messages(msgs, "SYS")
    assert out[0] == {"role": "system", "content": "SYS"}
    assert out[1] == {"role": "user", "content": "hello"}
    assistant = out[2]
    assert assistant["role"] == "assistant"
    assert assistant["tool_calls"][0]["function"]["name"] == "list_dir"
    assert assistant["tool_calls"][0]["id"] == "t1"
    assert out[3] == {"role": "tool", "tool_call_id": "t1", "content": "ok"}


# ------------------------------------------------------------------- send


def _canned_response() -> object:
    tool_call = SimpleNamespace(
        id="call_1",
        function=SimpleNamespace(name="list_dir", arguments='{"path": "."}'),
    )
    message = SimpleNamespace(content="thinking", tool_calls=[tool_call])
    usage = SimpleNamespace(
        prompt_tokens=100,
        completion_tokens=20,
        prompt_tokens_details=SimpleNamespace(cached_tokens=40, cache_creation_tokens=10),
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason="tool_calls")],
        usage=usage,
    )


def _patch_completion(monkeypatch: pytest.MonkeyPatch, response: object) -> dict:
    captured: dict = {}

    def fake_completion(**kwargs: object) -> object:
        captured.update(kwargs)
        return response

    monkeypatch.setattr(lc.litellm, "completion", fake_completion)
    return captured


def test_send_parses_tool_calls_and_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lc, "supports_prompt_caching", lambda _model: False)
    captured = _patch_completion(monkeypatch, _canned_response())

    client = LiteLLMClient("gpt-4o-mini", api_key="sk-explicit")
    reply = client.send([{"role": "user", "content": "go"}], [], system="SYS")

    assert reply.text == "thinking"
    assert reply.stop_reason == "tool_use"
    assert reply.tool_calls[0].name == "list_dir"
    assert reply.tool_calls[0].input == {"path": "."}
    # prompt_tokens is the full total; uncached remainder = 100 - 40 - 10.
    assert reply.usage.input_tokens == 50
    assert reply.usage.cache_read_input_tokens == 40
    assert reply.usage.cache_creation_input_tokens == 10
    assert reply.usage.output_tokens == 20
    assert captured["model"] == "gpt-4o-mini"
    assert captured["api_key"] == "sk-explicit"


# ------------------------------------------------------------- prompt caching


def _tools() -> list[ToolSpec]:
    return [
        ToolSpec(name="t1", description="d1", input_schema={"type": "object"}, handler=lambda _: ""),
        ToolSpec(name="t2", description="d2", input_schema={"type": "object"}, handler=lambda _: ""),
    ]


def test_caching_marks_prefixes_when_supported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lc, "supports_prompt_caching", lambda _model: True)
    captured = _patch_completion(monkeypatch, _canned_response())

    client = LiteLLMClient("anthropic/claude-haiku-4-5", api_key="sk-explicit")
    messages = [{"role": "user", "content": "go"}]
    client.send(messages, _tools(), system="SYS")

    ttl = {"type": "ephemeral", "ttl": "1h"}
    sent_messages = captured["messages"]
    # (a) system promoted to a cached text block
    assert sent_messages[0]["content"] == [{"type": "text", "text": "SYS", "cache_control": ttl}]
    # (b) last message's final block carries cache_control
    assert sent_messages[-1]["content"][-1]["cache_control"] == ttl
    # (c) last tool carries cache_control, earlier tool does not
    assert captured["tools"][-1]["cache_control"] == ttl
    assert "cache_control" not in captured["tools"][0]
    # (d) caller's messages list is not mutated
    assert messages == [{"role": "user", "content": "go"}]


def test_no_caching_when_unsupported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lc, "supports_prompt_caching", lambda _model: False)
    captured = _patch_completion(monkeypatch, _canned_response())

    client = LiteLLMClient("gpt-4o", api_key="sk-explicit")
    client.send([{"role": "user", "content": "go"}], _tools(), system="SYS")

    assert captured["messages"][0] == {"role": "system", "content": "SYS"}
    assert "cache_control" not in captured["tools"][-1]


# --------------------------------------------------------------- credentials


def test_resolve_api_key_prefers_explicit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _isolate_credentials(monkeypatch, tmp_path)
    client = LiteLLMClient("gpt-4o", api_key="sk-explicit")
    assert client._api_key == "sk-explicit"


def test_resolve_api_key_none_when_unset(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # No key anywhere: resolve to None (litellm falls back to provider env vars),
    # never raising — construction still succeeds.
    _isolate_credentials(monkeypatch, tmp_path)
    client = LiteLLMClient("gpt-4o")
    assert client._api_key is None


def test_build_client_returns_litellm_client(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _isolate_credentials(monkeypatch, tmp_path)
    client = build_client("gpt-4o", api_key="sk-explicit")
    assert isinstance(client, LiteLLMClient)
    assert client.model == "gpt-4o"
