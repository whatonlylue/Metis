"""Tests for the provider-agnostic model layer: registry, selection, credentials,
and the OpenAI client's message translation.

No network calls are made. The OpenAI SDK is optional, so the one test that
constructs ``OpenAIClient`` injects a fake ``openai`` module into ``sys.modules``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from metis.agent.providers import (
    PROVIDERS,
    all_models,
    build_client,
    find_model,
    provider_spec,
    resolve_model,
)


# ------------------------------------------------------------------- registry


def test_no_default_provider_symbol_exists() -> None:
    # Metis is neutral between providers: there is deliberately no DEFAULT_PROVIDER.
    import metis.agent.providers as providers

    assert not hasattr(providers, "DEFAULT_PROVIDER")


def test_each_provider_has_a_small_default_model() -> None:
    for spec in PROVIDERS.values():
        # The per-provider default model must be one the provider actually lists.
        assert any(m.id == spec.default_model for m in spec.models)


def test_each_provider_default_is_the_cheapest_listed() -> None:
    # "Default to smaller options" — the default should be the cheapest by input price.
    for spec in PROVIDERS.values():
        cheapest = min(spec.models, key=lambda m: m.input_per_mtok)
        default = next(m for m in spec.models if m.id == spec.default_model)
        assert default.input_per_mtok == cheapest.input_per_mtok


def test_resolve_model_falls_back_to_default() -> None:
    assert resolve_model("openai", None) == provider_spec("openai").default_model
    assert resolve_model("openai", "gpt-4o") == "gpt-4o"


def test_find_model_longest_prefix_resolves_dated_snapshot() -> None:
    spec, model = find_model("claude-haiku-4-5-20251001")  # type: ignore[misc]
    assert spec.name == "anthropic"
    assert model.id == "claude-haiku-4-5"


def test_find_model_unknown_returns_none() -> None:
    assert find_model("nonexistent-model") is None


def test_provider_spec_unknown_raises() -> None:
    with pytest.raises(KeyError):
        provider_spec("gemini")


def test_all_models_covers_every_provider() -> None:
    names = {spec.name for spec, _ in all_models()}
    assert names == set(PROVIDERS)


# ----------------------------------------------------------------- selection


def _isolate_model_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("METIS_PROVIDER", raising=False)
    monkeypatch.delenv("METIS_MODEL", raising=False)
    monkeypatch.setenv("METIS_MODEL_CONFIG", str(tmp_path / "model.json"))


def test_selection_is_none_when_nothing_saved(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from metis.agent.model_config import load_selection

    _isolate_model_config(monkeypatch, tmp_path)
    # No provider preference: with nothing configured the user must pick.
    assert load_selection() is None


def test_selection_env_provider_uses_small_default_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from metis.agent.model_config import load_selection

    _isolate_model_config(monkeypatch, tmp_path)
    monkeypatch.setenv("METIS_PROVIDER", "openai")
    sel = load_selection()
    assert sel is not None
    assert sel.provider == "openai"
    assert sel.model == provider_spec("openai").default_model


def test_selection_roundtrips_through_disk(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from metis.agent.model_config import load_selection, save_selection

    _isolate_model_config(monkeypatch, tmp_path)
    save_selection("openai", "gpt-4o")
    sel = load_selection()
    assert sel is not None
    assert sel.provider == "openai"
    assert sel.model == "gpt-4o"


def test_env_vars_override_saved_selection(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from metis.agent.model_config import load_selection, save_selection

    _isolate_model_config(monkeypatch, tmp_path)
    save_selection("anthropic", "claude-opus-4-8")
    monkeypatch.setenv("METIS_PROVIDER", "openai")
    monkeypatch.setenv("METIS_MODEL", "gpt-4.1")
    sel = load_selection()
    assert sel is not None
    assert sel.provider == "openai"
    assert sel.model == "gpt-4.1"


def test_save_unknown_provider_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from metis.agent.model_config import save_selection

    _isolate_model_config(monkeypatch, tmp_path)
    with pytest.raises(KeyError):
        save_selection("gemini")


# --------------------------------------------------------------- credentials


def test_credentials_store_keeps_providers_separate(tmp_path: Path) -> None:
    from metis.agent.credentials import FileCredentialStore

    store = FileCredentialStore(path=tmp_path / "creds.json")
    store.set("sk-ant-123456", "anthropic")
    store.set("sk-oai-123456", "openai")

    assert store.get("anthropic") == "sk-ant-123456"
    assert store.get("openai") == "sk-oai-123456"

    # Clearing one provider leaves the other intact.
    assert store.clear("anthropic") is True
    assert store.get("anthropic") is None
    assert store.get("openai") == "sk-oai-123456"


def test_credential_provider_reads_per_provider_env_var(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from metis.agent.credentials import credential_provider_for

    monkeypatch.setenv("METIS_CREDENTIALS_FILE", str(tmp_path / "creds.json"))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-oai-from-env")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    assert credential_provider_for("openai").get_api_key() == "sk-oai-from-env"
    assert credential_provider_for("anthropic").get_api_key() is None


# ----------------------------------------------------- openai translation


def test_openai_translation_maps_tool_use_and_results() -> None:
    from metis.agent.openai_client import _to_openai_messages

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
    tool_msg = out[3]
    assert tool_msg == {"role": "tool", "tool_call_id": "t1", "content": "ok"}


def _install_fake_openai(monkeypatch: pytest.MonkeyPatch, response: object) -> dict:
    """Inject a stub ``openai`` module whose client records the request and returns
    ``response``. Returns a dict that captures the create() kwargs."""
    captured: dict = {}

    class _Completions:
        def create(self, **kwargs: object) -> object:
            captured.update(kwargs)
            return response

    class _Chat:
        completions = _Completions()

    class _OpenAI:
        def __init__(self, api_key: str) -> None:
            self.api_key = api_key
            self.chat = _Chat()

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=_OpenAI))
    return captured


def test_openai_client_send_parses_tool_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    from metis.agent.openai_client import OpenAIClient

    tool_call = SimpleNamespace(
        id="call_1",
        function=SimpleNamespace(name="list_dir", arguments='{"path": "."}'),
    )
    message = SimpleNamespace(content="thinking", tool_calls=[tool_call])
    usage = SimpleNamespace(
        prompt_tokens=100,
        completion_tokens=20,
        prompt_tokens_details=SimpleNamespace(cached_tokens=40),
    )
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason="tool_calls")],
        usage=usage,
    )
    captured = _install_fake_openai(monkeypatch, response)

    client = OpenAIClient(api_key="sk-oai-explicit", model="gpt-4o-mini")
    reply = client.send([{"role": "user", "content": "go"}], [], system="SYS")

    assert reply.text == "thinking"
    assert reply.stop_reason == "tool_use"
    assert reply.tool_calls[0].name == "list_dir"
    assert reply.tool_calls[0].input == {"path": "."}
    # Cached prompt tokens surface as cache reads; uncached remainder as input.
    assert reply.usage.cache_read_input_tokens == 40
    assert reply.usage.input_tokens == 60
    assert reply.usage.output_tokens == 20
    assert captured["model"] == "gpt-4o-mini"


def test_build_client_dispatches_to_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    from metis.agent.openai_client import OpenAIClient

    _install_fake_openai(monkeypatch, SimpleNamespace())
    client = build_client("openai", "gpt-4o", api_key="sk-oai-explicit")
    assert isinstance(client, OpenAIClient)
    assert client.model == "gpt-4o"
