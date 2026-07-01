"""Tests for the litellm-native model layer: selection persistence and the
generic credential store.

Metis no longer keeps a curated provider registry — the agent is driven by a
free-form litellm model string plus one generic API key. No network calls.
"""

from __future__ import annotations

from pathlib import Path

import pytest


# ----------------------------------------------------------------- selection


def _isolate_model_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("METIS_MODEL", raising=False)
    monkeypatch.setenv("METIS_MODEL_CONFIG", str(tmp_path / "model.json"))


def test_selection_is_none_when_nothing_saved(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from metis.agent.model_config import load_selection

    _isolate_model_config(monkeypatch, tmp_path)
    # No default model: with nothing configured the user must pick.
    assert load_selection() is None


def test_selection_roundtrips_through_disk(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from metis.agent.model_config import load_selection, save_selection

    _isolate_model_config(monkeypatch, tmp_path)
    save_selection("anthropic/claude-opus-4-8")
    sel = load_selection()
    assert sel is not None
    assert sel.model == "anthropic/claude-opus-4-8"


def test_env_var_overrides_saved_selection(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from metis.agent.model_config import load_selection, save_selection

    _isolate_model_config(monkeypatch, tmp_path)
    save_selection("anthropic/claude-opus-4-8")
    monkeypatch.setenv("METIS_MODEL", "gpt-4o")
    sel = load_selection()
    assert sel is not None
    assert sel.model == "gpt-4o"


def test_save_empty_model_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from metis.agent.model_config import save_selection

    _isolate_model_config(monkeypatch, tmp_path)
    with pytest.raises(ValueError):
        save_selection("   ")


# --------------------------------------------------------------- credentials


def test_credentials_store_roundtrips_a_single_key(tmp_path: Path) -> None:
    from metis.agent.credentials import FileCredentialStore

    store = FileCredentialStore(path=tmp_path / "creds.json")
    assert store.get() is None
    store.set("sk-generic-123456")
    assert store.get() == "sk-generic-123456"
    assert store.has() is True
    assert store.clear() is True
    assert store.get() is None


def test_credential_provider_reads_generic_env_var(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from metis.agent.credentials import default_credential_provider

    monkeypatch.setenv("METIS_CREDENTIALS_FILE", str(tmp_path / "creds.json"))
    monkeypatch.setenv("METIS_API_KEY", "sk-from-env")
    assert default_credential_provider().get_api_key() == "sk-from-env"
    monkeypatch.delenv("METIS_API_KEY", raising=False)
    assert default_credential_provider().get_api_key() is None
