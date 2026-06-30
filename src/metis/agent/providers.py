"""Registry of LLM providers and the models Metis can drive the agent with.

This is the single source of truth that makes Metis *provider-agnostic*: the TUI
model picker, the credentials boundary, the pricing table, and the client factory
all read from here. Adding a new provider (or model) means adding an entry to
``PROVIDERS`` and, if it speaks a new wire format, one new ``LLMClient`` — nothing
in the loop, tools, or session changes.

Metis is deliberately **neutral between providers** — there is no default
provider, and neither Anthropic nor OpenAI is preferred. The user always picks
which provider (and model) drives the agent; the harness imposes no preference.

Each provider lists a curated set of models with approximate USD prices (per 1M
tokens) used only for the live cost readout, and a ``default_model`` that is the
*smallest / cheapest* sensible option *within that provider* — so once a user has
chosen a provider but not a specific model, they still get an efficient model
rather than the priciest flagship. This is a model-size default, not a provider
preference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from metis.agent.client import LLMClient
    from metis.agent.credentials import CredentialProvider


@dataclass(frozen=True)
class ModelInfo:
    """One selectable model: its API id, a human label, and per-1M-token prices."""

    id: str
    label: str
    input_per_mtok: float
    output_per_mtok: float


@dataclass(frozen=True)
class ProviderSpec:
    """An LLM provider: how to authenticate it and which models it offers."""

    name: str
    label: str
    env_var: str
    #: JSON field the local credentials file stores this provider's key under.
    credential_field: str
    models: tuple[ModelInfo, ...]
    #: The smallest / cheapest model — the default when the user picks nothing.
    default_model: str


# Prices are approximate USD per 1M tokens and used only for the live session-cost
# estimate, never to gate anything. They are intentionally easy to update.
_ANTHROPIC = ProviderSpec(
    name="anthropic",
    label="Anthropic (Claude)",
    env_var="ANTHROPIC_API_KEY",
    credential_field="anthropic_api_key",
    default_model="claude-haiku-4-5",
    models=(
        ModelInfo("claude-haiku-4-5", "Claude Haiku 4.5 (small, fast)", 1.0, 5.0),
        ModelInfo("claude-sonnet-4-6", "Claude Sonnet 4.6 (balanced)", 3.0, 15.0),
        ModelInfo("claude-opus-4-8", "Claude Opus 4.8 (most capable)", 5.0, 25.0),
    ),
)

_OPENAI = ProviderSpec(
    name="openai",
    label="OpenAI (GPT)",
    env_var="OPENAI_API_KEY",
    credential_field="openai_api_key",
    default_model="gpt-4o-mini",
    models=(
        ModelInfo("gpt-4o-mini", "GPT-4o mini (small, fast)", 0.15, 0.60),
        ModelInfo("gpt-4.1-mini", "GPT-4.1 mini (balanced)", 0.40, 1.60),
        ModelInfo("gpt-4o", "GPT-4o (capable)", 2.50, 10.0),
        ModelInfo("gpt-4.1", "GPT-4.1 (most capable)", 2.0, 8.0),
    ),
)

#: Registry / display order in the model picker. This order is purely cosmetic
#: and implies **no default** — Metis never picks a provider for the user, who
#: must choose one explicitly before the agent can run.
PROVIDERS: dict[str, ProviderSpec] = {
    _ANTHROPIC.name: _ANTHROPIC,
    _OPENAI.name: _OPENAI,
}


def provider_spec(provider: str) -> ProviderSpec:
    """Look up a provider, raising a clear error for an unknown name."""
    try:
        return PROVIDERS[provider]
    except KeyError:
        raise KeyError(
            f"Unknown LLM provider {provider!r}. Known providers: {sorted(PROVIDERS)}"
        ) from None


def all_models() -> list[tuple[ProviderSpec, ModelInfo]]:
    """Every (provider, model) pair, in registry order — used by the model picker."""
    return [(spec, model) for spec in PROVIDERS.values() for model in spec.models]


def find_model(model_id: str) -> tuple[ProviderSpec, ModelInfo] | None:
    """Resolve a bare model id back to its (provider, model). Longest-prefix match
    so dated snapshots (e.g. ``claude-haiku-4-5-20251001``) still resolve."""
    best: tuple[int, ProviderSpec, ModelInfo] | None = None
    for spec in PROVIDERS.values():
        for model in spec.models:
            if model_id.startswith(model.id) and (best is None or len(model.id) > best[0]):
                best = (len(model.id), spec, model)
    if best is None:
        return None
    return best[1], best[2]


def resolve_model(provider: str, model: str | None) -> str:
    """Return ``model`` if given, else the provider's small default."""
    spec = provider_spec(provider)
    return model or spec.default_model


def build_client(
    provider: str,
    model: str | None = None,
    *,
    api_key: str | None = None,
    credential_provider: "CredentialProvider | None" = None,
    max_tokens: int | None = None,
) -> "LLMClient":
    """Construct the right ``LLMClient`` for ``provider``.

    Concrete client modules are imported lazily so that, e.g., a user who only
    ever drives Anthropic never needs the ``openai`` package installed.
    """
    spec = provider_spec(provider)
    chosen = resolve_model(provider, model)
    if credential_provider is None:
        from metis.agent.credentials import credential_provider_for

        credential_provider = credential_provider_for(provider)

    if provider == "anthropic":
        from metis.agent.anthropic_client import DEFAULT_MAX_TOKENS, AnthropicClient

        return AnthropicClient(
            api_key=api_key,
            model=chosen,
            max_tokens=max_tokens or DEFAULT_MAX_TOKENS,
            credential_provider=credential_provider,
        )
    if provider == "openai":
        from metis.agent.openai_client import DEFAULT_MAX_TOKENS, OpenAIClient

        return OpenAIClient(
            api_key=api_key,
            model=chosen,
            max_tokens=max_tokens or DEFAULT_MAX_TOKENS,
            credential_provider=credential_provider,
        )
    raise KeyError(f"No client implementation for provider {spec.name!r}")
