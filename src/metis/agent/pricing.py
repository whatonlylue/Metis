"""Token pricing for the driving LLM, used to surface a live session cost.

Prices are USD per 1M tokens, per the Anthropic pricing schedule. Prompt-cache
reads bill at ~0.1x the base input rate; cache writes at 1.25x (5-minute TTL) or
2x (1-hour TTL). Metis writes its caches with a 1-hour TTL (see
``anthropic_client._EPHEMERAL``), so the write multiplier here is 2x.

These are approximate and intentionally easy to update — the harness uses them
only to show the human a running estimate, never to gate anything.
"""

from __future__ import annotations

from dataclasses import dataclass

from metis.agent.client import Usage

CACHE_READ_MULTIPLIER = 0.1
CACHE_WRITE_MULTIPLIER = 2.0  # 1-hour TTL


@dataclass(frozen=True)
class ModelPrice:
    """Base per-1M-token input/output prices for a model."""

    input_per_mtok: float
    output_per_mtok: float


#: Fallback when the model id matches no known prefix (use small-model pricing).
_DEFAULT_PRICE = ModelPrice(1.0, 5.0)


def price_for(model: str) -> ModelPrice:
    """Resolve the base price for a model id via the provider registry.

    Longest-prefix match so dated snapshots and aliases (e.g.
    ``claude-haiku-4-5-20251001``) resolve to the right base price.
    """
    from metis.agent.providers import find_model

    found = find_model(model)
    if found is None:
        return _DEFAULT_PRICE
    _, info = found
    return ModelPrice(info.input_per_mtok, info.output_per_mtok)


def cost_usd(usage: Usage, model: str) -> float:
    """Estimate the USD cost of one (or accumulated) ``Usage`` for ``model``.

    Cache reads are charged at 0.1x base input; cache writes at 2x (1-hour TTL).
    """
    price = price_for(model)
    in_rate = price.input_per_mtok / 1_000_000
    out_rate = price.output_per_mtok / 1_000_000
    return (
        usage.input_tokens * in_rate
        + usage.cache_creation_input_tokens * in_rate * CACHE_WRITE_MULTIPLIER
        + usage.cache_read_input_tokens * in_rate * CACHE_READ_MULTIPLIER
        + usage.output_tokens * out_rate
    )
