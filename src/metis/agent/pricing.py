"""Token pricing for the driving LLM, used to surface a live session cost.

Pricing is delegated to litellm's model database (``litellm.cost_per_token``),
which knows per-model input/output rates plus prompt-cache read/write rates for
every provider it supports. The harness uses this only to show the human a running
estimate, never to gate anything, so an unknown model simply costs ``0``.
"""

from __future__ import annotations

import litellm

from metis.agent.client import Usage


def cost_usd(usage: Usage, model: str) -> float:
    """Estimate the USD cost of one (or accumulated) ``Usage`` for ``model``.

    litellm's ``prompt_tokens`` is the *full* prompt total, so we hand it the
    uncached remainder plus both cache buckets and let litellm price the buckets
    (cache reads/writes) at their per-model rates. Unknown models estimate to 0.
    """
    try:
        prompt_cost, completion_cost = litellm.cost_per_token(
            model=model,
            prompt_tokens=(
                usage.input_tokens
                + usage.cache_creation_input_tokens
                + usage.cache_read_input_tokens
            ),
            completion_tokens=usage.output_tokens,
            cache_creation_input_tokens=usage.cache_creation_input_tokens,
            cache_read_input_tokens=usage.cache_read_input_tokens,
        )
    except Exception:
        return 0.0
    return prompt_cost + completion_cost
