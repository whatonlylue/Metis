"""PRUNE: drop the weakest variants from the active search (harness-side).

Pruning is a harness operation that reads ``results.db`` and *marks* the losing
variants pruned rather than deleting anything. Per CLAUDE.md's reproducibility
principle the recorded recipe, weights, and benchmark history are preserved, so a
prune is fully reversible (``store.mark_pruned(..., pruned=False)``). Pruned
variants drop out of the default leaderboard but their results remain on record.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

from metis.benchmark.store import get_leaderboard, mark_pruned
from metis.projects import load_project


def select_for_pruning(
    ranked_variant_ids: Sequence[str],
    *,
    keep_top_k: int | None = None,
    drop_bottom_fraction: float | None = None,
) -> list[str]:
    """Choose variant ids to prune from a best-first ranked list.

    ``keep_top_k`` takes precedence: keep the first k, prune the rest. Otherwise
    ``drop_bottom_fraction`` prunes that fraction of the tail (rounded down, never
    pruning the single best). If neither is set, nothing is pruned.
    """
    ids = list(ranked_variant_ids)
    n = len(ids)
    if keep_top_k is not None:
        if keep_top_k < 0:
            raise ValueError("keep_top_k must be >= 0")
        return ids[keep_top_k:]
    if drop_bottom_fraction is not None:
        if not 0.0 <= drop_bottom_fraction <= 1.0:
            raise ValueError("drop_bottom_fraction must be in [0, 1]")
        k_drop = math.floor(n * drop_bottom_fraction)
        k_drop = min(k_drop, max(0, n - 1))  # always keep at least the top variant
        return ids[n - k_drop :] if k_drop > 0 else []
    return []


def prune_project(
    project_root: Path,
    *,
    keep_top_k: int | None = None,
    drop_bottom_fraction: float | None = None,
    reason: str | None = None,
) -> list[str]:
    """Rank the current leaderboard and mark the weakest variants pruned.

    Policy comes from the project's ``prune_policy`` unless overridden by the
    keyword args. Returns the variant ids that were pruned.
    """
    spec = load_project(project_root)
    policy = spec.prune_policy
    if keep_top_k is None and drop_bottom_fraction is None:
        keep_top_k = policy.keep_top_k
        drop_bottom_fraction = policy.drop_bottom_fraction

    rows = get_leaderboard(
        project_root / "benchmark",
        task_metric_name=spec.target_metric,
        n=10_000,
    )
    ranked_ids = [str(r["variant_id"]) for r in rows]
    to_prune = select_for_pruning(
        ranked_ids,
        keep_top_k=keep_top_k,
        drop_bottom_fraction=drop_bottom_fraction,
    )
    if to_prune:
        mark_pruned(
            project_root / "benchmark",
            to_prune,
            pruned=True,
            reason=reason or "pruned by policy",
        )
    return to_prune
