"""Multi-metric ranking over benchmark results: Pareto frontier + weighted sum.

CLAUDE.md core principle #2 makes efficiency a first-class metric: we never rank
on accuracy alone unless the project asks for it. This module reads ranked rows
from the harness-side store (``get_leaderboard``) and annotates them with a
``pareto_rank`` (1 = non-dominated frontier) and, for the weighted objective, a
``weighted_score``. It lives harness-side; the agent only ever sees the result.

Objectives considered for efficiency (all "lower is better"):
  - param_count       — trainable parameters
  - model_size_mb     — serialized size on disk
  - latency_ms_p50    — single-sample inference latency

The task metric direction is taken from the store (rmse/mae/… are lower-better).
``throughput_sps`` is intentionally omitted from the default frontier because it
is largely the inverse of latency; it remains available for weighted sums.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from metis.benchmark.store import get_leaderboard, is_lower_better
from metis.projects import load_project
from metis.projects.schema import RankObjective

# Efficiency objectives and whether lower is better for each.
_EFFICIENCY_OBJECTIVES: tuple[tuple[str, bool], ...] = (
    ("param_count", True),
    ("model_size_mb", True),
    ("latency_ms_p50", True),
)


@dataclass(frozen=True)
class Objective:
    """One ranking axis: which row key to read and its preferred direction."""

    key: str
    lower_is_better: bool


def default_objectives(task_metric_name: str) -> list[Objective]:
    """The default accuracy-vs-efficiency objective set for a project."""
    objs = [Objective("task_metric_value", is_lower_better(task_metric_name))]
    objs += [Objective(key, low) for key, low in _EFFICIENCY_OBJECTIVES]
    return objs


def _oriented(value: object, lower_is_better: bool) -> float:
    """Map a (possibly None) metric to a higher-is-better float.

    Missing values are treated as worst-possible so an un-measured variant never
    spuriously dominates a fully-measured one.
    """
    if value is None:
        return float("-inf")
    v = float(value)  # type: ignore[arg-type]
    return -v if lower_is_better else v


def _dominates(a: Sequence[float], b: Sequence[float]) -> bool:
    """True if oriented vector *a* dominates *b* (>= on all, > on at least one)."""
    at_least_as_good = all(x >= y for x, y in zip(a, b))
    strictly_better = any(x > y for x, y in zip(a, b))
    return at_least_as_good and strictly_better


def pareto_ranks(
    rows: Sequence[dict[str, object]],
    objectives: Sequence[Objective],
) -> list[int]:
    """Non-dominated sort: return a 1-based Pareto rank for each row.

    Rank 1 is the non-dominated frontier; rank 2 is the frontier of what remains
    once rank 1 is removed; and so on. Order of the returned list matches *rows*.
    """
    vectors = [tuple(_oriented(r.get(o.key), o.lower_is_better) for o in objectives) for r in rows]
    n = len(rows)
    ranks = [0] * n
    remaining = set(range(n))
    current_rank = 1
    while remaining:
        frontier = [
            i
            for i in remaining
            if not any(j != i and _dominates(vectors[j], vectors[i]) for j in remaining)
        ]
        if not frontier:  # pragma: no cover - defensive; a frontier always exists
            frontier = list(remaining)
        for i in frontier:
            ranks[i] = current_rank
            remaining.discard(i)
        current_rank += 1
    return ranks


def weighted_scores(
    rows: Sequence[dict[str, object]],
    objectives: Sequence[Objective],
    weights: dict[str, float],
) -> list[float]:
    """Min-max normalize each objective to [0, 1] (higher better) and weight-sum.

    ``weights`` is keyed by row column (e.g. ``accuracy``/``task_metric_value``,
    ``param_count``, ``model_size_mb``, ``latency_ms_p50``, ``throughput_sps``).
    The project's target-metric weight may be keyed either by the metric name or
    by ``task_metric_value``. Objectives without a weight contribute nothing.
    """
    if not rows:
        return []
    metric_name = str(rows[0].get("task_metric_name", "")).lower()

    def weight_for(key: str) -> float:
        if key == "task_metric_value":
            return float(weights.get("task_metric_value", weights.get(metric_name, 0.0)))
        return float(weights.get(key, 0.0))

    scores = [0.0] * len(rows)
    for obj in objectives:
        w = weight_for(obj.key)
        if w == 0.0:
            continue
        oriented = [_oriented(r.get(obj.key), obj.lower_is_better) for r in rows]
        finite = [v for v in oriented if v != float("-inf")]
        if not finite:
            continue
        lo, hi = min(finite), max(finite)
        span = hi - lo
        for i, v in enumerate(oriented):
            norm = 0.0 if v == float("-inf") else (1.0 if span == 0 else (v - lo) / span)
            scores[i] += w * norm
    return scores


def ranked_leaderboard(
    project_root: Path,
    *,
    n: int = 25,
    include_pruned: bool = False,
) -> list[dict[str, object]]:
    """Project-aware leaderboard: rows annotated and ordered per ``rank_objective``.

    Every row gains a ``pareto_rank`` (informational regardless of objective) and,
    for the weighted objective, a ``weighted_score``. Ordering:
      - accuracy  → store order (target metric alone)
      - weighted  → descending weighted score
      - pareto    → ascending Pareto rank, ties broken by the target metric
    """
    spec = load_project(project_root)
    # Pull a generous slice so ranking sees the whole field, then trim to n.
    rows = get_leaderboard(
        project_root / "benchmark",
        task_metric_name=spec.target_metric,
        n=max(n, 1000),
        include_pruned=include_pruned,
    )
    if not rows:
        return []

    objectives = default_objectives(spec.target_metric)
    ranks = pareto_ranks(rows, objectives)
    for row, pr in zip(rows, ranks):
        row["pareto_rank"] = pr

    if spec.rank_objective == RankObjective.weighted:
        weights = spec.metric_weights or {spec.target_metric: 1.0}
        scores = weighted_scores(rows, objectives, weights)
        for row, sc in zip(rows, scores):
            row["weighted_score"] = sc
        rows.sort(key=lambda r: r["weighted_score"], reverse=True)  # type: ignore[arg-type,return-value]
    elif spec.rank_objective == RankObjective.pareto:
        lower = is_lower_better(spec.target_metric)
        rows.sort(
            key=lambda r: (
                r["pareto_rank"],
                _oriented(r.get("task_metric_value"), lower) * -1,
            )
        )
    # accuracy (single-objective): keep the store's metric ordering as-is.
    return rows[:n]
