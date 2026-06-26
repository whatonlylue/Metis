"""Plateau detection over the benchmark history (harness-side).

Drives the BRANCH step of the agent loop: when the leaderboard's best objective
stops improving, the harness signals that the search should branch out (mutate
top performers / try new families) instead of continuing to refine. Detection is
purely a read over ``results.db`` ordered by insertion, so each benchmarked
variant is one "round".
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from metis.benchmark.store import _connect, is_lower_better
from metis.projects import load_project


def objective_history(project_root: Path) -> list[float]:
    """Per-round task-metric values in benchmark order (NULL/errored rounds skipped)."""
    spec = load_project(project_root)
    conn = _connect(project_root / "benchmark")
    try:
        rows = conn.execute(
            """SELECT task_metric_value FROM benchmark_runs
               WHERE task_metric_value IS NOT NULL AND task_metric_name = ?
               ORDER BY id ASC""",
            (spec.target_metric,),
        ).fetchall()
    finally:
        conn.close()
    return [float(r["task_metric_value"]) for r in rows]


def detect_plateau(
    round_scores: Sequence[float],
    *,
    epsilon: float,
    window: int,
    lower_is_better: bool = False,
) -> bool:
    """True if the best-so-far objective improved by <= epsilon over the last *window* rounds.

    We compare the running best at the end against the running best ``window``
    rounds earlier. Fewer than ``window + 1`` rounds is never a plateau (not
    enough evidence). ``window`` must be >= 1.
    """
    if window < 1:
        raise ValueError("window must be >= 1")
    if len(round_scores) <= window:
        return False

    oriented = [(-s if lower_is_better else s) for s in round_scores]
    running_best: list[float] = []
    cur = float("-inf")
    for v in oriented:
        cur = max(cur, v)
        running_best.append(cur)

    improvement = running_best[-1] - running_best[-1 - window]
    return improvement <= epsilon


def is_plateaued(project_root: Path) -> bool:
    """Apply the project's plateau policy to its benchmark history."""
    spec = load_project(project_root)
    history = objective_history(project_root)
    return detect_plateau(
        history,
        epsilon=spec.plateau.epsilon,
        window=spec.plateau.window,
        lower_is_better=is_lower_better(spec.target_metric),
    )
