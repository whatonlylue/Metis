"""Resource budget tracking + enforcement (harness-side).

CLAUDE.md guardrail: "Resource budgets (time, compute, $) are enforced by the
harness, not trusted to the agent." Usage is recorded into ``results.db`` (the
sealed, harness-owned store) every time the harness runs training; this module
sums that usage against the budgets declared in ``project.yaml`` and reports
whether the search must STOP.

Tracked dimensions:
  - wall-clock minutes — cumulative training time (measured run durations).
  - variants trained   — a count/compute proxy.
  - dollars            — simple cost model: wall-clock minutes * dollars_per_minute.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from metis.benchmark.store import get_usage_totals, record_usage
from metis.projects import load_project


@dataclass
class BudgetStatus:
    """A point-in-time view of resource usage vs. the project's declared budgets."""

    wall_clock_minutes_used: float
    variants_trained: int
    dollars_used: float
    wall_clock_minutes_remaining: float | None = None
    variants_remaining: int | None = None
    dollars_remaining: float | None = None
    exhausted: bool = False
    reasons: list[str] = field(default_factory=list)

    @property
    def should_stop(self) -> bool:
        """True when any declared budget is exhausted — the search must STOP."""
        return self.exhausted


def record_training_usage(
    project_root: Path,
    *,
    variant_id: str,
    wall_clock_s: float,
    detail: str | None = None,
) -> None:
    """Record one training run's wall-clock cost into the harness-owned store."""
    benchmark_dir = project_root / "benchmark"
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    record_usage(
        benchmark_dir,
        kind="train",
        variant_id=variant_id,
        wall_clock_s=wall_clock_s,
        detail=detail,
    )


def compute_budget_status(project_root: Path) -> BudgetStatus:
    """Sum recorded usage and compare against ``project.yaml`` budgets."""
    spec = load_project(project_root)
    budgets = spec.budgets
    totals = get_usage_totals(project_root / "benchmark")

    minutes_used = totals["wall_clock_s"] / 60.0
    variants_trained = int(totals["n_train"])
    dollars_used = minutes_used * budgets.dollars_per_minute

    reasons: list[str] = []
    minutes_remaining: float | None = None
    variants_remaining: int | None = None
    dollars_remaining: float | None = None

    if budgets.max_wall_clock_minutes is not None:
        minutes_remaining = budgets.max_wall_clock_minutes - minutes_used
        if minutes_remaining <= 0:
            reasons.append(
                f"wall-clock budget exhausted: {minutes_used:.2f} / "
                f"{budgets.max_wall_clock_minutes:.2f} min"
            )
    if budgets.max_variants is not None:
        variants_remaining = budgets.max_variants - variants_trained
        if variants_remaining <= 0:
            reasons.append(f"variant budget exhausted: {variants_trained} / {budgets.max_variants}")
    if budgets.max_dollars is not None:
        dollars_remaining = budgets.max_dollars - dollars_used
        if dollars_remaining <= 0:
            reasons.append(f"$ budget exhausted: ${dollars_used:.2f} / ${budgets.max_dollars:.2f}")

    return BudgetStatus(
        wall_clock_minutes_used=minutes_used,
        variants_trained=variants_trained,
        dollars_used=dollars_used,
        wall_clock_minutes_remaining=minutes_remaining,
        variants_remaining=variants_remaining,
        dollars_remaining=dollars_remaining,
        exhausted=bool(reasons),
        reasons=reasons,
    )
