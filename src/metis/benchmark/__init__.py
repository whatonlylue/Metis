"""Harness-side benchmark engine.

Public API:
    BenchmarkRunner    — evaluate a model variant on the sealed holdout.
    seal_holdout       — split processed data and lock the test set away.
    BenchmarkRecord    — dataclass for a single benchmark run.
    append_result      — write a record to results.db.
    get_leaderboard    — read ranked results from results.db (single-objective).
    ranked_leaderboard — project-aware ranking (Pareto / weighted / single).
    prune_project      — mark the weakest variants pruned (reversible).
    is_plateaued       — detect a stalled leaderboard (drives BRANCH).
    compute_budget_status — cumulative resource usage vs. declared budgets.
"""

from metis.benchmark.budget import BudgetStatus, compute_budget_status, record_training_usage
from metis.benchmark.plateau import detect_plateau, is_plateaued, objective_history
from metis.benchmark.prune import prune_project, select_for_pruning
from metis.benchmark.ranking import (
    Objective,
    pareto_ranks,
    ranked_leaderboard,
    weighted_scores,
)
from metis.benchmark.robustness import (
    Corruption,
    apply_corruption,
    corruptions_from_config,
    default_corruptions,
)
from metis.benchmark.runner import BenchmarkRunner
from metis.benchmark.sealer import detect_seal_mode, ensure_holdout_sealed, seal_holdout
from metis.benchmark.store import (
    BenchmarkRecord,
    RobustnessRecord,
    append_result,
    append_robustness_result,
    get_failed_variants,
    get_latest_robustness,
    get_leaderboard,
    mark_pruned,
)

__all__ = [
    "BenchmarkRunner",
    "seal_holdout",
    "ensure_holdout_sealed",
    "detect_seal_mode",
    "BenchmarkRecord",
    "RobustnessRecord",
    "append_result",
    "append_robustness_result",
    "get_failed_variants",
    "get_latest_robustness",
    "get_leaderboard",
    "mark_pruned",
    "Corruption",
    "apply_corruption",
    "corruptions_from_config",
    "default_corruptions",
    "ranked_leaderboard",
    "pareto_ranks",
    "weighted_scores",
    "Objective",
    "prune_project",
    "select_for_pruning",
    "detect_plateau",
    "is_plateaued",
    "objective_history",
    "BudgetStatus",
    "compute_budget_status",
    "record_training_usage",
]
