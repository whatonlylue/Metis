"""Tests for M4 evolutionary search: prune, plateau, Pareto/weighted, budgets."""

from __future__ import annotations

from pathlib import Path

import pytest

from metis.benchmark.budget import compute_budget_status, record_training_usage
from metis.benchmark.plateau import detect_plateau, is_plateaued, objective_history
from metis.benchmark.prune import prune_project, select_for_pruning
from metis.benchmark.ranking import Objective, default_objectives, pareto_ranks, weighted_scores
from metis.benchmark.store import (
    BenchmarkRecord,
    append_result,
    get_leaderboard,
    get_usage_totals,
    mark_pruned,
)
from metis.projects import create_project
from metis.projects.schema import (
    Budgets,
    PlateauPolicy,
    PrunePolicy,
    ProjectSpec,
    RankObjective,
    TaskType,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_project(tmp_path: Path, **overrides: object) -> Path:
    spec = ProjectSpec(
        name="evo",
        description="evolutionary search test",
        task_type=TaskType.tabular_classification,
        classes=["a", "b"],
        target_metric=str(overrides.pop("target_metric", "accuracy")),
        **overrides,  # type: ignore[arg-type]
    )
    return create_project(tmp_path / "evo", spec)


def _row(variant_id: str, **kw: object) -> dict[str, object]:
    base: dict[str, object] = {
        "variant_id": variant_id,
        "task_metric_name": "accuracy",
        "task_metric_value": None,
        "param_count": None,
        "model_size_mb": None,
        "latency_ms_p50": None,
        "latency_ms_p95": None,
        "throughput_sps": None,
    }
    base.update(kw)
    return base


def _add(benchmark_dir: Path, variant_id: str, value: float, **kw: object) -> None:
    append_result(
        benchmark_dir,
        BenchmarkRecord(
            variant_id=variant_id,
            task_metric_name=str(kw.pop("metric", "accuracy")),
            task_metric_value=value,
            param_count=kw.pop("param_count", None),  # type: ignore[arg-type]
            model_size_mb=kw.pop("model_size_mb", None),  # type: ignore[arg-type]
            latency_ms_p50=kw.pop("latency_ms_p50", None),  # type: ignore[arg-type]
            latency_ms_p95=None,
            throughput_sps=None,
        ),
    )


# ---------------------------------------------------------------------------
# Schema backward compatibility
# ---------------------------------------------------------------------------


def test_schema_defaults_and_backward_compat() -> None:
    # A project.yaml written before M4 (no prune/plateau/cost fields) still loads.
    spec = ProjectSpec.model_validate(
        {
            "name": "old",
            "description": "legacy project",
            "task_type": "tabular_classification",
            "target_metric": "accuracy",
        }
    )
    assert spec.prune_policy.keep_top_k is None
    assert spec.prune_policy.drop_bottom_fraction is None
    assert spec.plateau.window == 3
    assert spec.plateau.epsilon == pytest.approx(1e-3)
    assert spec.budgets.dollars_per_minute == 0.0


# ---------------------------------------------------------------------------
# Prune selection
# ---------------------------------------------------------------------------


def test_select_keep_top_k() -> None:
    ranked = ["a", "b", "c", "d", "e"]
    assert select_for_pruning(ranked, keep_top_k=2) == ["c", "d", "e"]


def test_select_drop_bottom_fraction() -> None:
    ranked = ["a", "b", "c", "d"]
    # floor(4 * 0.5) = 2 worst pruned.
    assert select_for_pruning(ranked, drop_bottom_fraction=0.5) == ["c", "d"]


def test_select_fraction_keeps_at_least_top() -> None:
    ranked = ["a", "b"]
    # fraction 1.0 would drop everything; the top variant is always kept.
    assert select_for_pruning(ranked, drop_bottom_fraction=1.0) == ["b"]


def test_select_no_policy_is_noop() -> None:
    assert select_for_pruning(["a", "b", "c"]) == []


def test_select_keep_top_k_precedence() -> None:
    ranked = ["a", "b", "c", "d"]
    assert select_for_pruning(ranked, keep_top_k=1, drop_bottom_fraction=0.99) == ["b", "c", "d"]


def test_prune_project_marks_reversibly(tmp_path: Path) -> None:
    root = _make_project(tmp_path, prune_policy=PrunePolicy(keep_top_k=1))
    bench = root / "benchmark"
    _add(bench, "good", 0.95)
    _add(bench, "mid", 0.80)
    _add(bench, "bad", 0.50)

    pruned = prune_project(root)
    assert set(pruned) == {"mid", "bad"}

    # Default leaderboard now excludes pruned variants.
    rows = get_leaderboard(bench, task_metric_name="accuracy")
    assert [r["variant_id"] for r in rows] == ["good"]

    # ...but their records are preserved and visible with include_pruned.
    all_rows = get_leaderboard(bench, task_metric_name="accuracy", include_pruned=True)
    assert {r["variant_id"] for r in all_rows} == {"good", "mid", "bad"}
    assert any(r["variant_id"] == "mid" and r["pruned"] for r in all_rows)

    # Pruning is reversible.
    mark_pruned(bench, ["mid"], pruned=False)
    rows2 = get_leaderboard(bench, task_metric_name="accuracy")
    assert {r["variant_id"] for r in rows2} == {"good", "mid"}


# ---------------------------------------------------------------------------
# Plateau detection
# ---------------------------------------------------------------------------


def test_detect_plateau_improving() -> None:
    assert not detect_plateau([0.5, 0.6, 0.7, 0.8], epsilon=0.01, window=2)


def test_detect_plateau_stalled() -> None:
    assert detect_plateau([0.80, 0.81, 0.811, 0.8111], epsilon=0.01, window=2)


def test_detect_plateau_insufficient_history() -> None:
    assert not detect_plateau([0.5, 0.9], epsilon=0.01, window=3)


def test_detect_plateau_lower_is_better() -> None:
    # RMSE dropping steadily = still improving.
    assert not detect_plateau([0.5, 0.4, 0.3, 0.2], epsilon=0.01, window=2, lower_is_better=True)
    # RMSE flat = plateau.
    assert detect_plateau([0.30, 0.30, 0.301, 0.299], epsilon=0.01, window=2, lower_is_better=True)


def test_detect_plateau_late_breakthrough() -> None:
    # A stall followed by a jump in the last window is NOT a plateau.
    assert not detect_plateau([0.5, 0.5, 0.5, 0.9], epsilon=0.01, window=2)


def test_is_plateaued_end_to_end(tmp_path: Path) -> None:
    root = _make_project(tmp_path, plateau=PlateauPolicy(epsilon=0.01, window=2))
    bench = root / "benchmark"
    for v in [0.80, 0.805, 0.806, 0.8061]:
        _add(bench, f"v{v}", v)
    assert objective_history(root) == [0.80, 0.805, 0.806, 0.8061]
    assert is_plateaued(root)


# ---------------------------------------------------------------------------
# Pareto frontier + weighted sum
# ---------------------------------------------------------------------------


def test_pareto_known_dominated_set() -> None:
    objectives = [Objective("task_metric_value", False), Objective("param_count", True)]
    rows = [
        _row("A", task_metric_value=0.90, param_count=100),  # frontier
        _row("B", task_metric_value=0.80, param_count=10),  # frontier (tiny)
        _row("C", task_metric_value=0.85, param_count=50),  # frontier (middle)
        _row("D", task_metric_value=0.70, param_count=200),  # dominated by all
    ]
    ranks = pareto_ranks(rows, objectives)
    by_id = {r["variant_id"]: rank for r, rank in zip(rows, ranks)}
    assert by_id["A"] == 1
    assert by_id["B"] == 1
    assert by_id["C"] == 1
    assert by_id["D"] == 2  # strictly dominated -> later front


def test_pareto_clear_domination() -> None:
    objectives = [Objective("task_metric_value", False), Objective("param_count", True)]
    rows = [
        _row("best", task_metric_value=0.99, param_count=10),
        _row("worse", task_metric_value=0.50, param_count=100),
    ]
    ranks = pareto_ranks(rows, objectives)
    assert ranks[0] == 1
    assert ranks[1] == 2


def test_weighted_sum_prefers_efficiency_when_weighted() -> None:
    objectives = default_objectives("accuracy")
    rows = [
        _row(
            "accurate",
            task_metric_value=0.95,
            param_count=1_000_000,
            model_size_mb=50.0,
            latency_ms_p50=100.0,
        ),
        _row(
            "efficient",
            task_metric_value=0.90,
            param_count=1_000,
            model_size_mb=0.5,
            latency_ms_p50=1.0,
        ),
    ]
    # Heavily weight efficiency: the small/fast model should win despite lower acc.
    weights = {"accuracy": 0.1, "param_count": 0.4, "model_size_mb": 0.3, "latency_ms_p50": 0.2}
    scores = weighted_scores(rows, objectives, weights)
    assert scores[1] > scores[0]

    # Weight accuracy alone: the accurate model wins.
    acc_scores = weighted_scores(rows, objectives, {"accuracy": 1.0})
    assert acc_scores[0] > acc_scores[1]


def test_ranked_leaderboard_pareto_orders_frontier_first(tmp_path: Path) -> None:
    from metis.benchmark.ranking import ranked_leaderboard

    root = _make_project(tmp_path, rank_objective=RankObjective.pareto)
    bench = root / "benchmark"
    _add(bench, "dominated", 0.70, param_count=200, model_size_mb=5.0, latency_ms_p50=10.0)
    _add(bench, "frontier", 0.95, param_count=20, model_size_mb=0.2, latency_ms_p50=1.0)

    rows = ranked_leaderboard(root, n=10)
    assert rows[0]["variant_id"] == "frontier"
    assert rows[0]["pareto_rank"] == 1


def test_ranked_leaderboard_weighted(tmp_path: Path) -> None:
    from metis.benchmark.ranking import ranked_leaderboard

    root = _make_project(
        tmp_path,
        rank_objective=RankObjective.weighted,
        metric_weights={"accuracy": 0.1, "param_count": 0.9},
    )
    bench = root / "benchmark"
    _add(bench, "big", 0.95, param_count=1_000_000, model_size_mb=10.0, latency_ms_p50=5.0)
    _add(bench, "small", 0.90, param_count=100, model_size_mb=0.1, latency_ms_p50=1.0)

    rows = ranked_leaderboard(root, n=10)
    assert rows[0]["variant_id"] == "small"
    assert "weighted_score" in rows[0]


# ---------------------------------------------------------------------------
# Budget tracking + enforcement
# ---------------------------------------------------------------------------


def test_usage_totals_accumulate(tmp_path: Path) -> None:
    root = _make_project(tmp_path)
    record_training_usage(root, variant_id="a", wall_clock_s=30.0)
    record_training_usage(root, variant_id="b", wall_clock_s=90.0)
    totals = get_usage_totals(root / "benchmark")
    assert totals["n_train"] == 2
    assert totals["wall_clock_s"] == pytest.approx(120.0)


def test_budget_not_exhausted_without_limits(tmp_path: Path) -> None:
    root = _make_project(tmp_path)
    record_training_usage(root, variant_id="a", wall_clock_s=600.0)
    status = compute_budget_status(root)
    assert not status.should_stop
    assert status.wall_clock_minutes_remaining is None


def test_budget_wall_clock_exhaustion(tmp_path: Path) -> None:
    root = _make_project(tmp_path, budgets=Budgets(max_wall_clock_minutes=1.0))
    record_training_usage(root, variant_id="a", wall_clock_s=90.0)  # 1.5 min > 1.0
    status = compute_budget_status(root)
    assert status.should_stop
    assert any("wall-clock" in r for r in status.reasons)


def test_budget_variant_exhaustion(tmp_path: Path) -> None:
    root = _make_project(tmp_path, budgets=Budgets(max_variants=2))
    for v in ["a", "b"]:
        record_training_usage(root, variant_id=v, wall_clock_s=10.0)
    status = compute_budget_status(root)
    assert status.should_stop
    assert status.variants_remaining == 0


def test_budget_dollar_cost_model(tmp_path: Path) -> None:
    root = _make_project(tmp_path, budgets=Budgets(max_dollars=1.0, dollars_per_minute=2.0))
    record_training_usage(root, variant_id="a", wall_clock_s=60.0)  # 1 min * $2 = $2 > $1
    status = compute_budget_status(root)
    assert status.dollars_used == pytest.approx(2.0)
    assert status.should_stop


# ---------------------------------------------------------------------------
# Branch candidate generation
# ---------------------------------------------------------------------------


def test_branch_mutates_and_introduces_families() -> None:
    from metis.training.evolve import branch_candidates

    # Only the proposed families exist; branching should mutate a top performer
    # AND introduce at least one untried family (e.g. random_forest / mlp).
    existing = ["logreg", "decision_tree", "knn"]
    cands = branch_candidates(
        existing, existing_ids=existing, max_mutations=2, max_new_families=1, seed=1
    )
    assert len(cands) == 3
    ids = [c.variant_id for c in cands]
    assert all(i not in existing for i in ids)  # no id collisions
    assert len(ids) == len(set(ids))  # unique within the batch

    families = {c.family for c in cands}
    # A new family not among the proposed three appears.
    assert families - {"logistic_regression", "decision_tree", "k_nearest_neighbors"}


def test_branch_mutation_changes_hyperparameters() -> None:
    from metis.training.evolve import branch_candidates

    cands = branch_candidates(
        ["decision_tree"],
        existing_ids=["decision_tree"],
        max_mutations=1,
        max_new_families=0,
        seed=3,
    )
    assert len(cands) == 1
    mutant = cands[0]
    assert mutant.family == "decision_tree"
    # Default depth is 10; the mutant must differ from the default ctor.
    assert "max_depth=10" not in mutant.train_py


def test_branch_is_deterministic() -> None:
    from metis.training.evolve import branch_candidates

    a = branch_candidates(["logreg", "knn"], seed=7)
    b = branch_candidates(["logreg", "knn"], seed=7)
    assert [c.variant_id for c in a] == [c.variant_id for c in b]
    assert [c.train_py for c in a] == [c.train_py for c in b]
