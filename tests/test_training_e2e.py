"""End-to-end M3 test: PROPOSE -> TRAIN -> BENCHMARK on the toy digits dataset."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("numpy")
pytest.importorskip("sklearn")

from metis.benchmark import get_leaderboard  # noqa: E402
from metis.projects import create_project  # noqa: E402
from metis.projects.schema import ProjectSpec, TaskType  # noqa: E402
from metis.training import (  # noqa: E402
    prepare_toy_dataset,
    propose_candidates,
    run_toy_pipeline,
)


@pytest.fixture
def project_root(tmp_path: Path) -> Path:
    spec = ProjectSpec(
        name="digits",
        description="classify 8x8 handwritten digits",
        task_type=TaskType.image_classification,
        classes=[str(d) for d in range(10)],
        target_metric="accuracy",
    )
    return create_project(tmp_path / "digits", spec)


def test_proposes_multiple_families() -> None:
    candidates = propose_candidates()
    assert len(candidates) >= 2
    families = {c.family for c in candidates}
    assert len(families) == len(candidates)  # all distinct families


def test_prepare_toy_dataset(project_root: Path) -> None:
    import numpy as np

    processed = prepare_toy_dataset(project_root)
    X = np.load(processed / "X.npy")
    y = np.load(processed / "y.npy")
    assert X.shape[0] == y.shape[0]
    assert X.shape[1] == 64  # 8x8 flattened
    assert set(np.unique(y)) == set(range(10))


def test_end_to_end_pipeline_and_leaderboard(project_root: Path) -> None:
    records = run_toy_pipeline(project_root, timeout_s=180)

    # >= 2 variants trained and benchmarked successfully.
    assert len(records) >= 2
    for r in records:
        assert r.error is None, r.error
        assert r.task_metric_value is not None and r.task_metric_value > 0.5
        # Efficiency metrics are all populated for real now.
        assert r.param_count is not None and r.param_count > 0
        assert r.model_size_mb is not None and r.model_size_mb > 0
        assert r.latency_ms_p50 is not None and r.latency_ms_p50 >= 0
        assert r.latency_ms_p95 is not None and r.latency_ms_p95 >= r.latency_ms_p50
        assert r.throughput_sps is not None and r.throughput_sps > 0

    # Leaderboard ranks by accuracy (DESC) with all columns populated.
    rows = get_leaderboard(project_root / "benchmark", task_metric_name="accuracy", n=10)
    assert len(rows) == len(records)
    scores = [r["task_metric_value"] for r in rows]
    assert scores == sorted(scores, reverse=True)
    for row in rows:
        assert row["param_count"] is not None
        assert row["model_size_mb"] is not None
        assert row["latency_ms_p50"] is not None
        assert row["latency_ms_p95"] is not None
        assert row["throughput_sps"] is not None


def test_training_cannot_touch_holdout(project_root: Path) -> None:
    # The sealed holdout exists after the pipeline; a re-run of any train.py is
    # still confined and never reads benchmark/. Here we just assert the holdout
    # was sealed away from the training data the scripts can see.
    prepare_toy_dataset(project_root)
    from metis.benchmark import seal_holdout

    seal_holdout(project_root, fraction=0.2, mode="numpy")
    assert (project_root / "benchmark" / "holdout" / "X.npy").exists()
    # Training data dir holds only the train split; holdout is not under it.
    assert not (project_root / "data" / "processed" / "holdout").exists()
