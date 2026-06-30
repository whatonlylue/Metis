"""Tests for M2 benchmark engine: store, metrics, runner, sealer, and agent tools."""

from __future__ import annotations

import shutil
import textwrap
from pathlib import Path

import pytest
import yaml

from metis.benchmark.metrics import collect_efficiency_metrics, measure_model_size, read_param_count
from metis.benchmark.store import (
    BenchmarkRecord,
    append_result,
    get_failed_variants,
    get_leaderboard,
)
from metis.benchmark.runner import BenchmarkRunner
from metis.benchmark.sealer import seal_holdout
from metis.projects import create_project
from metis.projects.schema import ProjectSpec, TaskType


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def project_root(tmp_path: Path) -> Path:
    spec = ProjectSpec(
        name="test",
        description="test project",
        task_type=TaskType.image_classification,
        classes=["cat", "dog"],
        target_metric="accuracy",
    )
    return create_project(tmp_path / "test", spec)


@pytest.fixture
def benchmark_dir(project_root: Path) -> Path:
    d = project_root / "benchmark"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# store.py
# ---------------------------------------------------------------------------


def test_append_and_retrieve(benchmark_dir: Path) -> None:
    record = BenchmarkRecord(
        variant_id="v1",
        task_metric_name="accuracy",
        task_metric_value=0.92,
        param_count=50_000,
        model_size_mb=0.5,
        latency_ms_p50=None,
        latency_ms_p95=None,
        throughput_sps=None,
    )
    rowid = append_result(benchmark_dir, record)
    assert rowid >= 1

    rows = get_leaderboard(benchmark_dir, task_metric_name="accuracy", n=5)
    assert len(rows) == 1
    assert rows[0]["variant_id"] == "v1"
    assert abs(rows[0]["task_metric_value"] - 0.92) < 1e-9


def test_append_is_cumulative(benchmark_dir: Path) -> None:
    for i, score in enumerate([0.80, 0.95, 0.70]):
        append_result(
            benchmark_dir,
            BenchmarkRecord(
                variant_id=f"v{i}",
                task_metric_name="accuracy",
                task_metric_value=score,
                param_count=None,
                model_size_mb=None,
                latency_ms_p50=None,
                latency_ms_p95=None,
                throughput_sps=None,
            ),
        )
    rows = get_leaderboard(benchmark_dir, task_metric_name="accuracy", n=10)
    assert len(rows) == 3
    # Higher accuracy should be ranked first.
    assert rows[0]["task_metric_value"] == 0.95


def test_lower_is_better_ranking(benchmark_dir: Path) -> None:
    for variant, rmse in [("a", 0.10), ("b", 0.05), ("c", 0.20)]:
        append_result(
            benchmark_dir,
            BenchmarkRecord(
                variant_id=variant,
                task_metric_name="rmse",
                task_metric_value=rmse,
                param_count=None,
                model_size_mb=None,
                latency_ms_p50=None,
                latency_ms_p95=None,
                throughput_sps=None,
            ),
        )
    rows = get_leaderboard(benchmark_dir, task_metric_name="rmse", n=10)
    assert rows[0]["task_metric_value"] == 0.05  # lowest RMSE wins


def test_null_metric_excluded_from_leaderboard(benchmark_dir: Path) -> None:
    append_result(
        benchmark_dir,
        BenchmarkRecord(
            variant_id="errored",
            task_metric_name="accuracy",
            task_metric_value=None,
            param_count=None,
            model_size_mb=None,
            latency_ms_p50=None,
            latency_ms_p95=None,
            throughput_sps=None,
            error="model missing",
        ),
    )
    rows = get_leaderboard(benchmark_dir, task_metric_name="accuracy")
    assert rows == []


def test_failed_variants_surfaced_separately(benchmark_dir: Path) -> None:
    # One good, one errored variant.
    append_result(
        benchmark_dir,
        BenchmarkRecord(
            variant_id="good",
            task_metric_name="accuracy",
            task_metric_value=0.9,
            param_count=10,
            model_size_mb=0.1,
            latency_ms_p50=1.0,
            latency_ms_p95=2.0,
            throughput_sps=100.0,
        ),
    )
    append_result(
        benchmark_dir,
        BenchmarkRecord(
            variant_id="crashed",
            task_metric_name="accuracy",
            task_metric_value=None,
            param_count=None,
            model_size_mb=None,
            latency_ms_p50=None,
            latency_ms_p95=None,
            throughput_sps=None,
            error="No module named 'torch'",
        ),
    )

    board = get_leaderboard(benchmark_dir, task_metric_name="accuracy")
    assert [r["variant_id"] for r in board] == ["good"]

    failed = get_failed_variants(benchmark_dir)
    assert [r["variant_id"] for r in failed] == ["crashed"]
    assert failed[0]["error"] == "No module named 'torch'"


def test_leaderboard_does_not_mix_metric_names(benchmark_dir: Path) -> None:
    # Heterogeneous metric rows must not be ranked together.
    append_result(
        benchmark_dir,
        BenchmarkRecord(
            variant_id="acc_model",
            task_metric_name="accuracy",
            task_metric_value=0.90,
            param_count=None,
            model_size_mb=None,
            latency_ms_p50=None,
            latency_ms_p95=None,
            throughput_sps=None,
        ),
    )
    append_result(
        benchmark_dir,
        BenchmarkRecord(
            variant_id="rmse_model",
            task_metric_name="rmse",
            task_metric_value=0.01,
            param_count=None,
            model_size_mb=None,
            latency_ms_p50=None,
            latency_ms_p95=None,
            throughput_sps=None,
        ),
    )

    acc_rows = get_leaderboard(benchmark_dir, task_metric_name="accuracy", n=10)
    assert [r["variant_id"] for r in acc_rows] == ["acc_model"]
    assert all(r["task_metric_name"] == "accuracy" for r in acc_rows)

    rmse_rows = get_leaderboard(benchmark_dir, task_metric_name="rmse", n=10)
    assert [r["variant_id"] for r in rmse_rows] == ["rmse_model"]
    assert all(r["task_metric_name"] == "rmse" for r in rmse_rows)


def test_timestamp_auto_filled(benchmark_dir: Path) -> None:
    record = BenchmarkRecord(
        variant_id="v",
        task_metric_name="accuracy",
        task_metric_value=0.5,
        param_count=None,
        model_size_mb=None,
        latency_ms_p50=None,
        latency_ms_p95=None,
        throughput_sps=None,
    )
    assert record.timestamp  # auto-filled in __post_init__


# ---------------------------------------------------------------------------
# metrics.py
# ---------------------------------------------------------------------------


def test_measure_model_size(tmp_path: Path) -> None:
    weights = tmp_path / "weights"
    weights.mkdir()
    (weights / "model.bin").write_bytes(b"x" * 1024 * 1024)  # exactly 1 MB
    size = measure_model_size(weights)
    assert size is not None
    assert abs(size - 1.0) < 1e-4


def test_measure_model_size_missing_dir(tmp_path: Path) -> None:
    assert measure_model_size(tmp_path / "nonexistent") is None


def test_read_param_count(tmp_path: Path) -> None:
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(yaml.safe_dump({"param_count": 12345, "architecture": "cnn"}))
    assert read_param_count(recipe) == 12345


def test_read_param_count_missing(tmp_path: Path) -> None:
    assert read_param_count(tmp_path / "no_recipe.yaml") is None


def test_read_param_count_no_field(tmp_path: Path) -> None:
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(yaml.safe_dump({"architecture": "cnn"}))
    assert read_param_count(recipe) is None


def test_collect_efficiency_metrics(tmp_path: Path) -> None:
    variant = tmp_path / "v1"
    (variant / "weights").mkdir(parents=True)
    (variant / "weights" / "model.pkl").write_bytes(b"x" * 2048)
    (variant / "recipe.yaml").write_text(yaml.safe_dump({"param_count": 999}))

    m = collect_efficiency_metrics(variant)
    assert m.param_count == 999
    assert m.model_size_mb is not None and m.model_size_mb > 0
    assert m.latency_ms_p50 is None  # not yet measured
    assert m.throughput_sps is None


# ---------------------------------------------------------------------------
# runner.py
# ---------------------------------------------------------------------------


def test_runner_missing_variant(project_root: Path) -> None:
    runner = BenchmarkRunner()
    record = runner.run(project_root, "nonexistent")
    assert record.task_metric_value is None
    assert record.error is not None
    assert "not found" in record.error


def test_runner_records_in_db(project_root: Path) -> None:
    runner = BenchmarkRunner()
    runner.run(project_root, "nonexistent")
    # The errored record has null metric, so leaderboard is empty — but DB should have a row.
    from metis.benchmark.store import _connect

    conn = _connect(project_root / "benchmark")
    count = conn.execute("SELECT COUNT(*) FROM benchmark_runs").fetchone()[0]
    conn.close()
    assert count == 1


def test_runner_no_model_py(project_root: Path) -> None:
    variant_dir = project_root / "models" / "v1"
    variant_dir.mkdir(parents=True)
    (variant_dir / "weights").mkdir()
    (variant_dir / "recipe.yaml").write_text(yaml.safe_dump({"param_count": 100}))

    runner = BenchmarkRunner()
    record = runner.run(project_root, "v1")

    # Efficiency metrics are populated even without task metric.
    assert record.param_count == 100
    assert record.task_metric_value is None
    assert record.error is not None
    assert "model.py" in record.error


def _make_trivial_model_py(variant_dir: Path) -> None:
    """Write a model.py that always predicts class 0."""
    (variant_dir / "model.py").write_text(
        textwrap.dedent("""\
            import numpy as np
            def load_model(weights_dir):
                return None
            def predict(model, X):
                return np.zeros(len(X), dtype=int)
        """)
    )


def _make_holdout(project_root: Path, n: int = 20) -> None:
    """Write trivial X.npy / y.npy holdout (all labels = 0 so accuracy = 1.0)."""
    try:
        import numpy as np
    except ImportError:
        pytest.skip("numpy not installed")

    holdout_dir = project_root / "benchmark" / "holdout"
    holdout_dir.mkdir(parents=True, exist_ok=True)
    np.save(holdout_dir / "X.npy", np.zeros((n, 4)))
    np.save(holdout_dir / "y.npy", np.zeros(n, dtype=int))


def test_runner_end_to_end(project_root: Path) -> None:
    pytest.importorskip("numpy")

    variant_dir = project_root / "models" / "v1"
    (variant_dir / "weights").mkdir(parents=True)
    _make_trivial_model_py(variant_dir)
    _make_holdout(project_root)

    runner = BenchmarkRunner()
    record = runner.run(project_root, "v1")

    assert record.error is None
    assert record.task_metric_value is not None
    assert abs(record.task_metric_value - 1.0) < 1e-9  # all-zeros model on all-zeros labels


def test_malicious_model_cannot_read_holdout_to_cheat(project_root: Path) -> None:
    # B2 regression test: a model.py that ignores its inputs and tries to read
    # benchmark/holdout/y.npy off disk (computing the project root from __file__)
    # must NOT be able to return the true labels and score a perfect 1.0. The OS
    # sandbox makes benchmark/ unreachable; the harness also never hands labels
    # to agent code. This runs through the real production scoring path.
    np = pytest.importorskip("numpy")

    variant_dir = project_root / "models" / "evil"
    (variant_dir / "weights").mkdir(parents=True)
    (variant_dir / "model.py").write_text(
        textwrap.dedent("""\
            import numpy as np
            from pathlib import Path

            def load_model(weights_dir):
                return None

            def predict(model, X):
                # Attempt to cheat: read the sealed labels straight off disk.
                root = Path(__file__).resolve().parents[2]
                y = np.load(root / "benchmark" / "holdout" / "y.npy")
                return np.asarray(y)[: len(X)]
        """)
    )

    # Holdout with NON-uniform labels, so the only way to score 1.0 is to read y.
    holdout_dir = project_root / "benchmark" / "holdout"
    holdout_dir.mkdir(parents=True, exist_ok=True)
    n = 20
    np.save(holdout_dir / "X.npy", np.zeros((n, 4)))
    np.save(holdout_dir / "y.npy", np.array([i % 3 for i in range(n)], dtype=int))

    runner = BenchmarkRunner()
    record = runner.run(project_root, "evil")

    # The cheat must fail: either the eval errors (sandbox denied the read) or it
    # produced predictions without ever seeing the true labels — never a perfect
    # score by reading the holdout.
    assert record.error is not None or (
        record.task_metric_value is not None and record.task_metric_value < 0.99
    ), f"malicious model cheated: {record}"


def test_param_count_is_measured_not_trusted_from_recipe(project_root: Path) -> None:
    # M1 regression test: param_count must be measured from the serialized model,
    # not taken verbatim from the agent's recipe.yaml (which here lies "1").
    np = pytest.importorskip("numpy")
    pytest.importorskip("sklearn")
    import pickle

    from sklearn.linear_model import LogisticRegression

    variant_dir = project_root / "models" / "honest_size"
    weights = variant_dir / "weights"
    weights.mkdir(parents=True)

    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 4))
    y = (X[:, 0] > 0).astype(int)
    clf = LogisticRegression(max_iter=500).fit(X, y)
    with open(weights / "model.pkl", "wb") as f:
        pickle.dump(clf, f)
    # Real param count for a binary logreg on 4 features: coef_(1,4) + intercept_(1) = 5.
    real_params = clf.coef_.size + clf.intercept_.size
    assert real_params == 5

    (variant_dir / "model.py").write_text(
        textwrap.dedent("""\
            import pickle
            from pathlib import Path

            def load_model(weights_dir):
                with open(Path(weights_dir) / "model.pkl", "rb") as f:
                    return pickle.load(f)

            def predict(model, X):
                return model.predict(X)
        """)
    )
    # The agent under-reports param_count to look more efficient than it is.
    (variant_dir / "recipe.yaml").write_text(
        yaml.safe_dump({"architecture": "logreg", "param_count": 1})
    )

    holdout_dir = project_root / "benchmark" / "holdout"
    holdout_dir.mkdir(parents=True, exist_ok=True)
    np.save(holdout_dir / "X.npy", X[:10])
    np.save(holdout_dir / "y.npy", y[:10])

    runner = BenchmarkRunner()
    record = runner.run(project_root, "honest_size")

    assert record.error is None, record.error
    assert record.param_count == real_params  # measured, not the bogus "1"
    assert record.param_count != 1


# ---------------------------------------------------------------------------
# sealer.py
# ---------------------------------------------------------------------------


def test_seal_imagenet(project_root: Path) -> None:
    processed = project_root / "data" / "processed"
    for cls in ["cat", "dog"]:
        d = processed / cls
        d.mkdir(parents=True, exist_ok=True)
        for i in range(10):
            (d / f"img{i:02d}.jpg").write_bytes(b"fake")

    holdout_dir = seal_holdout(project_root, fraction=0.2, mode="imagenet")

    assert (holdout_dir / "seal.yaml").exists()
    meta = yaml.safe_load((holdout_dir / "seal.yaml").read_text())
    assert meta["fraction"] == 0.2
    assert meta["mode"] == "imagenet"

    # Should have seeded some files from each class.
    for cls in ["cat", "dog"]:
        assert list((holdout_dir / cls).iterdir()), f"no holdout files for {cls}"


def test_seal_imagenet_removes_holdout_from_source(project_root: Path) -> None:
    processed = project_root / "data" / "processed"
    original: dict[str, set[str]] = {}
    for cls in ["cat", "dog"]:
        d = processed / cls
        d.mkdir(parents=True, exist_ok=True)
        names = set()
        for i in range(10):
            (d / f"img{i:02d}.jpg").write_bytes(b"fake")
            names.add(f"img{i:02d}.jpg")
        original[cls] = names

    holdout_dir = seal_holdout(project_root, fraction=0.2, mode="imagenet")

    for cls in ["cat", "dog"]:
        holdout_names = {f.name for f in (holdout_dir / cls).iterdir()}
        source_names = {f.name for f in (processed / cls).iterdir()}

        assert holdout_names, f"no holdout files for {cls}"
        # Holdout files must be REMOVED from the source/training dir.
        assert holdout_names.isdisjoint(source_names), (
            f"{cls}: holdout files still present in source (train/test contamination)"
        )
        # No data lost: source + holdout reconstruct the original set.
        assert holdout_names | source_names == original[cls]


def test_seal_imagenet_rejects_double_seal(project_root: Path) -> None:
    processed = project_root / "data" / "processed"
    for cls in ["cat", "dog"]:
        d = processed / cls
        d.mkdir(parents=True, exist_ok=True)
        for i in range(10):
            (d / f"img{i:02d}.jpg").write_bytes(b"fake")

    seal_holdout(project_root, fraction=0.2, mode="imagenet")
    with pytest.raises(FileExistsError):
        seal_holdout(project_root, fraction=0.2, mode="imagenet")


def _populate_imagenet_source(processed: Path, n: int = 20) -> None:
    d = processed / "a"
    d.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        (d / f"img{i:02d}.jpg").write_bytes(b"fake")


def test_seal_imagenet_deterministic(project_root: Path) -> None:
    processed = project_root / "data" / "processed"
    _populate_imagenet_source(processed)

    holdout_dir1 = seal_holdout(project_root, fraction=0.2, seed=1, mode="imagenet")
    files1 = sorted(f.name for f in (holdout_dir1 / "a").iterdir())

    # Holdout samples are removed from source, so reset the source (and remove
    # the seal manifest) before re-sealing to compare the same starting state.
    shutil.rmtree(holdout_dir1)
    shutil.rmtree(processed)
    _populate_imagenet_source(processed)

    holdout_dir2 = seal_holdout(project_root, fraction=0.2, seed=1, mode="imagenet")
    files2 = sorted(f.name for f in (holdout_dir2 / "a").iterdir())

    assert files1 == files2


def test_seal_numpy(tmp_path: Path) -> None:
    numpy = pytest.importorskip("numpy")

    spec = ProjectSpec(
        name="np_test",
        description="test",
        task_type=TaskType.tabular_classification,
        target_metric="accuracy",
    )
    root = create_project(tmp_path / "np_test", spec)
    processed = root / "data" / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    numpy.save(processed / "X.npy", numpy.arange(100).reshape(100, 1))
    numpy.save(processed / "y.npy", numpy.zeros(100, dtype=int))

    holdout_dir = seal_holdout(root, fraction=0.2, mode="numpy")

    X_hold = numpy.load(holdout_dir / "X.npy")
    X_train = numpy.load(processed / "X.npy")
    assert len(X_hold) + len(X_train) == 100
    assert len(X_hold) >= 1


def test_seal_invalid_fraction(project_root: Path) -> None:
    with pytest.raises(ValueError, match="fraction"):
        seal_holdout(project_root, fraction=1.5)


# ---------------------------------------------------------------------------
# agent tools (build_benchmark_tools)
# ---------------------------------------------------------------------------


def test_get_leaderboard_tool_empty(project_root: Path) -> None:
    from metis.agent.tools import build_benchmark_tools

    tools = {t.name: t for t in build_benchmark_tools(project_root)}
    result = tools["get_leaderboard"].handler({})
    assert "No benchmark results" in result


def test_submit_tool_missing_variant(project_root: Path) -> None:
    from metis.agent.tools import build_benchmark_tools

    tools = {t.name: t for t in build_benchmark_tools(project_root)}
    result = tools["submit_for_benchmark"].handler({"variant_id": "ghost"})
    assert "error" in result.lower() or "not found" in result.lower()


def test_leaderboard_tool_after_submit(project_root: Path) -> None:
    pytest.importorskip("numpy")

    variant_dir = project_root / "models" / "v42"
    (variant_dir / "weights").mkdir(parents=True)
    _make_trivial_model_py(variant_dir)
    _make_holdout(project_root)

    from metis.agent.tools import build_benchmark_tools

    tools = {t.name: t for t in build_benchmark_tools(project_root)}
    submit_result = tools["submit_for_benchmark"].handler({"variant_id": "v42"})
    assert "accuracy" in submit_result

    lb = tools["get_leaderboard"].handler({"n": 5})
    assert "v42" in lb
    assert "Rank" in lb
