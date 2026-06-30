"""Tests for M5 data sourcing: providers, provenance/license, dedup/validate/split."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

np = pytest.importorskip("numpy")

from metis.data_sources import (  # noqa: E402
    LicenseError,
    LicensePolicy,
    LocalRegistryProvider,
    SklearnDatasetProvider,
    dedup,
    ingest_arrays,
    ingest_dataset,
    read_manifest,
    validate,
)
from metis.data_sources.ingest import ValidationError  # noqa: E402
from metis.projects import create_project, load_project, record_data_source  # noqa: E402
from metis.projects.schema import (  # noqa: E402
    DataSourceRef,
    ProjectSpec,
    SplitRatios,
    TaskType,
)
from metis.sandbox import LockboxViolation, read_file  # noqa: E402
from metis.sandbox.lockbox import resolve_within_project  # noqa: E402


def _make_project(tmp_path: Path, **kw: object) -> Path:
    name = str(kw.pop("name", "ds"))
    spec = ProjectSpec(
        name=name,
        description="data sources test",
        task_type=TaskType.tabular_classification,
        target_metric="accuracy",
        **kw,  # type: ignore[arg-type]
    )
    return create_project(tmp_path / name, spec)


# ---------------------------------------------------------------------------
# Providers: search + fetch (offline)
# ---------------------------------------------------------------------------


def test_sklearn_provider_search() -> None:
    prov = SklearnDatasetProvider()
    hits = prov.search("iris")
    assert any(h.dataset_id == "iris" for h in hits)
    # empty query lists the catalogue
    assert len(prov.search("")) >= 3
    assert all(h.license for h in prov.search(""))  # every catalogue entry has a license


def test_sklearn_provider_fetch_writes_data_and_manifest(tmp_path: Path) -> None:
    pytest.importorskip("sklearn")
    prov = SklearnDatasetProvider()
    raw = tmp_path / "raw"
    result = prov.fetch("iris", raw)

    ds_dir = raw / "iris"
    assert (ds_dir / "X.npy").exists() and (ds_dir / "y.npy").exists()
    assert result.license_ok
    manifest = read_manifest(ds_dir)
    assert manifest is not None
    assert manifest.source == "sklearn"
    assert manifest.license == "CC-BY-4.0"
    assert manifest.checksum and len(manifest.checksum) == 64
    assert manifest.retrieved_at
    assert manifest.n_samples == 150


def test_local_registry_provider(tmp_path: Path) -> None:
    reg = tmp_path / "registry"
    ds = reg / "toy"
    ds.mkdir(parents=True)
    np.save(ds / "X.npy", np.arange(20).reshape(10, 2))
    np.save(ds / "y.npy", np.array([0, 1] * 5))
    (ds / "meta.yaml").write_text(
        yaml.safe_dump({"name": "Toy", "license": "MIT", "url": "http://x"})
    )

    prov = LocalRegistryProvider(reg)
    assert prov.search("toy")[0].dataset_id == "toy"
    result = prov.fetch("toy", tmp_path / "raw")
    assert result.license_ok
    assert (tmp_path / "raw" / "toy" / "X.npy").exists()
    assert read_manifest(tmp_path / "raw" / "toy").license == "MIT"




# ---------------------------------------------------------------------------
# License policy: refusal + flagging
# ---------------------------------------------------------------------------


def _registry_without_license(tmp_path: Path) -> LocalRegistryProvider:
    reg = tmp_path / "registry"
    ds = reg / "mystery"
    ds.mkdir(parents=True)
    np.save(ds / "X.npy", np.zeros((4, 2)))
    np.save(ds / "y.npy", np.array([0, 1, 0, 1]))
    (ds / "meta.yaml").write_text(yaml.safe_dump({"name": "Mystery"}))  # no license
    return LocalRegistryProvider(reg)


def test_missing_license_refused_under_strict_policy(tmp_path: Path) -> None:
    prov = _registry_without_license(tmp_path)
    with pytest.raises(LicenseError):
        prov.fetch("mystery", tmp_path / "raw", policy=LicensePolicy(require_license=True))


def test_missing_license_flagged_when_not_strict(tmp_path: Path) -> None:
    prov = _registry_without_license(tmp_path)
    result = prov.fetch("mystery", tmp_path / "raw", policy=LicensePolicy(require_license=False))
    assert result.license_ok is False
    assert any("FLAGGED" in w for w in result.warnings)
    # The manifest still records the flag for auditability.
    assert read_manifest(tmp_path / "raw" / "mystery").license_ok is False


def test_disallowed_license_refused(tmp_path: Path) -> None:
    prov = SklearnDatasetProvider()
    policy = LicensePolicy(require_license=True, allowed=frozenset({"MIT"}))
    with pytest.raises(LicenseError):
        prov.fetch("iris", tmp_path / "raw", policy=policy)  # iris is CC-BY-4.0, not MIT


# ---------------------------------------------------------------------------
# Dedup + validation
# ---------------------------------------------------------------------------


def test_dedup_removes_duplicate_rows() -> None:
    X = np.array([[1, 2], [1, 2], [3, 4], [1, 2]], dtype=float)
    y = np.array([0, 0, 1, 0])
    X2, y2, removed = dedup(X, y)
    assert removed == 2
    assert len(X2) == 2


def test_dedup_keeps_same_features_different_label() -> None:
    X = np.array([[1, 2], [1, 2]], dtype=float)
    y = np.array([0, 1])  # same features but different label -> not a duplicate
    _, _, removed = dedup(X, y)
    assert removed == 0


def test_validate_empty_raises() -> None:
    with pytest.raises(ValidationError):
        validate(np.empty((0, 3)), np.empty((0,)))


def test_validate_length_mismatch_raises() -> None:
    with pytest.raises(ValidationError):
        validate(np.zeros((5, 2)), np.zeros((4,)))


def test_validate_drops_corrupt_samples() -> None:
    X = np.array([[1.0, 2.0], [np.nan, 1.0], [3.0, 4.0]])
    y = np.array([0, 1, 1])
    X2, y2, report = validate(X, y)
    assert report.n_corrupt_removed == 1
    assert report.n_samples == 2
    assert any("corrupt" in w for w in report.warnings)


def test_validate_class_balance_and_coverage_warning() -> None:
    X = np.zeros((4, 2))
    y = np.array([0, 0, 0, 1])
    _, _, report = validate(X, y, min_per_class=2)
    assert report.class_balance == {0: 3, 1: 1}
    assert any("class 1" in w for w in report.warnings)


# ---------------------------------------------------------------------------
# Auto split + seal + lockbox
# ---------------------------------------------------------------------------


def _synthetic(n: int = 100) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n, 4)).astype(np.float32)
    y = (X[:, 0] > 0).astype(np.int64)
    # ensure unique rows so dedup doesn't shrink it
    X += np.arange(n).reshape(n, 1).astype(np.float32) * 1e-3
    return X, y


def test_split_ratios_and_holdout_sealed(tmp_path: Path) -> None:
    root = _make_project(tmp_path)
    X, y = _synthetic(100)
    result = ingest_arrays(root, X, y, train=0.7, val=0.15, test=0.15, seed=42)

    # Sizes roughly match the requested ratios and account for every sample.
    assert result.test_size == 15
    assert result.train_size + result.val_size == 85
    assert abs(result.val_size - 13) <= 2  # 15% of the 85-sample remainder

    # Test split is sealed in the lockbox.
    assert (result.holdout_dir / "X.npy").exists()
    assert result.holdout_dir == root / "benchmark" / "holdout"
    # Train/val are visible in data/processed; test is not.
    assert (root / "data" / "processed" / "X.npy").exists()
    assert (root / "data" / "processed" / "X_val.npy").exists()


def test_split_is_deterministic_by_seed(tmp_path: Path) -> None:
    X, y = _synthetic(80)

    root_a = _make_project(tmp_path / "a")
    r1 = ingest_arrays(root_a, X.copy(), y.copy(), seed=7)
    hold1 = np.load(root_a / "benchmark" / "holdout" / "X.npy")
    train1 = np.load(root_a / "data" / "processed" / "X.npy")

    root_b = _make_project(tmp_path / "b")
    r2 = ingest_arrays(root_b, X.copy(), y.copy(), seed=7)
    hold2 = np.load(root_b / "benchmark" / "holdout" / "X.npy")
    train2 = np.load(root_b / "data" / "processed" / "X.npy")

    assert (r1.train_size, r1.val_size, r1.test_size) == (
        r2.train_size,
        r2.val_size,
        r2.test_size,
    )
    assert np.array_equal(hold1, hold2)
    assert np.array_equal(train1, train2)


def test_agent_cannot_read_sealed_test_split(tmp_path: Path) -> None:
    root = _make_project(tmp_path)
    X, y = _synthetic(60)
    ingest_arrays(root, X, y, seed=1)

    # The lockbox blocks the sealed holdout for the agent's sandbox tools.
    with pytest.raises(LockboxViolation):
        resolve_within_project(root, "benchmark/holdout/X.npy")
    with pytest.raises(LockboxViolation):
        read_file(root, "benchmark/holdout/y.npy")


def test_no_train_test_contamination(tmp_path: Path) -> None:
    root = _make_project(tmp_path)
    X, y = _synthetic(50)
    ingest_arrays(root, X, y, seed=3)

    train = np.load(root / "data" / "processed" / "X.npy")
    val = np.load(root / "data" / "processed" / "X_val.npy")
    held = np.load(root / "benchmark" / "holdout" / "X.npy")

    train_keys = {row.tobytes() for row in np.ascontiguousarray(train)}
    val_keys = {row.tobytes() for row in np.ascontiguousarray(val)}
    held_keys = {row.tobytes() for row in np.ascontiguousarray(held)}
    assert train_keys.isdisjoint(held_keys)
    assert val_keys.isdisjoint(held_keys)
    assert train_keys.isdisjoint(val_keys)


# ---------------------------------------------------------------------------
# Schema: new data config + backward compatibility
# ---------------------------------------------------------------------------


def test_split_ratios_must_sum_to_one() -> None:
    with pytest.raises(ValueError):
        SplitRatios(train=0.5, val=0.3, test=0.3)


def test_project_yaml_backward_compatible_without_data_block(tmp_path: Path) -> None:
    # A pre-M5 project.yaml has no `data:` block; it must still load with defaults.
    legacy = {
        "name": "legacy",
        "description": "old project",
        "task_type": "tabular_classification",
        "target_metric": "accuracy",
    }
    root = tmp_path / "legacy"
    root.mkdir()
    (root / "project.yaml").write_text(yaml.safe_dump(legacy))
    spec = load_project(root)
    assert spec.data.split.test == 0.15
    assert spec.data.license_policy.require_license is True


def test_record_data_source_dedupes(tmp_path: Path) -> None:
    root = _make_project(tmp_path)
    record_data_source(root, DataSourceRef(dataset="iris", source="sklearn", license="CC-BY-4.0"))
    record_data_source(root, DataSourceRef(dataset="iris", source="sklearn", license="MIT"))
    spec = load_project(root)
    assert len(spec.data.sources) == 1
    assert spec.data.sources[0].license == "MIT"  # latest wins


# ---------------------------------------------------------------------------
# Agent-facing data tools
# ---------------------------------------------------------------------------


def test_agent_data_tool_is_ingest_only(tmp_path: Path) -> None:
    """The agent gets exactly one DATA tool — ingest of HUMAN-PROVIDED data.

    Metis no longer sources or scrapes data, so there is no search/download tool.
    """
    pytest.importorskip("sklearn")
    from metis.agent.tools import build_data_tools

    root = _make_project(tmp_path, classes=["0", "1", "2"])
    tools = {t.name: t for t in build_data_tools(root)}
    assert set(tools) == {"ingest_dataset"}

    # The human provides data under data/raw/ (here, via the harness-side provider
    # standing in for a human drop). The agent only ingests it.
    SklearnDatasetProvider().fetch("iris", root / "data" / "raw")

    ing = tools["ingest_dataset"].handler({"dataset": "iris"})
    assert "train=" in ing and "sealed" in ing
    # The agent's returned text must not leak holdout arrays; the test set is sealed.
    assert (root / "benchmark" / "holdout" / "X.npy").exists()
    with pytest.raises(LockboxViolation):
        read_file(root, "benchmark/holdout/X.npy")


# ---------------------------------------------------------------------------
# End-to-end: source -> dedup -> validate -> split -> seal -> train + benchmark
# ---------------------------------------------------------------------------


def test_end_to_end_source_to_benchmark(tmp_path: Path) -> None:
    pytest.importorskip("sklearn")
    from metis.benchmark import BenchmarkRunner
    from metis.training import FAMILIES, build_candidate, train_candidate

    root = _make_project(tmp_path, classes=[str(d) for d in range(10)])

    # SOURCE
    SklearnDatasetProvider().fetch("digits", root / "data" / "raw")
    # DEDUP + VALIDATE + SPLIT + SEAL (test set sealed harness-side)
    result = ingest_dataset(root, root / "data" / "raw" / "digits", seed=42)
    assert result.train_size > 0 and result.test_size > 0
    assert (root / "benchmark" / "holdout" / "X.npy").exists()

    # TRAIN a single candidate on the agent-visible train split, then BENCHMARK.
    cand = build_candidate(FAMILIES["logreg"], dict(FAMILIES["logreg"].default_hparams), "logreg")
    run = train_candidate(root, cand, timeout_s=180)
    assert run.exit_code == 0, run.stderr

    record = BenchmarkRunner().run(root, "logreg")
    assert record.error is None, record.error
    assert record.task_metric_value is not None and record.task_metric_value > 0.5
