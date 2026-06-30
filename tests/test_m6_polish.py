"""Tests for M6 polish: token management, export, reproducibility bundle, robustness."""

from __future__ import annotations

import pickle
import textwrap
import zipfile
from pathlib import Path

import pytest

from metis.projects import create_project
from metis.projects.schema import ProjectSpec, TaskType


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def project_root(tmp_path: Path) -> Path:
    spec = ProjectSpec(
        name="m6",
        description="m6 test project",
        task_type=TaskType.tabular_classification,
        classes=["0", "1", "2"],
        target_metric="accuracy",
    )
    return create_project(tmp_path / "m6", spec)


_MODEL_PY = textwrap.dedent("""\
    import pickle
    from pathlib import Path

    def load_model(weights_dir):
        with open(Path(weights_dir) / "model.pkl", "rb") as f:
            return pickle.load(f)

    def predict(model, X):
        return model.predict(X)
""")


@pytest.fixture
def trained_variant(project_root: Path) -> str:
    """Train a real sklearn logreg variant + seal a numpy holdout. Returns variant id."""
    np = pytest.importorskip("numpy")
    pytest.importorskip("sklearn")
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 5)).astype("float32")
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    # Three classes so robustness/scoring exercise multi-class.
    y = (X[:, 0] > 0).astype(int) + (X[:, 1] > 0).astype(int)

    clf = LogisticRegression(max_iter=500).fit(X, y)

    variant_dir = project_root / "models" / "logreg"
    weights = variant_dir / "weights"
    weights.mkdir(parents=True)
    with (weights / "model.pkl").open("wb") as f:
        pickle.dump(clf, f)
    (variant_dir / "model.py").write_text(_MODEL_PY)
    (variant_dir / "train.py").write_text("# training code for logreg\n")
    (variant_dir / "recipe.yaml").write_text("architecture: logistic_regression\nparam_count: 18\n")

    # Train-data snapshot under data/processed (NOT the holdout).
    processed = project_root / "data" / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    np.save(processed / "X.npy", X[:80])
    np.save(processed / "y.npy", y[:80])

    # Sealed holdout with DISTINCTIVE labels (so leakage would be detectable).
    holdout = project_root / "benchmark" / "holdout"
    holdout.mkdir(parents=True, exist_ok=True)
    np.save(holdout / "X.npy", X[80:])
    np.save(holdout / "y.npy", y[80:])
    return "logreg"


# ---------------------------------------------------------------------------
# 1. Token / credential management
# ---------------------------------------------------------------------------


def _store(tmp_path: Path):
    from metis.agent.credentials import FileCredentialStore

    return FileCredentialStore(path=tmp_path / "creds.json")


def test_credential_roundtrip_set_get_clear(tmp_path: Path) -> None:
    store = _store(tmp_path)
    assert store.get("anthropic") is None
    assert store.has("anthropic") is False

    store.set("sk-ant-secret-value-123456", "anthropic")
    assert store.get("anthropic") == "sk-ant-secret-value-123456"
    assert store.has("anthropic") is True

    assert store.clear("anthropic") is True
    assert store.get("anthropic") is None
    assert store.clear("anthropic") is False  # nothing left to clear


def test_credential_file_permissions_are_owner_only(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.set("sk-ant-secret-value-123456", "anthropic")
    mode = (tmp_path / "creds.json").stat().st_mode & 0o777
    assert mode == 0o600


def test_credential_rejects_empty(tmp_path: Path) -> None:
    store = _store(tmp_path)
    with pytest.raises(ValueError):
        store.set("   ", "anthropic")


def test_mask_key_never_reveals_secret() -> None:
    from metis.agent.credentials import mask_key

    secret = "sk-ant-super-secret-key-abcdef"
    masked = mask_key(secret)
    assert secret not in masked
    # No 4+ char run of the real secret should appear in the masked string.
    assert "abcdef" not in masked
    assert "sk-ant" not in masked
    assert mask_key(None) == "(not set)"
    assert mask_key("") == "(not set)"


def test_provider_chain_prefers_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from metis.agent.credentials import (
        ChainedCredentialProvider,
        EnvCredentialProvider,
        StoredCredentialProvider,
    )

    store = _store(tmp_path)
    store.set("sk-from-file", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-from-env")
    chain = ChainedCredentialProvider([EnvCredentialProvider(), StoredCredentialProvider(store)])
    assert chain.get_api_key() == "sk-from-env"
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert chain.get_api_key() == "sk-from-file"


def test_anthropic_client_uses_stored_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("anthropic")
    from metis.agent.anthropic_client import AnthropicClient
    from metis.agent.credentials import StoredCredentialProvider

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    store = _store(tmp_path)
    store.set("sk-ant-stored-key-123456", "anthropic")
    # Construction succeeds resolving the key through the stored provider.
    client = AnthropicClient(credential_provider=StoredCredentialProvider(store))
    assert client is not None


def test_oauth_provider_is_stubbed() -> None:
    from metis.agent.credentials import OAuthCredentialProvider

    provider = OAuthCredentialProvider()
    assert provider.get_api_key() is None
    with pytest.raises(NotImplementedError):
        provider.begin_authorization()


def test_secret_never_written_to_logs_results_or_project(
    project_root: Path, tmp_path: Path
) -> None:
    """The secret must not land in runs/, results.db, or project.yaml."""
    from metis.agent.credentials import FileCredentialStore
    from metis.benchmark.store import BenchmarkRecord, append_result
    from metis.sandbox.runlog import log_action

    secret = "sk-ant-DO-NOT-LEAK-987654321"
    store = FileCredentialStore(path=tmp_path / "creds.json")
    store.set(secret, "anthropic")

    # Simulate ordinary harness activity after a key is set.
    log_action(project_root, "save_project_spec", {"name": "m6"}, ok=True)
    append_result(
        project_root / "benchmark",
        BenchmarkRecord(
            variant_id="v",
            task_metric_name="accuracy",
            task_metric_value=0.9,
            param_count=10,
            model_size_mb=0.01,
            latency_ms_p50=None,
            latency_ms_p95=None,
            throughput_sps=None,
        ),
    )

    # None of the project files may contain the secret.
    for path in project_root.rglob("*"):
        if path.is_file():
            data = path.read_bytes()
            assert secret.encode() not in data, f"secret leaked into {path}"

    # The credentials file lives OUTSIDE the project tree.
    assert (tmp_path / "creds.json").exists()
    assert not str(tmp_path / "creds.json").startswith(str(project_root))


# ---------------------------------------------------------------------------
# 2. Export / portability
# ---------------------------------------------------------------------------


def test_export_pickle_bundle_is_loadable_and_recorded(
    project_root: Path, trained_variant: str
) -> None:
    from metis.portability import export_variant

    # Force the fallback so this always exercises the pickle path.
    result = export_variant(project_root, trained_variant, prefer_onnx=False)
    assert result.format == "pickle_bundle"
    artifact = Path(result.path)
    assert artifact.exists()
    assert result.checksum  # recorded

    # The bundle is a self-contained, loadable zip.
    with zipfile.ZipFile(artifact) as z:
        names = z.namelist()
        assert "model.pkl" in names
        with z.open("model.pkl") as f:
            model = pickle.load(f)
    np = pytest.importorskip("numpy")
    preds = model.predict(np.zeros((3, 5), dtype="float32"))
    assert len(preds) == 3

    # The export manifest records format + path + checksum.
    import yaml

    manifest = yaml.safe_load(
        (project_root / "models" / trained_variant / "exports" / "export.yaml").read_text()
    )
    assert manifest["exports"][0]["format"] == "pickle_bundle"
    assert manifest["exports"][0]["checksum"] == result.checksum


def test_export_onnx_when_available(project_root: Path, trained_variant: str) -> None:
    pytest.importorskip("skl2onnx")
    pytest.importorskip("onnx")
    import onnx

    from metis.portability import export_variant

    result = export_variant(project_root, trained_variant, prefer_onnx=True)
    assert result.format == "onnx"
    artifact = Path(result.path)
    assert artifact.exists()
    onnx.checker.check_model(onnx.load(str(artifact)))  # valid ONNX
    assert result.checksum


def test_export_missing_variant(project_root: Path) -> None:
    from metis.portability import export_variant

    with pytest.raises(FileNotFoundError):
        export_variant(project_root, "ghost")


# ---------------------------------------------------------------------------
# 3. Reproducibility bundle
# ---------------------------------------------------------------------------


def test_bundle_contains_required_parts(project_root: Path, trained_variant: str) -> None:
    from metis.benchmark import BenchmarkRunner
    from metis.portability import build_repro_bundle

    # Produce a recorded benchmark result first so the bundle can include it.
    BenchmarkRunner().run(project_root, trained_variant)

    result = build_repro_bundle(project_root, trained_variant)
    d = result.directory
    assert (d / "recipe.yaml").exists()
    assert (d / "code" / "train.py").exists()
    assert (d / "code" / "model.py").exists()
    assert (d / "environment.yaml").exists()
    assert (d / "data_manifest.yaml").exists()
    assert (d / "benchmark_results.yaml").exists()
    assert (d / "manifest.yaml").exists()
    assert result.archive.exists()

    import yaml

    env = yaml.safe_load((d / "environment.yaml").read_text())
    assert "python_version" in env and "platform" in env and "packages" in env

    data_manifest = yaml.safe_load((d / "data_manifest.yaml").read_text())
    assert "processed_files" in data_manifest
    # Train snapshot checksummed (manifest, not raw holdout).
    assert any(f["sha256"] for f in data_manifest["processed_files"])

    results = yaml.safe_load((d / "benchmark_results.yaml").read_text())
    assert results["benchmark_runs"], "expected the variant's recorded result rows"


def test_bundle_excludes_holdout_and_scoring(project_root: Path, trained_variant: str) -> None:
    """The bundle must never contain sealed holdout data or scoring code."""
    np = pytest.importorskip("numpy")
    from metis.portability import build_repro_bundle

    # Plant a scoring-code marker in benchmark/ to be sure it is never copied.
    (project_root / "benchmark" / "suite.py").write_text("# SECRET SCORING CODE\n")

    result = build_repro_bundle(project_root, trained_variant)

    holdout_y = np.load(project_root / "benchmark" / "holdout" / "y.npy")

    for path in result.directory.rglob("*"):
        rel = str(path.relative_to(result.directory))
        assert "holdout" not in rel, f"holdout path leaked into bundle: {rel}"
        assert path.name != "suite.py", "scoring code leaked into bundle"
        assert path.name != "results.db", "results.db must not be copied into the bundle"
        if path.is_file():
            assert b"SECRET SCORING CODE" not in path.read_bytes(), (
                f"scoring code leaked into {rel}"
            )

    # Also check the zipped archive carries no holdout/scoring entries.
    with zipfile.ZipFile(result.archive) as z:
        for name in z.namelist():
            assert "holdout" not in name
            assert not name.endswith("suite.py")
            assert not name.endswith("results.db")

    # Sanity: holdout still exists where it belongs (we didn't move it).
    assert (project_root / "benchmark" / "holdout" / "y.npy").exists()
    assert len(holdout_y) > 0


# ---------------------------------------------------------------------------
# 4. Robustness benchmarks
# ---------------------------------------------------------------------------


def test_robustness_records_per_corruption_and_aggregate(
    project_root: Path, trained_variant: str
) -> None:
    from metis.benchmark import BenchmarkRunner, get_latest_robustness

    record = BenchmarkRunner().run_robustness(project_root, trained_variant)
    assert record.error is None, record.error
    assert record.clean_score is not None
    assert record.aggregate_robustness is not None
    # Default suite: three named corruptions, each with a recorded score.
    assert set(record.per_corruption) == {"gaussian_noise", "feature_dropout", "scaling"}
    for score in record.per_corruption.values():
        assert 0.0 <= score <= 1.0

    # Persisted to results.db and retrievable.
    latest = get_latest_robustness(project_root / "benchmark")
    assert trained_variant in latest
    assert latest[trained_variant]["aggregate_robustness"] is not None
    assert set(latest[trained_variant]["per_corruption"]) == set(record.per_corruption)


def test_robustness_is_harness_side_not_an_agent_tool(project_root: Path) -> None:
    """The agent has no robustness tool and never sees holdout/perturbations."""
    from metis.agent.tools import build_benchmark_tools

    tool_names = {t.name for t in build_benchmark_tools(project_root)}
    assert not any("robust" in n for n in tool_names)
    assert not any("corrupt" in n for n in tool_names)


def test_robustness_corruption_eval_cannot_read_holdout(project_root: Path) -> None:
    """A malicious model.py cannot read the sealed labels during robustness eval."""
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
                root = Path(__file__).resolve().parents[2]
                y = np.load(root / "benchmark" / "holdout" / "y.npy")
                return np.asarray(y)[: len(X)]
        """)
    )
    holdout = project_root / "benchmark" / "holdout"
    holdout.mkdir(parents=True, exist_ok=True)
    n = 24
    np.save(holdout / "X.npy", np.zeros((n, 4)))
    np.save(holdout / "y.npy", np.array([i % 3 for i in range(n)], dtype=int))

    from metis.benchmark import BenchmarkRunner

    record = BenchmarkRunner().run_robustness(project_root, "evil")
    # Either the sandbox denied the read (error), or it predicted without the
    # true labels — never a perfect clean score by reading the holdout.
    assert record.error is not None or (
        record.clean_score is not None and record.clean_score < 0.99
    ), f"malicious model cheated robustness eval: {record}"
