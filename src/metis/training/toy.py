"""A toy, fully-runnable PROPOSE -> TRAIN path (sklearn digits, 8x8 images).

Dependencies are deliberately light (numpy + scikit-learn, no torch) so the
loop runs fast in CI. The dataset is scikit-learn's ``load_digits`` — 1797
8x8 grayscale handwritten-digit images, 10 classes — flattened to 64 features.

Each candidate is a distinct model *family*; both serialize a pickled estimator
to ``weights/model.pkl`` and share the same ``model.py`` (load + predict) so the
harness benchmark can evaluate them uniformly. Training runs through the
sandboxed ``run_python`` tool, so it inherits the lockbox + budget guarantees.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from metis.benchmark import BenchmarkRecord, BenchmarkRunner, seal_holdout
from metis.sandbox import RunResult, run_python, write_file

# Shared variant contract: load a pickled sklearn estimator and predict with it.
_MODEL_PY = '''\
"""Variant inference contract used by the harness benchmark runner."""

from __future__ import annotations

import pickle
from pathlib import Path


def load_model(weights_dir):
    with open(Path(weights_dir) / "model.pkl", "rb") as f:
        return pickle.load(f)


def predict(model, X):
    return model.predict(X)
'''

# A train.py template. {imports}/{ctor}/{family}/{param_expr} are filled per
# candidate. The script loads the (already holdout-stripped) processed data,
# fits, pickles weights, and records a recipe.yaml with the real param count.
_TRAIN_PY = '''\
"""PROPOSE/TRAIN candidate: {family}."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import yaml
{imports}

variant_dir = Path(__file__).resolve().parent
project_root = variant_dir.parents[1]
processed = project_root / "data" / "processed"

X = np.load(processed / "X.npy")
y = np.load(processed / "y.npy")

clf = {ctor}
clf.fit(X, y)

weights = variant_dir / "weights"
weights.mkdir(parents=True, exist_ok=True)
with open(weights / "model.pkl", "wb") as f:
    pickle.dump(clf, f)

param_count = int({param_expr})
(variant_dir / "recipe.yaml").write_text(
    yaml.safe_dump(
        {{"architecture": "{family}", "param_count": param_count}},
        sort_keys=False,
    )
)
print(f"trained {family}: {{param_count}} params on {{len(X)}} samples")
'''


@dataclass(frozen=True)
class Candidate:
    """A proposed model family: id + the train.py/model.py the harness writes."""

    variant_id: str
    family: str
    train_py: str
    model_py: str


def _candidate(variant_id: str, family: str, imports: str, ctor: str, param_expr: str) -> Candidate:
    return Candidate(
        variant_id=variant_id,
        family=family,
        train_py=_TRAIN_PY.format(family=family, imports=imports, ctor=ctor, param_expr=param_expr),
        model_py=_MODEL_PY,
    )


def propose_candidates() -> list[Candidate]:
    """Propose a breadth of model families suited to small image/tabular data."""
    return [
        _candidate(
            variant_id="logreg",
            family="logistic_regression",
            imports="from sklearn.linear_model import LogisticRegression",
            ctor="LogisticRegression(max_iter=2000)",
            param_expr="clf.coef_.size + clf.intercept_.size",
        ),
        _candidate(
            variant_id="decision_tree",
            family="decision_tree",
            imports="from sklearn.tree import DecisionTreeClassifier",
            ctor="DecisionTreeClassifier(max_depth=10, random_state=0)",
            param_expr="clf.tree_.node_count",
        ),
        _candidate(
            variant_id="knn",
            family="k_nearest_neighbors",
            imports="from sklearn.neighbors import KNeighborsClassifier",
            ctor="KNeighborsClassifier(n_neighbors=5)",
            param_expr="X.size",
        ),
    ]


def prepare_toy_dataset(project_root: Path) -> Path:
    """Write the toy digits dataset to ``data/processed/{X,y}.npy``.

    Returns the processed-data directory. Must be called before sealing.
    """
    import numpy as np
    from sklearn.datasets import load_digits

    digits = load_digits()
    X = np.asarray(digits.data, dtype=np.float32)
    y = np.asarray(digits.target, dtype=np.int64)

    processed = project_root / "data" / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    np.save(processed / "X.npy", X)
    np.save(processed / "y.npy", y)
    return processed


def scaffold_candidate(project_root: Path, candidate: Candidate) -> Path:
    """Write a candidate's train.py + model.py under ``models/<variant_id>/``."""
    base = f"models/{candidate.variant_id}"
    write_file(project_root, f"{base}/train.py", candidate.train_py)
    write_file(project_root, f"{base}/model.py", candidate.model_py)
    return project_root / "models" / candidate.variant_id


def train_candidate(
    project_root: Path,
    candidate: Candidate,
    *,
    timeout_s: float = 120.0,
    memory_mb: int | None = None,
) -> RunResult:
    """Scaffold then train a candidate via the sandboxed ``run_python`` tool."""
    scaffold_candidate(project_root, candidate)
    return run_python(
        project_root,
        f"models/{candidate.variant_id}/train.py",
        timeout_s=timeout_s,
        memory_mb=memory_mb,
    )


def run_toy_pipeline(
    project_root: Path,
    *,
    fraction: float = 0.2,
    seed: int = 42,
    timeout_s: float = 120.0,
) -> list[BenchmarkRecord]:
    """Full PROPOSE -> TRAIN -> BENCHMARK loop on the toy dataset.

    Prepares data, seals a holdout (harness-side), trains every proposed
    candidate through the sandbox, then benchmarks each on the sealed holdout.
    Returns the per-variant benchmark records.
    """
    prepare_toy_dataset(project_root)
    seal_holdout(project_root, fraction=fraction, seed=seed, mode="numpy")

    runner = BenchmarkRunner()
    records: list[BenchmarkRecord] = []
    for candidate in propose_candidates():
        result = train_candidate(project_root, candidate, timeout_s=timeout_s)
        if result.exit_code != 0:
            raise RuntimeError(
                f"training {candidate.variant_id} failed (exit {result.exit_code}):\n{result.stderr}"
            )
        records.append(runner.run(project_root, candidate.variant_id))
    return records
