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
    """A proposed model variant: id + the train.py/model.py the harness writes."""

    variant_id: str
    family: str
    train_py: str
    model_py: str


@dataclass(frozen=True)
class FamilySpec:
    """A model family + its hyperparameter space, used to PROPOSE and to BRANCH.

    ``ctor_template`` is a ``.format``-style string with one slot per hyperparameter
    (e.g. ``"LogisticRegression(C={C}, max_iter=2000)"``). ``hparam_grid`` defines
    the discrete values the evolutionary search may perturb each hyperparameter to.
    """

    key: str  # short prefix used in variant ids, e.g. "logreg"
    family: str  # human-readable family name
    import_line: str
    ctor_template: str
    param_expr: str
    default_hparams: dict[str, object]
    hparam_grid: dict[str, list[object]]


# Registry of supported families. The first three are PROPOSEd up front; the
# remainder are held back so BRANCH can introduce genuinely new families when the
# leaderboard plateaus.
FAMILIES: dict[str, FamilySpec] = {
    "logreg": FamilySpec(
        key="logreg",
        family="logistic_regression",
        import_line="from sklearn.linear_model import LogisticRegression",
        ctor_template="LogisticRegression(C={C}, max_iter=2000)",
        param_expr="clf.coef_.size + clf.intercept_.size",
        default_hparams={"C": 1.0},
        hparam_grid={"C": [0.01, 0.1, 1.0, 10.0]},
    ),
    "decision_tree": FamilySpec(
        key="decision_tree",
        family="decision_tree",
        import_line="from sklearn.tree import DecisionTreeClassifier",
        ctor_template="DecisionTreeClassifier(max_depth={max_depth}, random_state=0)",
        param_expr="clf.tree_.node_count",
        default_hparams={"max_depth": 10},
        hparam_grid={"max_depth": [3, 5, 10, 20]},
    ),
    "knn": FamilySpec(
        key="knn",
        family="k_nearest_neighbors",
        import_line="from sklearn.neighbors import KNeighborsClassifier",
        ctor_template="KNeighborsClassifier(n_neighbors={n_neighbors})",
        param_expr="X.size",
        default_hparams={"n_neighbors": 5},
        hparam_grid={"n_neighbors": [1, 3, 5, 11]},
    ),
    "random_forest": FamilySpec(
        key="random_forest",
        family="random_forest",
        import_line="from sklearn.ensemble import RandomForestClassifier",
        ctor_template=(
            "RandomForestClassifier(n_estimators={n_estimators}, "
            "max_depth={max_depth}, random_state=0)"
        ),
        param_expr="sum(int(e.tree_.node_count) for e in clf.estimators_)",
        default_hparams={"n_estimators": 50, "max_depth": 10},
        hparam_grid={"n_estimators": [20, 50, 100], "max_depth": [5, 10, None]},
    ),
    "mlp": FamilySpec(
        key="mlp",
        family="mlp",
        import_line="from sklearn.neural_network import MLPClassifier",
        ctor_template="MLPClassifier(hidden_layer_sizes=({hidden},), max_iter=500, random_state=0)",
        param_expr="sum(c.size for c in clf.coefs_) + sum(b.size for b in clf.intercepts_)",
        default_hparams={"hidden": 32},
        hparam_grid={"hidden": [16, 32, 64]},
    ),
}

# Families PROPOSEd at the start of a project (a breadth of distinct families).
_PROPOSE_KEYS: tuple[str, ...] = ("logreg", "decision_tree", "knn")


def _render_ctor(spec: FamilySpec, hparams: dict[str, object]) -> str:
    """Render the constructor call, emitting valid Python literals (None, ints, …)."""
    rendered = {k: repr(v) for k, v in hparams.items()}
    return spec.ctor_template.format(**rendered)


def build_candidate(
    spec: FamilySpec,
    hparams: dict[str, object],
    variant_id: str,
) -> Candidate:
    """Materialize a concrete training candidate from a family + hyperparameters."""
    ctor = _render_ctor(spec, hparams)
    return Candidate(
        variant_id=variant_id,
        family=spec.family,
        train_py=_TRAIN_PY.format(
            family=spec.family,
            imports=spec.import_line,
            ctor=ctor,
            param_expr=spec.param_expr,
        ),
        model_py=_MODEL_PY,
    )


def propose_candidates() -> list[Candidate]:
    """Propose a breadth of model families suited to small image/tabular data."""
    return [
        build_candidate(FAMILIES[key], dict(FAMILIES[key].default_hparams), key)
        for key in _PROPOSE_KEYS
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
    """Scaffold then train a candidate via the sandboxed ``run_python`` tool.

    The wall-clock cost of the run is recorded into the harness-owned resource
    ledger (``results.db``) so budgets can be enforced harness-side.
    """
    from metis.benchmark.budget import record_training_usage

    scaffold_candidate(project_root, candidate)
    result = run_python(
        project_root,
        f"models/{candidate.variant_id}/train.py",
        timeout_s=timeout_s,
        memory_mb=memory_mb,
    )
    record_training_usage(
        project_root,
        variant_id=candidate.variant_id,
        wall_clock_s=result.duration_s,
        detail=f"exit={result.exit_code} timed_out={result.timed_out}",
    )
    return result


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
    from metis.benchmark.budget import compute_budget_status

    prepare_toy_dataset(project_root)
    seal_holdout(project_root, fraction=fraction, seed=seed, mode="numpy")

    runner = BenchmarkRunner()
    records: list[BenchmarkRecord] = []
    for candidate in propose_candidates():
        # Budget is enforced by the harness before each train, never trusted to
        # the agent: stop the loop the moment any declared budget is exhausted.
        if compute_budget_status(project_root).should_stop:
            break
        result = train_candidate(project_root, candidate, timeout_s=timeout_s)
        if result.exit_code != 0:
            raise RuntimeError(
                f"training {candidate.variant_id} failed (exit {result.exit_code}):\n{result.stderr}"
            )
        records.append(runner.run(project_root, candidate.variant_id))
    return records
