"""Harness-side benchmark runner.

The runner is the ONLY path by which a model gets evaluated. It:
  1. Validates the variant directory.
  2. Collects filesystem-derivable efficiency metrics (size, param count).
  3. Evaluates the model against the sealed holdout via an isolated subprocess.
  4. Records the result in benchmark/results.db (append-only).

The evaluation subprocess is harness-authored (written here as a string template)
AND wrapped in an OS-level sandbox (``ossandbox.wrap_sandboxed``) so the sealed
``benchmark/`` subtree is unreachable by the agent's model.py at the kernel layer.
The agent's code therefore never controls what holdout data it sees and cannot
read the holdout off disk to fabricate a perfect score — the anti-gaming
guarantee is structural, not prompt-based.

Defense-in-depth: even within the sandbox the harness never hands the holdout
labels to agent code. The *harness* (this trusted, unsandboxed process) reads
the holdout, writes only the feature array ``X`` to a scratch file outside the
project tree, and passes that to the sandboxed worker. The worker returns
predictions; the harness scores them against the labels it kept. param_count is
measured from the actually-loaded model object inside the sandbox, not trusted
from the agent's recipe.yaml.

Evaluation contract (fulfilled by M3+ training code):
  models/<variant-id>/model.py must export:
    load_model(weights_dir: Path) -> model
    predict(model, X: np.ndarray) -> np.ndarray
  weights are stored under models/<variant-id>/weights/.
  Holdout arrays are at benchmark/holdout/{X,y}.npy (sealed by sealer.py).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from metis.benchmark.metrics import collect_efficiency_metrics
from metis.benchmark.robustness import Corruption, apply_corruption, corruptions_from_config
from metis.benchmark.store import (
    BenchmarkRecord,
    RobustnessRecord,
    append_result,
    append_robustness_result,
)
from metis.projects import load_project
from metis.sandbox.ossandbox import wrap_sandboxed

# Harness-authored evaluation worker injected into a sandboxed subprocess.
#
# The worker NEVER touches benchmark/: it receives the feature array X via a
# scratch .npy outside the project tree (written by the harness, which alone
# reads the sealed holdout) and writes predictions back to another scratch file.
# It also reports param_count measured from the loaded model object, so the
# agent's self-reported recipe.yaml value is not trusted. The OS sandbox makes
# benchmark/ unreachable regardless; this argv contract is belt-and-suspenders.
_EVAL_SCRIPT = r"""
import importlib.util, json, sys, pathlib, time
import numpy as np

variant_dir = pathlib.Path(sys.argv[1])
x_path = pathlib.Path(sys.argv[2])
preds_out = pathlib.Path(sys.argv[3])

# Features only — no labels are ever exposed to agent code.
X = np.load(x_path, allow_pickle=False)

spec = importlib.util.spec_from_file_location("_variant_model", variant_dir / "model.py")
mod_obj = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod_obj)

model = mod_obj.load_model(variant_dir / "weights")

# Batched predictions for the task metric; scored by the harness against labels.
preds = np.asarray(mod_obj.predict(model, X))
np.save(preds_out, preds)


def _measure_params(m):
    # Harness-measured parameter count from the loaded estimator. Prefers exact
    # structural counts per family; returns None when introspection is impossible
    # (the harness then falls back to the recipe value).
    total = 0
    found = False
    for attr in ("coef_", "intercept_"):
        v = getattr(m, attr, None)
        if v is not None:
            total += int(np.asarray(v).size)
            found = True
    if found:
        return total
    tree = getattr(m, "tree_", None)
    if tree is not None and hasattr(tree, "node_count"):
        return int(tree.node_count)
    ests = getattr(m, "estimators_", None)
    if ests is not None:
        try:
            nodes = 0
            for e in np.asarray(ests).ravel():
                t = getattr(e, "tree_", None)
                if t is not None and hasattr(t, "node_count"):
                    nodes += int(t.node_count)
            if nodes:
                return nodes
        except Exception:
            pass
    fit_x = getattr(m, "_fit_X", None)
    if fit_x is None:
        fit_x = getattr(m, "_X", None)
    if fit_x is not None:
        return int(np.asarray(fit_x).size)
    return None


param_count = _measure_params(model)

# --- Live efficiency measurement (harness-owned, not the agent's) ---
# Single-sample latency: time predict() on one example at a time so the
# percentiles reflect real online-serving cost, not amortized batch cost.
n = len(X)
mod_obj.predict(model, X[:1])  # warm up caches / lazy init before timing
n_lat = min(n, 200)
samples_ms = []
for i in range(n_lat):
    t0 = time.perf_counter()
    mod_obj.predict(model, X[i : i + 1])
    samples_ms.append((time.perf_counter() - t0) * 1000.0)
samples_ms.sort()
p50 = samples_ms[int(0.50 * (len(samples_ms) - 1))]
p95 = samples_ms[int(0.95 * (len(samples_ms) - 1))]

# Throughput: batched predict over the full holdout, repeated for a stable
# estimate, reported as samples/second.
reps = 5
t0 = time.perf_counter()
for _ in range(reps):
    mod_obj.predict(model, X)
elapsed = time.perf_counter() - t0
throughput = (n * reps) / elapsed if elapsed > 0 else None

print(json.dumps({
    "param_count": param_count,
    "latency_ms_p50": p50,
    "latency_ms_p95": p95,
    "throughput_sps": throughput,
}))
"""


# Predict-only sandboxed worker, reused by the robustness eval path. Same lockbox
# guarantees as _EVAL_SCRIPT: it receives a feature array via a scratch file
# outside the project tree and can never reach benchmark/ (OS sandbox + the
# harness never handing it labels).
_PREDICT_SCRIPT = r"""
import importlib.util, sys, pathlib
import numpy as np

variant_dir = pathlib.Path(sys.argv[1])
x_path = pathlib.Path(sys.argv[2])
preds_out = pathlib.Path(sys.argv[3])

X = np.load(x_path, allow_pickle=False)

spec = importlib.util.spec_from_file_location("_variant_model", variant_dir / "model.py")
mod_obj = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod_obj)

model = mod_obj.load_model(variant_dir / "weights")
preds = np.asarray(mod_obj.predict(model, X))
np.save(preds_out, preds)
"""


class BenchmarkRunner:
    """Evaluates a model variant and writes the result to results.db."""

    def run(self, project_root: Path, variant_id: str) -> BenchmarkRecord:
        """Evaluate *variant_id*, persist to results.db, and return the record."""
        project = load_project(project_root)
        variant_dir = project_root / "models" / variant_id
        benchmark_dir = project_root / "benchmark"
        holdout_dir = benchmark_dir / "holdout"

        benchmark_dir.mkdir(parents=True, exist_ok=True)

        if not variant_dir.is_dir():
            record = BenchmarkRecord(
                variant_id=variant_id,
                task_metric_name=project.target_metric,
                task_metric_value=None,
                param_count=None,
                model_size_mb=None,
                latency_ms_p50=None,
                latency_ms_p95=None,
                throughput_sps=None,
                error=f"variant directory not found: models/{variant_id}",
            )
            append_result(benchmark_dir, record)
            return record

        eff = collect_efficiency_metrics(variant_dir)
        metrics, error = self._evaluate(
            project_root, variant_dir, holdout_dir, project.target_metric
        )
        metrics = metrics or {}

        # param_count is measured from the loaded model inside the sandbox; only
        # fall back to the agent's self-reported recipe value when introspection
        # was impossible. This closes the "lie in recipe.yaml" gaming vector.
        measured_params = metrics.get("param_count")
        param_count = int(measured_params) if measured_params is not None else eff.param_count

        record = BenchmarkRecord(
            variant_id=variant_id,
            task_metric_name=project.target_metric,
            task_metric_value=metrics.get("task_metric_value"),
            param_count=param_count,
            model_size_mb=eff.model_size_mb,
            # Latency/throughput are measured live by the eval subprocess; the
            # filesystem-only metrics carry None for these.
            latency_ms_p50=metrics.get("latency_ms_p50"),
            latency_ms_p95=metrics.get("latency_ms_p95"),
            throughput_sps=metrics.get("throughput_sps"),
            error=error,
        )
        append_result(benchmark_dir, record)
        return record

    @staticmethod
    def _score(metric_name: str, preds, y) -> float:
        """Compute the task metric in the trusted harness (labels never leave it)."""
        import numpy as np

        preds = np.asarray(preds)
        mn = metric_name.lower()
        if mn == "f1":
            from sklearn.metrics import f1_score

            return float(f1_score(y, preds, average="weighted"))
        if mn == "auroc":
            from sklearn.metrics import roc_auc_score

            return float(roc_auc_score(y, preds, multi_class="ovr"))
        if mn == "rmse":
            return float(np.sqrt(np.mean((preds.astype(float) - y.astype(float)) ** 2)))
        if mn == "mae":
            return float(np.mean(np.abs(preds.astype(float) - y.astype(float))))
        # accuracy (and default)
        return float(np.mean(preds == y))

    def _evaluate(
        self, project_root: Path, variant_dir: Path, holdout_dir: Path, metric_name: str
    ) -> tuple[dict[str, float] | None, str | None]:
        import numpy as np

        model_py = variant_dir / "model.py"
        holdout_X = holdout_dir / "X.npy"
        holdout_y = holdout_dir / "y.npy"

        if not model_py.exists():
            return None, "no model.py found in variant directory; train the model first"
        if not holdout_X.exists() or not holdout_y.exists():
            return None, "holdout not sealed yet; the harness must run seal_holdout first"

        # The harness (trusted, unsandboxed) reads the sealed holdout. The labels
        # never leave this process; only features are handed to the sandboxed
        # worker, which itself cannot reach benchmark/ at the kernel layer.
        X = np.load(holdout_X, allow_pickle=False)
        y = np.load(holdout_y, allow_pickle=False)

        # Scratch dir OUTSIDE the project tree: features in, predictions out.
        # Putting it outside the project means project-relative traversal from
        # the worker can never reach the holdout even before the OS sandbox.
        scratch = Path(tempfile.mkdtemp(prefix="metis-eval-"))
        script_path = scratch / "_eval.py"
        x_path = scratch / "X.npy"
        preds_path = scratch / "preds.npy"
        script_path.write_text(_EVAL_SCRIPT)
        np.save(x_path, X)

        benchmark_dir = (project_root.resolve() / "benchmark").resolve()
        # variant_dir must be ABSOLUTE: the worker runs with cwd=scratch (outside the
        # project tree), so a relative path would resolve against scratch and fail to
        # find model.py — and would also fall outside the OS sandbox's allowed root.
        cmd = [
            sys.executable,
            str(script_path),
            str(variant_dir.resolve()),
            str(x_path),
            str(preds_path),
        ]
        # Wrap in the OS sandbox so model.py cannot read benchmark/ via any means
        # (relative/absolute path, os.open, ctypes/libc, or a spawned subprocess).
        cmd = wrap_sandboxed(cmd, project_root.resolve(), benchmark_dir)
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
                cwd=str(scratch),  # cwd outside the project tree
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
        except subprocess.TimeoutExpired:
            return None, "evaluation timed out after 300 s"
        except Exception as exc:
            return None, f"evaluation subprocess error: {exc}"
        finally:
            script_path.unlink(missing_ok=True)
            x_path.unlink(missing_ok=True)

        if result.returncode != 0:
            stderr = result.stderr.strip()[:500]
            return None, f"evaluation script failed: {stderr}"

        try:
            data = json.loads(result.stdout.strip())
        except Exception as exc:
            return None, f"could not parse evaluation output: {exc}"

        if not preds_path.exists():
            return None, "evaluation worker produced no predictions"
        try:
            preds = np.load(preds_path, allow_pickle=False)
        except Exception as exc:
            return None, f"could not load predictions: {exc}"
        finally:
            preds_path.unlink(missing_ok=True)

        if len(preds) != len(y):
            return None, "prediction count does not match holdout size"

        try:
            score = self._score(metric_name, preds, y)
        except Exception as exc:
            return None, f"scoring failed: {exc}"

        metrics: dict[str, float] = {"task_metric_value": score}
        for key in ("latency_ms_p50", "latency_ms_p95", "throughput_sps", "param_count"):
            val = data.get(key)
            if val is not None:
                metrics[key] = float(val)
        return metrics, None

    def _run_predictions(
        self, project_root: Path, variant_dir: Path, X
    ) -> tuple[object | None, str | None]:
        """Run the variant's predict() over feature matrix *X* in the OS sandbox.

        Used by the robustness path. The features handed in are already the
        (possibly corrupted) holdout features the *harness* prepared; the worker
        never reaches benchmark/ and never receives labels.
        """
        import numpy as np

        scratch = Path(tempfile.mkdtemp(prefix="metis-robust-"))
        script_path = scratch / "_predict.py"
        x_path = scratch / "X.npy"
        preds_path = scratch / "preds.npy"
        script_path.write_text(_PREDICT_SCRIPT)
        np.save(x_path, np.asarray(X))

        benchmark_dir = (project_root.resolve() / "benchmark").resolve()
        # Absolute variant_dir — the worker's cwd is scratch, outside the project tree.
        cmd = [
            sys.executable,
            str(script_path),
            str(variant_dir.resolve()),
            str(x_path),
            str(preds_path),
        ]
        cmd = wrap_sandboxed(cmd, project_root.resolve(), benchmark_dir)
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
                cwd=str(scratch),
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
        except subprocess.TimeoutExpired:
            return None, "prediction timed out after 300 s"
        except Exception as exc:
            return None, f"prediction subprocess error: {exc}"
        finally:
            script_path.unlink(missing_ok=True)
            x_path.unlink(missing_ok=True)

        if result.returncode != 0:
            return None, f"prediction script failed: {result.stderr.strip()[:500]}"
        if not preds_path.exists():
            return None, "prediction worker produced no predictions"
        try:
            preds = np.load(preds_path, allow_pickle=False)
        except Exception as exc:
            return None, f"could not load predictions: {exc}"
        finally:
            preds_path.unlink(missing_ok=True)
        return preds, None

    def run_robustness(
        self,
        project_root: Path,
        variant_id: str,
        *,
        corruptions: list[Corruption] | None = None,
        seed: int | None = None,
    ) -> RobustnessRecord:
        """Score *variant_id* on clean + corrupted holdout and persist the result.

        Robustness stays entirely harness-side: this trusted process reads the
        sealed holdout, perturbs the features, evaluates predictions through the
        same OS-sandboxed worker the benchmark uses, and scores against labels it
        never exposes. Records per-corruption scores plus an aggregate retention
        ratio (mean corrupted-score / clean-score) to results.db.
        """
        import numpy as np

        project = load_project(project_root)
        metric = project.target_metric
        variant_dir = project_root / "models" / variant_id
        benchmark_dir = project_root / "benchmark"
        holdout_dir = benchmark_dir / "holdout"
        benchmark_dir.mkdir(parents=True, exist_ok=True)

        if corruptions is None:
            corruptions = corruptions_from_config(project.robustness)
        if seed is None:
            seed = project.robustness.seed

        def _fail(msg: str) -> RobustnessRecord:
            rec = RobustnessRecord(
                variant_id=variant_id,
                task_metric_name=metric,
                clean_score=None,
                aggregate_robustness=None,
                per_corruption={},
                error=msg,
            )
            append_robustness_result(benchmark_dir, rec)
            return rec

        if not (variant_dir / "model.py").exists():
            return _fail(f"no model.py for variant {variant_id!r}; train it first")
        holdout_X = holdout_dir / "X.npy"
        holdout_y = holdout_dir / "y.npy"
        if not holdout_X.exists() or not holdout_y.exists():
            return _fail("holdout not sealed yet; the harness must run seal_holdout first")

        # Trusted harness reads the sealed holdout; labels never leave this process.
        X = np.load(holdout_X, allow_pickle=False)
        y = np.load(holdout_y, allow_pickle=False)

        clean_preds, err = self._run_predictions(project_root, variant_dir, X)
        if err is not None or clean_preds is None:
            return _fail(err or "clean prediction failed")
        try:
            clean_score = self._score(metric, clean_preds, y)
        except Exception as exc:
            return _fail(f"scoring failed: {exc}")

        rng = np.random.default_rng(seed)
        per_corruption: dict[str, float] = {}
        for corruption in corruptions:
            Xc = apply_corruption(X, corruption, rng)
            preds, perr = self._run_predictions(project_root, variant_dir, Xc)
            if perr is not None or preds is None:
                return _fail(f"corruption {corruption.name!r} failed: {perr}")
            try:
                per_corruption[corruption.name] = self._score(metric, preds, y)
            except Exception as exc:
                return _fail(f"scoring corruption {corruption.name!r} failed: {exc}")

        lower_better = metric.lower() in {"rmse", "mae", "mse", "loss", "error_rate"}
        aggregate = self._aggregate_robustness(clean_score, per_corruption, lower_better)

        record = RobustnessRecord(
            variant_id=variant_id,
            task_metric_name=metric,
            clean_score=clean_score,
            aggregate_robustness=aggregate,
            per_corruption=per_corruption,
            error=None,
        )
        append_robustness_result(benchmark_dir, record)
        return record

    @staticmethod
    def _aggregate_robustness(
        clean_score: float, per_corruption: dict[str, float], lower_is_better: bool
    ) -> float | None:
        """Mean retention ratio across corruptions (1.0 = no degradation).

        For higher-is-better metrics this is mean(corrupted / clean); for
        lower-is-better metrics it is inverted (mean(clean / corrupted)) so the
        score stays "higher = more robust" in both cases.
        """
        if not per_corruption:
            return None
        ratios: list[float] = []
        for score in per_corruption.values():
            if lower_is_better:
                ratios.append(clean_score / score if score > 0 else 1.0)
            else:
                ratios.append(score / clean_score if clean_score > 0 else 0.0)
        return float(sum(ratios) / len(ratios))
