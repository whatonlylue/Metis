"""Harness-side benchmark runner.

The runner is the ONLY path by which a model gets evaluated. It:
  1. Validates the variant directory.
  2. Collects filesystem-derivable efficiency metrics (size, param count).
  3. Evaluates the model against the sealed holdout via an isolated subprocess.
  4. Records the result in benchmark/results.db (append-only).

The evaluation subprocess is harness-authored (written here as a string template),
so the agent's code never controls what holdout data it sees or how metrics are
computed — the anti-gaming guarantee is structural, not prompt-based.

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
from metis.benchmark.store import BenchmarkRecord, append_result
from metis.projects import load_project

# Harness-authored evaluation script injected into a subprocess.
# The agent's model.py provides only load_model / predict; it never sees
# the holdout path or the scoring logic.
#
# Lockbox hardening: the holdout directory is passed via the environment
# (METIS_HOLDOUT_DIR), the holdout arrays are loaded into memory, and THEN
# both sys.argv and the env var are scrubbed BEFORE the agent's model.py is
# imported/executed. This prevents a malicious model.py from reading the
# holdout path out of sys.argv or os.environ to access the sealed data
# directly. allow_pickle=False also blocks arbitrary code execution via
# crafted .npy files.
_EVAL_SCRIPT = r"""
import importlib.util, json, os, sys, pathlib
import numpy as np

variant_dir = pathlib.Path(sys.argv[1])
metric_name = sys.argv[2]
holdout_dir = pathlib.Path(os.environ["METIS_HOLDOUT_DIR"])

# Load holdout into memory before any agent code runs.
X = np.load(holdout_dir / "X.npy", allow_pickle=False)
y = np.load(holdout_dir / "y.npy", allow_pickle=False)

# Scrub the holdout location from process-visible state so the agent's
# model.py cannot recover it.
os.environ.pop("METIS_HOLDOUT_DIR", None)
del holdout_dir
sys.argv = [sys.argv[0]]

spec = importlib.util.spec_from_file_location("_variant_model", variant_dir / "model.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

model = mod.load_model(variant_dir / "weights")

preds = mod.predict(model, X)

mn = metric_name.lower()
if mn == "accuracy":
    score = float(np.mean(preds == y))
elif mn == "f1":
    from sklearn.metrics import f1_score
    score = float(f1_score(y, preds, average="weighted"))
elif mn == "auroc":
    from sklearn.metrics import roc_auc_score
    score = float(roc_auc_score(y, preds, multi_class="ovr"))
elif mn == "rmse":
    score = float(np.sqrt(np.mean((preds.astype(float) - y.astype(float)) ** 2)))
elif mn == "mae":
    score = float(np.mean(np.abs(preds.astype(float) - y.astype(float))))
else:
    score = float(np.mean(preds == y))

print(json.dumps({"task_metric_value": score}))
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
        task_metric_value, error = self._evaluate(variant_dir, holdout_dir, project.target_metric)

        record = BenchmarkRecord(
            variant_id=variant_id,
            task_metric_name=project.target_metric,
            task_metric_value=task_metric_value,
            param_count=eff.param_count,
            model_size_mb=eff.model_size_mb,
            latency_ms_p50=eff.latency_ms_p50,
            latency_ms_p95=eff.latency_ms_p95,
            throughput_sps=eff.throughput_sps,
            error=error,
        )
        append_result(benchmark_dir, record)
        return record

    def _evaluate(
        self, variant_dir: Path, holdout_dir: Path, metric_name: str
    ) -> tuple[float | None, str | None]:
        model_py = variant_dir / "model.py"
        holdout_X = holdout_dir / "X.npy"
        holdout_y = holdout_dir / "y.npy"

        if not model_py.exists():
            return None, "no model.py found in variant directory; train the model first"
        if not holdout_X.exists() or not holdout_y.exists():
            return None, "holdout not sealed yet; the harness must run seal_holdout first"

        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(_EVAL_SCRIPT)
            script_path = Path(f.name)

        # Pass the holdout location via the environment rather than argv; the
        # eval script consumes and scrubs it before importing the agent's code.
        env = {**os.environ, "METIS_HOLDOUT_DIR": str(holdout_dir)}
        try:
            result = subprocess.run(
                [sys.executable, str(script_path), str(variant_dir), metric_name],
                capture_output=True,
                text=True,
                timeout=300,
                env=env,
            )
        except subprocess.TimeoutExpired:
            return None, "evaluation timed out after 300 s"
        except Exception as exc:
            return None, f"evaluation subprocess error: {exc}"
        finally:
            script_path.unlink(missing_ok=True)

        if result.returncode != 0:
            stderr = result.stderr.strip()[:500]
            return None, f"evaluation script failed: {stderr}"

        try:
            data = json.loads(result.stdout.strip())
            return float(data["task_metric_value"]), None
        except Exception as exc:
            return None, f"could not parse evaluation output: {exc}"
