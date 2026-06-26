"""Harness-side efficiency metric measurements.

These are derived purely from the filesystem / recipe.yaml — no model loading
required. Live inference metrics (latency, throughput) are measured by the
runner when it evaluates a model; the dataclass carries None until then.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class EfficiencyMetrics:
    param_count: int | None
    model_size_mb: float | None
    latency_ms_p50: float | None
    latency_ms_p95: float | None
    throughput_sps: float | None


def measure_model_size(weights_dir: Path) -> float | None:
    """Total size in MB of all files under *weights_dir*."""
    if not weights_dir.is_dir():
        return None
    total_bytes = sum(f.stat().st_size for f in weights_dir.rglob("*") if f.is_file())
    return round(total_bytes / (1024**2), 4) if total_bytes else None


def read_param_count(recipe_path: Path) -> int | None:
    """Read the agent's self-reported ``param_count`` from ``recipe.yaml``.

    This value is NOT trusted as the recorded metric: the benchmark runner
    measures param_count by introspecting the actual loaded model object inside
    the sandbox (see ``runner._EVAL_SCRIPT``). This recipe value is only used as
    a fallback when structural introspection of the estimator is impossible.
    """
    if not recipe_path.exists():
        return None
    try:
        data = yaml.safe_load(recipe_path.read_text()) or {}
        val = data.get("param_count")
        return int(val) if val is not None else None
    except Exception:
        return None


def collect_efficiency_metrics(variant_dir: Path) -> EfficiencyMetrics:
    """Gather filesystem-derivable efficiency metrics for a model variant.

    ``param_count`` here is the agent's self-reported recipe value, kept only as
    a fallback; the runner overrides it with the value measured from the loaded
    model. ``model_size_mb`` is measured directly from the serialized weights.
    """
    return EfficiencyMetrics(
        param_count=read_param_count(variant_dir / "recipe.yaml"),
        model_size_mb=measure_model_size(variant_dir / "weights"),
        # Latency and throughput require live inference; the BenchmarkRunner
        # measures them in the eval subprocess and merges them into the record.
        latency_ms_p50=None,
        latency_ms_p95=None,
        throughput_sps=None,
    )
