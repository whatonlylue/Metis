"""Harness-side portability: model export + reproducibility bundles.

These operations are run by the harness/human, not the agent. They read a
trained variant's recipe, code, weights, environment and recorded benchmark
results — but never the sealed holdout or the scoring code — so a model can be
exported to a portable format (ONNX or a self-contained pickle bundle) and a
full reproducibility bundle can be assembled and verified offline.
"""

from __future__ import annotations

from metis.portability.bundle import BundleResult, build_repro_bundle
from metis.portability.export import ExportResult, export_variant, onnx_available

__all__ = [
    "ExportResult",
    "export_variant",
    "onnx_available",
    "BundleResult",
    "build_repro_bundle",
]
