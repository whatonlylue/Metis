"""Reproducibility bundle: everything needed to reconstruct/verify a variant.

A bundle captures (CLAUDE.md reproducibility principle):
  * ``recipe.yaml``        — architecture + hyperparameters.
  * ``code/train.py`` + ``code/model.py`` — the training + inference code.
  * ``weights/``           — the serialized model artifact (when present).
  * ``environment.yaml``   — python version, platform, key package versions.
  * ``data_manifest.yaml`` — checksums of the (train) data snapshot + the
                             dataset provenance recorded in project.yaml. A
                             *manifest*, not the raw bytes.
  * ``benchmark_results.yaml`` — the variant's recorded benchmark + robustness
                             result VALUES extracted from results.db.

LOCKBOX: the bundle is assembled by the harness but must never include sealed
holdout data or the scoring code. We only ever read the variant's own files plus
the *recorded result values* for that variant — we never copy benchmark/ (no
holdout arrays, no suite.py, no results.db file). A test asserts the exclusion.
"""

from __future__ import annotations

import hashlib
import platform
import shutil
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import yaml

from metis.benchmark.store import DB_FILENAME
from metis.projects import load_project

_ENV_PACKAGES = (
    "numpy",
    "scikit-learn",
    "scipy",
    "anthropic",
    "pydantic",
    "pyyaml",
    "textual",
    "onnx",
    "skl2onnx",
)


@dataclass
class BundleResult:
    """Where the assembled bundle lives + its top-level manifest."""

    variant_id: str
    directory: Path
    archive: Path
    manifest: dict[str, object]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _capture_environment() -> dict[str, object]:
    import importlib.metadata as im

    packages: dict[str, str] = {}
    for name in _ENV_PACKAGES:
        try:
            packages[name] = im.version(name)
        except Exception:
            continue  # not installed — simply omit
    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "packages": packages,
    }


def _data_manifest(project_root: Path) -> dict[str, object]:
    """Checksums of the train-data snapshot + recorded provenance (no raw bytes)."""
    processed = project_root / "data" / "processed"
    files: list[dict[str, object]] = []
    if processed.is_dir():
        for f in sorted(processed.rglob("*")):
            if f.is_file():
                files.append(
                    {
                        "path": str(f.relative_to(project_root)),
                        "sha256": _sha256(f),
                        "bytes": f.stat().st_size,
                    }
                )
    sources: list[dict[str, object]] = []
    try:
        spec = load_project(project_root)
        sources = [s.model_dump(mode="json") for s in spec.data.sources]
    except Exception:
        sources = []
    return {"processed_files": files, "sources": sources}


def _variant_results(project_root: Path, variant_id: str) -> dict[str, object]:
    """Extract this variant's recorded result VALUES from results.db.

    Reads only the rows for *variant_id*; the holdout, suite.py and the db file
    itself are never copied into the bundle.
    """
    db_path = project_root / "benchmark" / DB_FILENAME
    out: dict[str, object] = {"benchmark_runs": [], "robustness_runs": []}
    if not db_path.exists():
        return out
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        bench = conn.execute(
            "SELECT * FROM benchmark_runs WHERE variant_id = ? ORDER BY id ASC", (variant_id,)
        ).fetchall()
        out["benchmark_runs"] = [dict(r) for r in bench]
        try:
            rob = conn.execute(
                "SELECT * FROM robustness_runs WHERE variant_id = ? ORDER BY id ASC",
                (variant_id,),
            ).fetchall()
            out["robustness_runs"] = [dict(r) for r in rob]
        except sqlite3.OperationalError:
            pass  # robustness table absent on older DBs
    finally:
        conn.close()
    return out


def build_repro_bundle(
    project_root: Path,
    variant_id: str,
    *,
    out_dir: Path | None = None,
) -> BundleResult:
    """Assemble a self-contained reproducibility bundle (directory + zip archive)."""
    variant_dir = project_root / "models" / variant_id
    if not variant_dir.is_dir():
        raise FileNotFoundError(f"variant directory not found: models/{variant_id}")

    base = out_dir or (project_root / "bundles")
    base.mkdir(parents=True, exist_ok=True)
    staging = base / variant_id
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    contents: list[str] = []

    # recipe.yaml
    recipe_src = variant_dir / "recipe.yaml"
    if recipe_src.exists():
        shutil.copy2(recipe_src, staging / "recipe.yaml")
        contents.append("recipe.yaml")

    # code/
    code_dir = staging / "code"
    code_dir.mkdir()
    for fname in ("train.py", "model.py"):
        src = variant_dir / fname
        if src.exists():
            shutil.copy2(src, code_dir / fname)
            contents.append(f"code/{fname}")

    # weights/ (the model artifact itself, when present — not holdout)
    weights_src = variant_dir / "weights"
    if weights_src.is_dir() and any(weights_src.iterdir()):
        shutil.copytree(weights_src, staging / "weights")
        contents.append("weights/")

    # environment.yaml
    (staging / "environment.yaml").write_text(
        yaml.safe_dump(_capture_environment(), sort_keys=False)
    )
    contents.append("environment.yaml")

    # data_manifest.yaml
    (staging / "data_manifest.yaml").write_text(
        yaml.safe_dump(_data_manifest(project_root), sort_keys=False)
    )
    contents.append("data_manifest.yaml")

    # benchmark_results.yaml (recorded values only)
    (staging / "benchmark_results.yaml").write_text(
        yaml.safe_dump(_variant_results(project_root, variant_id), sort_keys=False)
    )
    contents.append("benchmark_results.yaml")

    manifest: dict[str, object] = {
        "variant_id": variant_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "project": str(project_root.name),
        "contents": contents,
        "excludes": ["benchmark/holdout (sealed)", "benchmark/ scoring code", "results.db file"],
    }
    (staging / "manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))

    archive = shutil.make_archive(str(staging), "zip", root_dir=staging)
    return BundleResult(
        variant_id=variant_id,
        directory=staging,
        archive=Path(archive),
        manifest=manifest,
    )
