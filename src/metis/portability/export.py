"""Export a trained variant to a portable artifact (ONNX, or pickle-bundle).

ONNX support is OPTIONAL: it is used only when both ``skl2onnx`` and ``onnx`` are
importable. Otherwise (or if conversion fails) we fall back to a self-contained
``.zip`` bundle that carries the pickled estimator plus its ``model.py`` inference
contract and ``recipe.yaml`` — loadable anywhere the same Python/sklearn is
available, with a clear recorded format. Every export records format, path and
SHA-256 checksum to an ``export.yaml`` manifest so it is reproducible/verifiable.

Lockbox note: export reads only the variant's own files (weights/code/recipe).
It never touches benchmark/.
"""

from __future__ import annotations

import hashlib
import pickle
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import yaml

EXPORT_MANIFEST = "export.yaml"


def onnx_available() -> bool:
    """True if both optional ONNX deps are importable."""
    try:
        import onnx  # noqa: F401
        import skl2onnx  # noqa: F401
    except Exception:
        return False
    return True


@dataclass
class ExportResult:
    """Record of one export: enough to locate and verify the artifact."""

    variant_id: str
    format: str  # "onnx" | "pickle_bundle"
    path: str  # absolute path to the artifact
    checksum: str  # sha256 hex of the artifact
    onnx_available: bool
    created_at: str


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _export_onnx(weights_pkl: Path, export_dir: Path) -> Path:
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType

    with weights_pkl.open("rb") as f:
        model = pickle.load(f)
    n_features = int(getattr(model, "n_features_in_", 0)) or 1
    initial_types = [("input", FloatTensorType([None, n_features]))]
    onx = convert_sklearn(model, initial_types=initial_types)
    out = export_dir / "model.onnx"
    out.write_bytes(onx.SerializeToString())
    return out


def _export_pickle_bundle(variant_dir: Path, export_dir: Path) -> Path:
    """Self-contained zip: pickled estimator + inference contract + recipe."""
    out = export_dir / "model_bundle.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(variant_dir / "weights" / "model.pkl", "model.pkl")
        for extra in ("model.py", "recipe.yaml"):
            src = variant_dir / extra
            if src.exists():
                z.write(src, extra)
        z.writestr(
            "README.txt",
            "Self-contained Metis model bundle.\n"
            "Load with: import pickle; pickle.load(open('model.pkl','rb'))\n"
            "model.py provides load_model(weights_dir)/predict(model, X).\n",
        )
    return out


def export_variant(
    project_root: Path,
    variant_id: str,
    *,
    prefer_onnx: bool = True,
    out_dir: Path | None = None,
) -> ExportResult:
    """Export *variant_id* to a portable artifact and record it.

    Tries ONNX when ``prefer_onnx`` and the optional deps are present; otherwise
    (or on conversion failure) writes a self-contained pickle bundle. Appends the
    export record to ``<export_dir>/export.yaml``.
    """
    variant_dir = project_root / "models" / variant_id
    weights_pkl = variant_dir / "weights" / "model.pkl"
    if not weights_pkl.exists():
        raise FileNotFoundError(
            f"no weights/model.pkl for variant {variant_id!r}; train it before exporting"
        )

    export_dir = out_dir or (variant_dir / "exports")
    export_dir.mkdir(parents=True, exist_ok=True)

    have_onnx = onnx_available()
    artifact: Path | None = None
    fmt = ""
    if prefer_onnx and have_onnx:
        try:
            artifact = _export_onnx(weights_pkl, export_dir)
            fmt = "onnx"
        except Exception:
            artifact = None  # graceful fallback below
    if artifact is None:
        artifact = _export_pickle_bundle(variant_dir, export_dir)
        fmt = "pickle_bundle"

    result = ExportResult(
        variant_id=variant_id,
        format=fmt,
        path=str(artifact.resolve()),
        checksum=_sha256(artifact),
        onnx_available=have_onnx,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    _record_export(export_dir, result)
    return result


def _record_export(export_dir: Path, result: ExportResult) -> None:
    """Append *result* to the export manifest (reproducibility record)."""
    manifest_path = export_dir / EXPORT_MANIFEST
    existing: list[dict[str, object]] = []
    if manifest_path.exists():
        try:
            loaded = yaml.safe_load(manifest_path.read_text()) or {}
            existing = list(loaded.get("exports", []))
        except Exception:
            existing = []
    existing.append(asdict(result))
    manifest_path.write_text(yaml.safe_dump({"exports": existing}, sort_keys=False))
