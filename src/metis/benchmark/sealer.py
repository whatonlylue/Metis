"""Holdout sealing: the harness splits processed data into benchmark/holdout/.

The agent never calls this — only the harness CLI / project setup flow does.
Once sealed, the lockbox blocks the agent from reading benchmark/holdout/.

Two modes are supported:
  "imagenet" — directory tree of class/sample files (e.g. data/processed/cat/img1.jpg).
  "numpy"    — flat X.npy / y.npy arrays (requires numpy; see optional [ml] deps).

In both cases a seal.yaml manifest is written to holdout/ recording provenance.
"""

from __future__ import annotations

import random
import shutil
from pathlib import Path
from typing import Literal

import yaml

SealMode = Literal["auto", "imagenet", "numpy"]


def detect_seal_mode(source_dir: Path) -> SealMode:
    """Infer the seal mode from what the human/agent actually put in ``source_dir``.

    ``numpy`` when flat ``X.npy``/``y.npy`` arrays are present; ``imagenet`` when the
    directory holds class sub-folders. Raises if neither shape is found, so callers
    can tell "nothing to seal yet" apart from a real layout.
    """
    if (source_dir / "X.npy").exists() and (source_dir / "y.npy").exists():
        return "numpy"
    if source_dir.is_dir() and any(p.is_dir() for p in source_dir.iterdir()):
        return "imagenet"
    raise FileNotFoundError(
        f"No sealable data in {source_dir} (expected X.npy/y.npy or class subdirectories)."
    )


def ensure_holdout_sealed(
    project_root: Path,
    *,
    source_dir: Path | None = None,
) -> Path | None:
    """Idempotently seal a holdout the moment processed data exists — however it got there.

    This is the harness's anti-gaming guarantee, decoupled from ``ingest_dataset``:
    whether the human ran ingest on raw data OR dropped pre-processed ``X.npy``/``y.npy``
    straight into ``data/processed/``, the harness still carves and locks a test holdout
    *before* the agent can train on it. The seal split/seed follow the project's
    ``project.yaml`` when present, else sensible defaults.

    Returns the holdout dir if it is (now) sealed, or ``None`` if there's nothing to
    seal yet (no processed data) — never raises on the "no data" case, so it's safe to
    call as a pre-training guard.
    """
    if source_dir is None:
        source_dir = project_root / "data" / "processed"
    holdout_dir = project_root / "benchmark" / "holdout"
    if (holdout_dir / "seal.yaml").exists():
        return holdout_dir  # already sealed — no-op
    if not source_dir.exists():
        return None
    try:
        mode = detect_seal_mode(source_dir)
    except FileNotFoundError:
        return None  # processed/ exists but isn't populated yet

    fraction, seed = 0.2, 42
    try:
        from metis.projects import load_project

        spec = load_project(project_root)
        fraction = spec.data.split.test
        seed = spec.data.split_seed
    except Exception:
        pass  # no/invalid project.yaml — fall back to defaults
    return seal_holdout(project_root, source_dir=source_dir, fraction=fraction, seed=seed, mode=mode)


def seal_holdout(
    project_root: Path,
    *,
    source_dir: Path | None = None,
    fraction: float = 0.2,
    seed: int = 42,
    mode: SealMode = "auto",
) -> Path:
    """Copy a random fraction of processed data into benchmark/holdout/.

    Args:
        project_root: Root of the project (contains data/, benchmark/, …).
        source_dir:   Where to draw samples from; defaults to data/processed/.
        fraction:     Fraction of samples to hold out (0 < fraction < 1).
        seed:         RNG seed for reproducibility.
        mode:         "imagenet" (class subdirectories) or "numpy" (X.npy/y.npy).

    Returns:
        Path to the holdout directory.
    """
    if not 0 < fraction < 1:
        raise ValueError(f"fraction must be in (0, 1), got {fraction}")

    if source_dir is None:
        source_dir = project_root / "data" / "processed"

    if mode == "auto":
        mode = detect_seal_mode(source_dir)

    holdout_dir = project_root / "benchmark" / "holdout"
    # Guard against double-sealing: re-splitting would corrupt the train/holdout
    # boundary and overwrite an existing sealed set.
    if (holdout_dir / "seal.yaml").exists():
        raise FileExistsError(
            f"Holdout already sealed (found {holdout_dir / 'seal.yaml'}); refusing to re-seal."
        )
    holdout_dir.mkdir(parents=True, exist_ok=True)

    if mode == "imagenet":
        _seal_imagenet(source_dir, holdout_dir, fraction=fraction, seed=seed)
    elif mode == "numpy":
        _seal_numpy(source_dir, holdout_dir, fraction=fraction, seed=seed)
    else:
        raise ValueError(f"Unknown seal mode: {mode!r}")

    manifest = {
        "source_dir": str(source_dir),
        "fraction": fraction,
        "seed": seed,
        "mode": mode,
    }
    (holdout_dir / "seal.yaml").write_text(yaml.safe_dump(manifest))
    return holdout_dir


def _seal_imagenet(source_dir: Path, holdout_dir: Path, *, fraction: float, seed: int) -> None:
    """Split an imagenet-style <class>/<sample> directory tree."""
    rng = random.Random(seed)
    class_dirs = sorted(p for p in source_dir.iterdir() if p.is_dir())
    if not class_dirs:
        raise FileNotFoundError(f"No class subdirectories found in {source_dir}")

    for class_dir in class_dirs:
        samples = sorted(class_dir.iterdir())
        k = max(1, round(len(samples) * fraction))
        chosen = rng.sample(samples, min(k, len(samples)))
        dest_class = holdout_dir / class_dir.name
        dest_class.mkdir(exist_ok=True)
        for src in chosen:
            shutil.copy2(src, dest_class / src.name)
            # Remove the holdout sample from the source/training dir so it can
            # never leak into training data (mirrors the numpy split below).
            src.unlink()


def _seal_numpy(source_dir: Path, holdout_dir: Path, *, fraction: float, seed: int) -> None:
    """Split numpy X.npy / y.npy arrays and write holdout arrays to holdout_dir."""
    try:
        import numpy as np
    except ImportError as exc:
        raise ImportError(
            "numpy is required for 'numpy' seal mode; install with: pip install metis[ml]"
        ) from exc

    X_path = source_dir / "X.npy"
    y_path = source_dir / "y.npy"
    if not X_path.exists() or not y_path.exists():
        raise FileNotFoundError(f"Expected X.npy and y.npy in {source_dir}")

    X = np.load(X_path, allow_pickle=True)
    y = np.load(y_path, allow_pickle=True)
    n = len(X)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    k = max(1, round(n * fraction))

    holdout_idx = indices[:k]
    train_idx = indices[k:]

    np.save(holdout_dir / "X.npy", X[holdout_idx])
    np.save(holdout_dir / "y.npy", y[holdout_idx])
    # Overwrite source with training-only portion.
    np.save(X_path, X[train_idx])
    np.save(y_path, y[train_idx])
