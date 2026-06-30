"""Ingest: de-dup, validate, and auto train/val/test split a sourced dataset.

This is the DATA step's processing half. It takes raw numpy arrays (as written by
a provider into ``data/raw/<dataset>/``), removes duplicate and corrupt samples,
runs validation checks, and produces a reproducible train/val/test split.

The **test split is handed to the harness sealer** (:func:`metis.benchmark.seal_holdout`)
so it lands in the lockboxed ``benchmark/holdout/`` the agent can never read. The
agent only ever gets the train and val splits in ``data/processed/``. This keeps
the anti-gaming guarantee intact: the harness splits and hides the test set; the
agent never receives it.

numpy is required (lazy-imported); scikit-learn is not.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from metis.benchmark.sealer import seal_holdout

if TYPE_CHECKING:
    import numpy as np


@dataclass(frozen=True)
class ValidationReport:
    """Result of validating a dataset before splitting."""

    n_samples: int
    n_features: int
    n_classes: int
    class_balance: dict[int, int]
    n_duplicates_removed: int
    n_corrupt_removed: int
    warnings: list[str] = field(default_factory=list)
    ok: bool = True

    def summary(self) -> str:
        lines = [
            f"samples={self.n_samples}, features={self.n_features}, classes={self.n_classes}",
            f"class_balance={self.class_balance}",
            f"duplicates_removed={self.n_duplicates_removed}, "
            f"corrupt_removed={self.n_corrupt_removed}",
        ]
        lines.extend(f"warning: {w}" for w in self.warnings)
        return "\n".join(lines)


@dataclass(frozen=True)
class SplitResult:
    """Sizes of the produced splits + where the sealed test set landed."""

    train_size: int
    val_size: int
    test_size: int
    seed: int
    holdout_dir: Path
    report: ValidationReport


class ValidationError(ValueError):
    """Raised when a dataset fails validation (empty, label mismatch, etc.)."""


def load_raw_arrays(dataset_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load ``X.npy``/``y.npy`` from a fetched dataset directory."""
    import numpy as np

    x_path = dataset_dir / "X.npy"
    y_path = dataset_dir / "y.npy"
    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError(f"Expected X.npy and y.npy in {dataset_dir}")
    X = np.load(x_path, allow_pickle=False)
    y = np.load(y_path, allow_pickle=False)
    return X, y


def dedup(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Drop duplicate samples by content hash (X-row + label), keeping first seen.

    Returns ``(X, y, n_removed)``.
    """
    import numpy as np

    seen: set[bytes] = set()
    keep: list[int] = []
    X2 = np.ascontiguousarray(X)
    for i in range(len(X2)):
        key = X2[i].tobytes() + b"|" + np.asarray(y[i]).tobytes()
        if key in seen:
            continue
        seen.add(key)
        keep.append(i)
    n_removed = len(X) - len(keep)
    idx = np.asarray(keep, dtype=int)
    return X[idx], y[idx], n_removed


def _flatten_features(X: np.ndarray) -> int:
    import numpy as np

    if X.ndim <= 1:
        return 1
    return int(np.prod(X.shape[1:]))


def validate(
    X: np.ndarray,
    y: np.ndarray,
    *,
    classes: list[str] | None = None,
    min_per_class: int = 1,
) -> tuple[np.ndarray, np.ndarray, ValidationReport]:
    """Validate + clean a dataset, returning cleaned arrays and a report.

    Checks performed:

    * non-empty and X/y length agreement (hard errors);
    * corrupt-sample detection — rows with NaN/inf features, or NaN labels, are
      dropped and counted;
    * label-coverage / class-balance reporting, with a warning when a class has
      fewer than ``min_per_class`` samples;
    * when ``classes`` is given, a warning if the observed class count differs.
    """
    import numpy as np

    if len(X) == 0:
        raise ValidationError("dataset is empty")
    if len(X) != len(y):
        raise ValidationError(f"X/y length mismatch: {len(X)} vs {len(y)}")

    warnings: list[str] = []

    # Corrupt-sample detection: non-finite features or labels.
    X_num = X.reshape(len(X), -1) if X.ndim > 1 else X.reshape(len(X), 1)
    finite_feat = np.isfinite(X_num.astype(np.float64, copy=False)).all(axis=1)
    y_float = _coerce_label_finiteness(y)
    if y_float is not None:
        y_finite_mask = np.isfinite(y_float)
        # Multi-label y is 2-D; reduce to per-sample bool so the shapes match finite_feat.
        if y_finite_mask.ndim > 1:
            y_finite_mask = y_finite_mask.all(axis=1)
        finite_lbl = y_finite_mask
    else:
        finite_lbl = np.ones(len(y), dtype=bool)
    good = finite_feat & finite_lbl
    n_corrupt = int((~good).sum())
    if n_corrupt:
        warnings.append(f"removed {n_corrupt} corrupt sample(s) with NaN/inf values")
    X, y = X[good], y[good]

    if len(X) == 0:
        raise ValidationError("all samples were corrupt; nothing left after cleaning")

    is_multilabel = y.ndim > 1
    if is_multilabel:
        # Multi-label: report per-column positive rates instead of class label counts.
        n_classes = int(y.shape[1])
        balance: dict[int, int] = {}
        if classes is not None and len(classes) != n_classes:
            warnings.append(
                f"declared {len(classes)} classes but y has {n_classes} columns"
            )
    else:
        labels, counts = np.unique(y, return_counts=True)
        balance = {int(lbl): int(c) for lbl, c in zip(labels.tolist(), counts.tolist())}
        n_classes = int(len(labels))
        for lbl, c in balance.items():
            if c < min_per_class:
                warnings.append(f"class {lbl} has only {c} sample(s) (< {min_per_class})")
        if classes is not None and len(classes) != n_classes:
            warnings.append(f"declared {len(classes)} classes but found {n_classes} in the data")

    report = ValidationReport(
        n_samples=int(len(X)),
        n_features=_flatten_features(X),
        n_classes=n_classes,
        class_balance=balance,
        n_duplicates_removed=0,  # filled in by ingest_arrays
        n_corrupt_removed=n_corrupt,
        warnings=warnings,
        ok=True,
    )
    return X, y, report


def _coerce_label_finiteness(y: np.ndarray) -> np.ndarray | None:
    """Return labels as float for finiteness checks, or None for non-numeric labels."""
    import numpy as np

    if np.issubdtype(y.dtype, np.number):
        return y.astype(np.float64, copy=False)
    return None


def split_and_seal(
    project_root: Path,
    X: np.ndarray,
    y: np.ndarray,
    *,
    train: float,
    val: float,
    test: float,
    seed: int,
    report: ValidationReport,
) -> SplitResult:
    """Split into train/val/test, write train/val to ``data/processed/``, seal test.

    The test fraction is handed to :func:`seal_holdout` (numpy mode), which removes
    it from ``data/processed`` and locks it in ``benchmark/holdout/``. The remainder
    is then split into train (``X.npy``/``y.npy``) and val (``X_val.npy``/``y_val.npy``)
    with the same ``seed`` for reproducibility.
    """
    import numpy as np

    _check_ratios(train, val, test)
    n = len(X)
    if n < 3:
        raise ValidationError(f"need at least 3 samples to split, got {n}")

    processed = project_root / "data" / "processed"
    processed.mkdir(parents=True, exist_ok=True)

    # 1) Write the full (cleaned) dataset, then let the harness seal the test split.
    np.save(processed / "X.npy", X)
    np.save(processed / "y.npy", y)
    holdout_dir = seal_holdout(project_root, fraction=test, seed=seed, mode="numpy")
    test_size = int(len(np.load(holdout_dir / "X.npy", allow_pickle=False)))

    # 2) data/processed now holds train+val only. Split it reproducibly.
    X_rem = np.load(processed / "X.npy", allow_pickle=False)
    y_rem = np.load(processed / "y.npy", allow_pickle=False)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(X_rem))
    # val proportion *within the remainder* preserves the requested global ratio.
    val_frac_rem = val / (train + val) if (train + val) > 0 else 0.0
    n_val = min(len(X_rem) - 1, max(1, round(len(X_rem) * val_frac_rem))) if val > 0 else 0
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    np.save(processed / "X.npy", X_rem[train_idx])
    np.save(processed / "y.npy", y_rem[train_idx])
    if n_val:
        np.save(processed / "X_val.npy", X_rem[val_idx])
        np.save(processed / "y_val.npy", y_rem[val_idx])

    return SplitResult(
        train_size=int(len(train_idx)),
        val_size=int(n_val),
        test_size=test_size,
        seed=seed,
        holdout_dir=holdout_dir,
        report=report,
    )


def ingest_arrays(
    project_root: Path,
    X: np.ndarray,
    y: np.ndarray,
    *,
    train: float = 0.7,
    val: float = 0.15,
    test: float = 0.15,
    seed: int = 42,
    classes: list[str] | None = None,
    do_dedup: bool = True,
) -> SplitResult:
    """Full DATA processing on in-memory arrays: dedup -> validate -> split -> seal."""
    n_dups = 0
    if do_dedup:
        X, y, n_dups = dedup(X, y)
    X, y, report = validate(X, y, classes=classes)
    report = ValidationReport(
        n_samples=report.n_samples,
        n_features=report.n_features,
        n_classes=report.n_classes,
        class_balance=report.class_balance,
        n_duplicates_removed=n_dups,
        n_corrupt_removed=report.n_corrupt_removed,
        warnings=report.warnings,
        ok=report.ok,
    )
    return split_and_seal(
        project_root, X, y, train=train, val=val, test=test, seed=seed, report=report
    )


def ingest_dataset(
    project_root: Path,
    dataset_dir: Path,
    *,
    train: float = 0.7,
    val: float = 0.15,
    test: float = 0.15,
    seed: int = 42,
    classes: list[str] | None = None,
    do_dedup: bool = True,
) -> SplitResult:
    """Load ``X.npy``/``y.npy`` from ``dataset_dir`` then run :func:`ingest_arrays`."""
    X, y = load_raw_arrays(dataset_dir)
    return ingest_arrays(
        project_root,
        X,
        y,
        train=train,
        val=val,
        test=test,
        seed=seed,
        classes=classes,
        do_dedup=do_dedup,
    )


def _check_ratios(train: float, val: float, test: float) -> None:
    for name, v in (("train", train), ("val", val), ("test", test)):
        if v < 0:
            raise ValueError(f"{name} ratio must be >= 0, got {v}")
    total = train + val + test
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"split ratios must sum to 1.0, got {total} ({train}/{val}/{test})")
    if not 0 < test < 1:
        raise ValueError(f"test ratio must be in (0, 1) to seal a holdout, got {test}")
