"""Harness-side robustness corruptions.

These perturbations are applied by the trusted harness to the holdout *feature*
array only — never by the agent, and never to anything the agent can observe.
The runner (``BenchmarkRunner.run_robustness``) reads the sealed holdout, applies
a corruption here, hands only the corrupted features to the sandboxed predict
worker, and scores the returned predictions against the labels it keeps. The
agent therefore never sees the holdout, the corruptions, or the per-corruption
inputs — the lockbox is preserved.

Supported corruptions (all operate on a float feature matrix ``X``):
  * ``gaussian_noise``  — add N(0, severity·std(X)) noise.
  * ``feature_dropout`` — zero each feature independently with probability severity.
  * ``scaling``         — multiply all features by (1 + severity).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

    from metis.projects.schema import RobustnessConfig


@dataclass(frozen=True)
class Corruption:
    """A named perturbation with a single severity knob."""

    name: str
    severity: float


def default_corruptions() -> list[Corruption]:
    """The default corruption suite when a project declares none explicitly."""
    return [
        Corruption("gaussian_noise", 0.5),
        Corruption("feature_dropout", 0.2),
        Corruption("scaling", 0.2),
    ]


def corruptions_from_config(config: RobustnessConfig | None) -> list[Corruption]:
    """Build the corruption list from a project's ``RobustnessConfig``."""
    if config is None or not config.corruptions:
        return default_corruptions()
    return [Corruption(c.name, c.severity) for c in config.corruptions]


def apply_corruption(
    X: np.ndarray,
    corruption: Corruption,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return a corrupted copy of *X*. Pure function of (X, corruption, rng)."""
    import numpy as np

    Xc = np.asarray(X, dtype=float).copy()
    name = corruption.name
    sev = float(corruption.severity)
    if name == "gaussian_noise":
        std = float(np.std(Xc)) or 1.0
        return Xc + rng.normal(0.0, sev * std, size=Xc.shape)
    if name == "feature_dropout":
        mask = rng.random(Xc.shape) < sev
        Xc[mask] = 0.0
        return Xc
    if name == "scaling":
        return Xc * (1.0 + sev)
    raise ValueError(f"unknown corruption: {name!r}")
