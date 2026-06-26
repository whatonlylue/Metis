"""BRANCH: generate new candidate variants when the leaderboard plateaus.

Two complementary strategies (CLAUDE.md evolutionary-search principle #3):
  1. Mutate the top performers — perturb a hyperparameter of an existing strong
     family to a neighbouring value from its grid.
  2. Introduce new families — propose families not yet tried at all.

This module is part of the training layer: it produces concrete ``Candidate``
specs compatible with the existing PROPOSE -> TRAIN pipeline. Plateau *detection*
is harness-side (``metis.benchmark.plateau``); this is the response to it.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

from metis.training.toy import FAMILIES, Candidate, FamilySpec, build_candidate


def _family_of(variant_id: str, families: dict[str, FamilySpec]) -> FamilySpec | None:
    """Resolve which family a variant id belongs to (by key prefix)."""
    for spec in families.values():
        if variant_id == spec.key or variant_id.startswith(spec.key + "_"):
            return spec
    return None


def _unique_id(base: str, taken: set[str]) -> str:
    """Return ``base`` (or ``base_2``, ``base_3``, …) not already in ``taken``."""
    candidate = base
    suffix = 2
    while candidate in taken:
        candidate = f"{base}_{suffix}"
        suffix += 1
    taken.add(candidate)
    return candidate


def _mutate_hparams(
    spec: FamilySpec,
    rng: random.Random,
) -> dict[str, object] | None:
    """Perturb one hyperparameter of *spec* to a different grid value.

    Returns the mutated hyperparameter dict, or None if no perturbation is
    possible (every grid axis has a single value).
    """
    axes = [name for name, values in spec.hparam_grid.items() if len(values) > 1]
    if not axes:
        return None
    name = rng.choice(axes)
    current = spec.default_hparams.get(name)
    alternatives = [v for v in spec.hparam_grid[name] if v != current]
    if not alternatives:
        return None
    hparams = dict(spec.default_hparams)
    hparams[name] = rng.choice(alternatives)
    return hparams


def branch_candidates(
    ranked_variant_ids: Sequence[str],
    *,
    existing_ids: Sequence[str] | None = None,
    families: dict[str, FamilySpec] | None = None,
    max_mutations: int = 2,
    max_new_families: int = 1,
    seed: int = 0,
) -> list[Candidate]:
    """Generate new candidates by mutating top performers and adding new families.

    Args:
        ranked_variant_ids: leaderboard variant ids, best first.
        existing_ids: every variant id already used (defaults to ``ranked_variant_ids``);
            generated ids are guaranteed not to collide with these.
        families: family registry (defaults to the toy ``FAMILIES``).
        max_mutations: cap on hyperparameter-perturbation candidates.
        max_new_families: cap on brand-new families to introduce.
        seed: RNG seed for reproducible branching.
    """
    families = families or FAMILIES
    rng = random.Random(seed)
    taken = set(existing_ids) if existing_ids is not None else set(ranked_variant_ids)
    new_candidates: list[Candidate] = []

    # 1) Mutate the top performers (best-first), one mutant each until the cap.
    for variant_id in ranked_variant_ids:
        if len(new_candidates) >= max_mutations:
            break
        spec = _family_of(variant_id, families)
        if spec is None:
            continue
        hparams = _mutate_hparams(spec, rng)
        if hparams is None:
            continue
        vid = _unique_id(f"{spec.key}_m", taken)
        new_candidates.append(build_candidate(spec, hparams, vid))

    # 2) Introduce families not present anywhere in the search so far.
    tried_keys = {spec.key for vid in taken if (spec := _family_of(vid, families)) is not None}
    untried = [spec for key, spec in families.items() if key not in tried_keys]
    for spec in untried[:max_new_families]:
        vid = _unique_id(spec.key, taken)
        new_candidates.append(build_candidate(spec, dict(spec.default_hparams), vid))

    return new_candidates
