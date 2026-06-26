"""Training helpers: the PROPOSE -> TRAIN path on a light, runnable toy dataset.

The agent normally writes ``train.py``/``model.py`` itself; this module provides
a fully runnable reference implementation (numpy + scikit-learn, no torch) so the
end-to-end PROPOSE -> TRAIN -> BENCHMARK loop can be exercised and tested cheaply.
``evolve.branch_candidates`` extends this into BRANCH for the evolutionary search.
"""

from metis.training.evolve import branch_candidates
from metis.training.toy import (
    FAMILIES,
    Candidate,
    FamilySpec,
    build_candidate,
    prepare_toy_dataset,
    propose_candidates,
    run_toy_pipeline,
    scaffold_candidate,
    train_candidate,
)

__all__ = [
    "Candidate",
    "FamilySpec",
    "FAMILIES",
    "build_candidate",
    "branch_candidates",
    "prepare_toy_dataset",
    "propose_candidates",
    "run_toy_pipeline",
    "scaffold_candidate",
    "train_candidate",
]
