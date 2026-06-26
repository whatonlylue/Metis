"""Training helpers: the PROPOSE -> TRAIN path on a light, runnable toy dataset.

The agent normally writes ``train.py``/``model.py`` itself; this module provides
a fully runnable reference implementation (numpy + scikit-learn, no torch) so the
end-to-end PROPOSE -> TRAIN -> BENCHMARK loop can be exercised and tested cheaply.
"""

from metis.training.toy import (
    Candidate,
    prepare_toy_dataset,
    propose_candidates,
    run_toy_pipeline,
    scaffold_candidate,
    train_candidate,
)

__all__ = [
    "Candidate",
    "prepare_toy_dataset",
    "propose_candidates",
    "run_toy_pipeline",
    "scaffold_candidate",
    "train_candidate",
]
