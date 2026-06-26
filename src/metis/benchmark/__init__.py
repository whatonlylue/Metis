"""Harness-side benchmark engine.

Public API:
    BenchmarkRunner   — evaluate a model variant on the sealed holdout.
    seal_holdout      — split processed data and lock the test set away.
    BenchmarkRecord   — dataclass for a single benchmark run.
    append_result     — write a record to results.db.
    get_leaderboard   — read ranked results from results.db.
"""

from metis.benchmark.runner import BenchmarkRunner
from metis.benchmark.sealer import seal_holdout
from metis.benchmark.store import BenchmarkRecord, append_result, get_leaderboard

__all__ = [
    "BenchmarkRunner",
    "seal_holdout",
    "BenchmarkRecord",
    "append_result",
    "get_leaderboard",
]
