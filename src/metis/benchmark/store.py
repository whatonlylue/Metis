"""Append-only SQLite store for benchmark results (benchmark/results.db).

The database lives inside benchmark/, which is sealed from the agent.
Only the harness (runner, CLI) reads or writes here.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

DB_FILENAME = "results.db"

# Metrics where a lower value is better (everything else is higher-is-better).
_LOWER_IS_BETTER: frozenset[str] = frozenset({"rmse", "mae", "mse", "loss", "error_rate"})

_DDL = """
CREATE TABLE IF NOT EXISTS benchmark_runs (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    variant_id          TEXT    NOT NULL,
    timestamp           TEXT    NOT NULL,
    task_metric_name    TEXT    NOT NULL,
    task_metric_value   REAL,
    param_count         INTEGER,
    model_size_mb       REAL,
    latency_ms_p50      REAL,
    latency_ms_p95      REAL,
    throughput_sps      REAL,
    error               TEXT
);
"""


@dataclass
class BenchmarkRecord:
    variant_id: str
    task_metric_name: str
    task_metric_value: float | None
    param_count: int | None
    model_size_mb: float | None
    latency_ms_p50: float | None
    latency_ms_p95: float | None
    throughput_sps: float | None
    error: str | None = None
    timestamp: str = field(default="")

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


def _connect(benchmark_dir: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(benchmark_dir / DB_FILENAME)
    conn.row_factory = sqlite3.Row
    conn.execute(_DDL)
    conn.commit()
    return conn


def append_result(benchmark_dir: Path, record: BenchmarkRecord) -> int:
    """Insert *record* into results.db and return its rowid."""
    conn = _connect(benchmark_dir)
    try:
        cur = conn.execute(
            """INSERT INTO benchmark_runs
               (variant_id, timestamp, task_metric_name, task_metric_value,
                param_count, model_size_mb, latency_ms_p50, latency_ms_p95,
                throughput_sps, error)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (
                record.variant_id,
                record.timestamp,
                record.task_metric_name,
                record.task_metric_value,
                record.param_count,
                record.model_size_mb,
                record.latency_ms_p50,
                record.latency_ms_p95,
                record.throughput_sps,
                record.error,
            ),
        )
        conn.commit()
        rowid = cur.lastrowid
    finally:
        conn.close()
    return rowid  # type: ignore[return-value]


def get_leaderboard(
    benchmark_dir: Path,
    *,
    task_metric_name: str,
    n: int = 10,
) -> list[dict[str, object]]:
    """Return top-*n* rows ranked by task_metric_value (single-objective).

    Lower-is-better metrics (rmse, mae, …) sort ASC; others sort DESC.
    Rows with NULL task_metric_value are excluded.
    """
    order = "ASC" if task_metric_name.lower() in _LOWER_IS_BETTER else "DESC"
    conn = _connect(benchmark_dir)
    try:
        # The f-string interpolation of {order} is safe ONLY because `order` is
        # derived from an internal constant set ("ASC"/"DESC"), never user input.
        rows = conn.execute(
            f"""SELECT variant_id, timestamp, task_metric_name, task_metric_value,
                       param_count, model_size_mb, latency_ms_p50, latency_ms_p95,
                       throughput_sps, error
                FROM benchmark_runs
                WHERE task_metric_value IS NOT NULL
                  AND task_metric_name = ?
                ORDER BY task_metric_value {order}
                LIMIT ?""",
            (task_metric_name, n),
        ).fetchall()
    finally:
        conn.close()
    return [dict(r) for r in rows]
