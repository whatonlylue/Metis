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


def is_lower_better(metric_name: str) -> bool:
    """True if a lower value of *metric_name* is better (rmse, mae, …)."""
    return metric_name.lower() in _LOWER_IS_BETTER


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
    error               TEXT,
    pruned              INTEGER NOT NULL DEFAULT 0,
    pruned_reason       TEXT
);

CREATE TABLE IF NOT EXISTS resource_usage (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp     TEXT    NOT NULL,
    kind          TEXT    NOT NULL,
    variant_id    TEXT,
    wall_clock_s  REAL    NOT NULL DEFAULT 0,
    detail        TEXT
);
"""

# Columns added after the initial schema; applied to pre-existing DBs on connect
# so older results.db files keep working (append-only / backward compatible).
_MIGRATIONS: tuple[tuple[str, str], ...] = (
    ("pruned", "ALTER TABLE benchmark_runs ADD COLUMN pruned INTEGER NOT NULL DEFAULT 0"),
    ("pruned_reason", "ALTER TABLE benchmark_runs ADD COLUMN pruned_reason TEXT"),
)


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
    conn.executescript(_DDL)
    _migrate(conn)
    conn.commit()
    return conn


def _migrate(conn: sqlite3.Connection) -> None:
    """Add columns introduced after the initial schema to pre-existing DBs."""
    existing = {row["name"] for row in conn.execute("PRAGMA table_info(benchmark_runs)")}
    for column, ddl in _MIGRATIONS:
        if column not in existing:
            conn.execute(ddl)


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
    include_pruned: bool = False,
) -> list[dict[str, object]]:
    """Return top-*n* rows ranked by task_metric_value (single-objective).

    Lower-is-better metrics (rmse, mae, …) sort ASC; others sort DESC.
    Rows with NULL task_metric_value are excluded. Pruned variants are excluded
    unless ``include_pruned`` is True (they still carry their ``pruned`` flag).
    """
    order = "ASC" if task_metric_name.lower() in _LOWER_IS_BETTER else "DESC"
    prune_clause = "" if include_pruned else "AND pruned = 0"
    conn = _connect(benchmark_dir)
    try:
        # The f-string interpolation of {order}/{prune_clause} is safe ONLY because
        # both are derived from internal constants, never from user input.
        rows = conn.execute(
            f"""SELECT variant_id, timestamp, task_metric_name, task_metric_value,
                       param_count, model_size_mb, latency_ms_p50, latency_ms_p95,
                       throughput_sps, error, pruned, pruned_reason
                FROM benchmark_runs
                WHERE task_metric_value IS NOT NULL
                  AND task_metric_name = ?
                  {prune_clause}
                ORDER BY task_metric_value {order}
                LIMIT ?""",
            (task_metric_name, n),
        ).fetchall()
    finally:
        conn.close()
    return [dict(r) for r in rows]


def mark_pruned(
    benchmark_dir: Path,
    variant_ids: list[str],
    *,
    pruned: bool = True,
    reason: str | None = None,
) -> int:
    """Mark (or unmark) all benchmark rows for *variant_ids* as pruned.

    Pruning is reversible: it only flips a flag, never deletes recorded recipes
    or artifacts (CLAUDE.md reproducibility principle). Returns the row count
    affected.
    """
    if not variant_ids:
        return 0
    placeholders = ",".join("?" for _ in variant_ids)
    conn = _connect(benchmark_dir)
    try:
        cur = conn.execute(
            f"""UPDATE benchmark_runs
                SET pruned = ?, pruned_reason = ?
                WHERE variant_id IN ({placeholders})""",
            (1 if pruned else 0, reason if pruned else None, *variant_ids),
        )
        conn.commit()
        return cur.rowcount
    finally:
        conn.close()


def record_usage(
    benchmark_dir: Path,
    *,
    kind: str,
    wall_clock_s: float,
    variant_id: str | None = None,
    detail: str | None = None,
) -> int:
    """Append a resource-usage event (harness-written, append-only)."""
    conn = _connect(benchmark_dir)
    try:
        cur = conn.execute(
            """INSERT INTO resource_usage (timestamp, kind, variant_id, wall_clock_s, detail)
               VALUES (?,?,?,?,?)""",
            (datetime.now(timezone.utc).isoformat(), kind, variant_id, float(wall_clock_s), detail),
        )
        conn.commit()
        return cur.lastrowid  # type: ignore[return-value]
    finally:
        conn.close()


def get_usage_totals(benchmark_dir: Path) -> dict[str, float]:
    """Return cumulative resource usage: total wall-clock seconds + event counts."""
    conn = _connect(benchmark_dir)
    try:
        row = conn.execute(
            "SELECT COUNT(*) AS n_events, COALESCE(SUM(wall_clock_s), 0.0) AS wall_clock_s "
            "FROM resource_usage"
        ).fetchone()
        n_train = conn.execute(
            "SELECT COUNT(*) FROM resource_usage WHERE kind = 'train'"
        ).fetchone()[0]
    finally:
        conn.close()
    return {
        "n_events": float(row["n_events"]),
        "n_train": float(n_train),
        "wall_clock_s": float(row["wall_clock_s"]),
    }
