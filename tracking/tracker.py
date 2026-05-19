from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import List

import pandas as pd

import config

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          TEXT NOT NULL,
    strategy        TEXT NOT NULL,
    query           TEXT NOT NULL,
    answer          TEXT NOT NULL,
    input_tokens    INTEGER NOT NULL,
    output_tokens   INTEGER NOT NULL,
    total_tokens    INTEGER NOT NULL,
    cost_usd        REAL NOT NULL,
    latency_ms      REAL NOT NULL,
    retrieval_ms    REAL NOT NULL,
    generation_ms   REAL NOT NULL,
    accuracy_score  REAL,
    chunks_used     INTEGER,
    timestamp       DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_runs_strategy ON runs(strategy);
CREATE INDEX IF NOT EXISTS idx_runs_run_id   ON runs(run_id);
"""


@dataclass
class RunRecord:
    run_id: str
    strategy: str
    query: str
    answer: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_usd: float
    latency_ms: float
    retrieval_ms: float
    generation_ms: float
    accuracy_score: float | None = None
    chunks_used: int | None = None


class BenchmarkTracker:
    def __init__(self, db_path: Path = config.DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript(_SCHEMA)

    def log(self, record: RunRecord) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO runs
                    (run_id, strategy, query, answer,
                     input_tokens, output_tokens, total_tokens,
                     cost_usd, latency_ms, retrieval_ms, generation_ms,
                     accuracy_score, chunks_used)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    record.run_id,
                    record.strategy,
                    record.query,
                    record.answer,
                    record.input_tokens,
                    record.output_tokens,
                    record.total_tokens,
                    record.cost_usd,
                    record.latency_ms,
                    record.retrieval_ms,
                    record.generation_ms,
                    record.accuracy_score,
                    record.chunks_used,
                ),
            )

    def get_all(self) -> pd.DataFrame:
        with self._conn() as conn:
            return pd.read_sql("SELECT * FROM runs ORDER BY timestamp DESC", conn)

    def get_summary(self) -> pd.DataFrame:
        df = self.get_all()
        if df.empty:
            return df
        return (
            df.groupby("strategy")
            .agg(
                queries=("id", "count"),
                avg_input_tokens=("input_tokens", "mean"),
                avg_output_tokens=("output_tokens", "mean"),
                avg_total_tokens=("total_tokens", "mean"),
                total_cost_usd=("cost_usd", "sum"),
                avg_cost_usd=("cost_usd", "mean"),
                avg_latency_ms=("latency_ms", "mean"),
                avg_retrieval_ms=("retrieval_ms", "mean"),
                avg_generation_ms=("generation_ms", "mean"),
                avg_accuracy=("accuracy_score", "mean"),
                avg_chunks_used=("chunks_used", "mean"),
            )
            .reset_index()
        )

    def clear_run(self, run_id: str) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))
