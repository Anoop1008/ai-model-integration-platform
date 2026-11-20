"""Prediction log in SQLite: every prediction is one row.

SQLite ships with Python (no extra service or dependency) and is enough for one process.
/metrics/recent, the drift window and /metrics/summary are all SQL queries over this table.
shortcut: one file on local disk; on Render's free plan that disk is wiped
on restart or redeploy. With several instances or durable history, point this at PostgreSQL.
"""

import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

from app.schemas import ChurnRequest, ChurnResponse

INPUTS = ("tenure_months", "monthly_charges", "total_charges", "support_tickets", "contract_type")
OUTPUTS = ("request_id", "prediction", "confidence_score", "churn_probability", "risk_level", "inference_latency_ms", "model_version")

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at           TEXT    NOT NULL,          -- UTC, ISO 8601
    request_id           TEXT    NOT NULL UNIQUE,
    tenure_months        INTEGER NOT NULL,
    monthly_charges      REAL    NOT NULL,
    total_charges        REAL    NOT NULL,
    support_tickets      INTEGER NOT NULL,
    contract_type        TEXT    NOT NULL,
    prediction           TEXT    NOT NULL,
    confidence_score     REAL    NOT NULL,
    churn_probability    REAL    NOT NULL,
    risk_level           TEXT    NOT NULL,
    inference_latency_ms REAL    NOT NULL,
    model_version        TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_predictions_created_at ON predictions (created_at);
"""


class PredictionStore:
    def __init__(self, path: str | Path):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        # One connection shared by the threadpool workers. sqlite3 connections are not safe to use
        # from two threads at once, so every call takes the lock.
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")  # readers do not block the writer
            # No fsync on every commit (WAL is still crash-safe for the app; a power cut can lose the
            # last few rows). With the default FULL, p50 at 10 concurrent went from 6.6 ms to 118 ms.
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.executescript(SCHEMA)

    def save(self, req: ChurnRequest, res: ChurnResponse) -> None:
        row = {**req.model_dump(), **res.model_dump(), "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        cols = ("created_at", *INPUTS, *OUTPUTS)
        # Parameterised query: values are never pasted into the SQL text (no SQL injection).
        sql = f"INSERT INTO predictions ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})"
        with self._lock:
            self._conn.execute(sql, [row[c] for c in cols])
            self._conn.commit()

    def _query(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, params).fetchall()

    def recent(self, limit: int) -> list[dict]:
        """Last `limit` predictions, oldest first (same shape the API has always returned)."""
        rows = self._query(f"SELECT {', '.join(INPUTS + OUTPUTS)} FROM predictions ORDER BY id DESC LIMIT ?", (limit,))
        return [{"input": {c: r[c] for c in INPUTS}, "output": {c: r[c] for c in OUTPUTS}} for r in reversed(rows)]

    def recent_inputs(self, limit: int) -> list[dict]:
        rows = self._query(f"SELECT {', '.join(INPUTS)} FROM predictions ORDER BY id DESC LIMIT ?", (limit,))
        return [dict(r) for r in rows]

    def summary(self) -> dict:
        totals = self._query(
            """SELECT COUNT(*)                         AS total,
                      COALESCE(SUM(prediction = 'Churn'), 0) AS churn,
                      AVG(churn_probability)           AS avg_probability,
                      AVG(inference_latency_ms)        AS avg_latency_ms,
                      MIN(created_at)                  AS first_at,
                      MAX(created_at)                  AS last_at
               FROM predictions"""
        )[0]
        by_risk = self._query("SELECT risk_level, COUNT(*) AS n FROM predictions GROUP BY risk_level")
        by_contract = self._query(
            """SELECT contract_type,
                      COUNT(*)                     AS n,
                      AVG(prediction = 'Churn')    AS churn_rate,
                      AVG(churn_probability)       AS avg_probability
               FROM predictions GROUP BY contract_type ORDER BY contract_type"""
        )
        r4 = lambda v: None if v is None else round(v, 4)  # noqa: E731
        return {
            "total": totals["total"],
            "churn": totals["churn"],
            "churn_rate": r4(totals["churn"] / totals["total"]) if totals["total"] else None,
            "avg_probability": r4(totals["avg_probability"]),
            "avg_latency_ms": r4(totals["avg_latency_ms"]),
            "first_at": totals["first_at"],
            "last_at": totals["last_at"],
            "by_risk": {r["risk_level"]: r["n"] for r in by_risk},
            "by_contract": [
                {"contract_type": r["contract_type"], "n": r["n"], "churn_rate": r4(r["churn_rate"]), "avg_probability": r4(r["avg_probability"])}
                for r in by_contract
            ],
        }

    def clear(self) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM predictions")
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()
