"""SQLite event log for the OpenJev front proxy.

Records every inbound inference request with the routing decision: which
host was chosen, every host's capacity score at decision time, and why
that host won. Best-effort: logging must never break serving.
"""

import json
import os
import sqlite3
import threading
import time
from datetime import datetime, timezone

DB_PATH = os.environ.get(
    "JEV_EVENT_DB",
    os.path.expanduser("~/workspace/openjev-ollama/events.db"),
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,                  -- epoch seconds
    ts_iso TEXT NOT NULL,              -- human-readable UTC
    kind TEXT NOT NULL,                -- 'request' | 'routing' | 'error'
    path TEXT,
    model TEXT,
    stream INTEGER,
    client TEXT,                       -- client host:port

    -- decision
    chosen_upstream TEXT,              -- url or 'local' / 'strata' / 'engine'
    why TEXT,                          -- rationale string
    route_latency_ms REAL,             -- ms from request start to first byte of decision

    -- scoring snapshot at decision time (JSON: per-host {capacity, free_mb, avg_util, max_temp})
    scores TEXT,

    -- outcome (filled when known; may be NULL for fire-and-forget)
    status INTEGER,
    duration_ms REAL,

    -- model test traces (kind='test')
    prompt TEXT,
    reply TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts);
CREATE INDEX IF NOT EXISTS idx_events_kind ON events(kind);
CREATE INDEX IF NOT EXISTS idx_events_chosen ON events(chosen_upstream);
CREATE INDEX IF NOT EXISTS idx_events_model ON events(model);
"""

_MIGRATE = (
    "ALTER TABLE events ADD COLUMN prompt TEXT",
    "ALTER TABLE events ADD COLUMN reply TEXT",
)

_INSERT = (
    "INSERT INTO events (ts, ts_iso, kind, path, model, stream, client, "
    "chosen_upstream, why, route_latency_ms, scores, status, duration_ms, prompt, reply) "
    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
)


class EventLog:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        with self._conn:
            self._conn.executescript(_SCHEMA)
        for stmt in _MIGRATE:  # existing DBs miss the new columns
            try:
                self._conn.execute(stmt)
            except Exception:
                pass  # column already exists

    def log(self, kind: str, **fields) -> None:
        """Best-effort insert; never raises."""
        try:
            now = time.time()
            row = (
                now,
                datetime.now(timezone.utc).isoformat(),
                kind,
                fields.get("path"),
                fields.get("model"),
                1 if fields.get("stream") else 0,
                fields.get("client"),
                fields.get("chosen_upstream"),
                fields.get("why"),
                fields.get("route_latency_ms"),
                json.dumps(fields["scores"], ensure_ascii=False) if fields.get("scores") else None,
                fields.get("status"),
                fields.get("duration_ms"),
                fields.get("prompt"),
                fields.get("reply"),
            )
            with self._lock:
                self._conn.execute(_INSERT, row)
                self._conn.commit()
        except Exception:
            pass

    def recent(self, limit: int = 100, kind: str | None = None):
        try:
            q = "SELECT * FROM events"
            args: list = []
            if kind:
                q += " WHERE kind=?"
                args.append(kind)
            q += " ORDER BY id DESC LIMIT ?"
            args.append(int(limit))
            cols = [c[1] for c in self._conn.execute("PRAGMA table_info(events)")]
            with self._lock:
                rows = self._conn.execute(q, args).fetchall()
            return [dict(zip(cols, r)) for r in rows]
        except Exception:
            return []

    def stats(self):
        try:
            with self._lock:
                total = self._conn.execute("SELECT COUNT(*) FROM events").fetchall()[0][0]
                by_upstream = self._conn.execute(
                    "SELECT chosen_upstream, COUNT(*) c FROM events "
                    "WHERE kind='request' GROUP BY chosen_upstream ORDER BY c DESC"
                ).fetchall()
                avg_dur = self._conn.execute(
                    "SELECT AVG(duration_ms) FROM events WHERE kind='request' AND duration_ms IS NOT NULL"
                ).fetchall()[0][0]
            return {"total": total, "by_upstream": [{"upstream": u, "count": c} for (u, c) in by_upstream],
                    "avg_duration_ms": round(avg_dur, 1) if avg_dur else None}
        except Exception:
            return {"total": 0, "by_upstream": [], "avg_duration_ms": None}

    def close(self):
        try:
            self._conn.close()
        except Exception:
            pass
