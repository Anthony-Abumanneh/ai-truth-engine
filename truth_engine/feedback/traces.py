"""SQLite store for investigations, feedback and watched claims (spec §14;
CONCEPTS.md §11: SQLite).

    traces    every audit / ask run: the spec §14 record (flat, for export
              and training) plus the full trace JSON (for replay and the UI)
    feedback  a person's judgement on one claim's verdict ("this should be
              contradicted"); becomes router rewards and calibration data
    watched   claims to re-check later; each re-check appends to history so a
              changed verdict (new study, retraction) can be reported

One file (data/truth_engine.sqlite by default, TRUTH_ENGINE_DB to move it);
safe to use from several threads.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

from ..config import DB_PATH
from ..schemas import EvidenceStatus, Trace

SCHEMA = """
CREATE TABLE IF NOT EXISTS traces (
    query_id TEXT PRIMARY KEY, created_at REAL, mode TEXT, question TEXT,
    record TEXT, trace TEXT
);
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT, query_id TEXT, claim_id TEXT, claim_text TEXT,
    predicted TEXT, label TEXT, note TEXT, created_at REAL, used INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS watched (
    claim_key TEXT PRIMARY KEY, text TEXT, created_at REAL, last_checked REAL,
    last_status TEXT, last_confidence REAL, history TEXT
);
"""


def claim_key(text: str) -> str:
    return " ".join(text.lower().split()).rstrip(".")


class TraceStore:
    def __init__(self, path: Path | str = DB_PATH):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(self.path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        with self._lock:
            self._db.executescript(SCHEMA)
            self._db.commit()

    def _exec(self, sql: str, args: tuple = ()) -> list[sqlite3.Row]:
        with self._lock:
            rows = self._db.execute(sql, args).fetchall()
            self._db.commit()
            return rows

    # ------------------------------------------------------------ traces
    def save_trace(self, trace: Trace) -> None:
        self._exec("INSERT OR REPLACE INTO traces VALUES (?,?,?,?,?,?)",
                   (trace.query_id, trace.created_at, trace.mode, trace.question,
                    json.dumps(trace.section14_record(), default=str), trace.model_dump_json()))

    def get_trace(self, query_id: str) -> Trace | None:
        rows = self._exec("SELECT trace FROM traces WHERE query_id=?", (query_id,))
        return Trace.model_validate_json(rows[0]["trace"]) if rows else None

    def list_traces(self, limit: int = 50) -> list[dict]:
        rows = self._exec("SELECT query_id, created_at, mode, question FROM traces ORDER BY created_at DESC LIMIT ?",
                          (limit,))
        return [dict(r) for r in rows]

    def export_section14(self) -> list[dict]:
        """The spec §14 training records for every stored run."""
        return [json.loads(r["record"]) for r in self._exec("SELECT record FROM traces ORDER BY created_at")]

    # ------------------------------------------------------------ feedback
    def add_feedback(self, query_id: str, claim_id: str, label: str, note: str = "") -> int:
        """`label` = the verdict the claim should have had (an EvidenceStatus value)."""
        if label not in {s.value for s in EvidenceStatus}:
            raise ValueError(f"label must be one of {[s.value for s in EvidenceStatus]}")
        trace = self.get_trace(query_id)
        claim = next((c for c in trace.claims if c.id == claim_id), None) if trace else None
        verdict = next((v for v in trace.verdicts if v.claim_id == claim_id), None) if trace else None
        if claim is None:
            raise KeyError(f"no claim {claim_id} in trace {query_id}")
        with self._lock:
            cur = self._db.execute(
                "INSERT INTO feedback (query_id, claim_id, claim_text, predicted, label, note, created_at) "
                "VALUES (?,?,?,?,?,?,?)",
                (query_id, claim_id, claim.text, verdict.status.value if verdict else None, label, note, time.time()))
            self._db.commit()
            return cur.lastrowid

    def feedback(self, unused_only: bool = False) -> list[dict]:
        sql = "SELECT * FROM feedback" + (" WHERE used=0" if unused_only else "") + " ORDER BY created_at"
        return [dict(r) for r in self._exec(sql)]

    def mark_used(self, ids: list[int]) -> None:
        for i in ids:
            self._exec("UPDATE feedback SET used=1 WHERE id=?", (i,))

    # ------------------------------------------------------------ watched claims
    def watch(self, text: str) -> str:
        key = claim_key(text)
        self._exec("INSERT OR IGNORE INTO watched VALUES (?,?,?,?,?,?,?)",
                   (key, text, time.time(), None, None, None, "[]"))
        return key

    def unwatch(self, text: str) -> bool:
        with self._lock:
            cur = self._db.execute("DELETE FROM watched WHERE claim_key=?", (claim_key(text),))
            self._db.commit()
            return cur.rowcount > 0

    def watched(self) -> list[dict]:
        out = []
        for r in self._exec("SELECT * FROM watched ORDER BY created_at"):
            d = dict(r)
            d["history"] = json.loads(d["history"] or "[]")
            out.append(d)
        return out

    def record_check(self, text: str, status: str, confidence: float, query_id: str) -> dict | None:
        """Append a re-check; returns a change description if the verdict moved."""
        key = claim_key(text)
        rows = self._exec("SELECT last_status, last_confidence, history FROM watched WHERE claim_key=?", (key,))
        if not rows:
            return None
        prev_status, prev_conf = rows[0]["last_status"], rows[0]["last_confidence"]
        history = json.loads(rows[0]["history"] or "[]")
        now = time.time()
        history.append({"t": now, "status": status, "confidence": confidence, "query_id": query_id})
        self._exec("UPDATE watched SET last_checked=?, last_status=?, last_confidence=?, history=? WHERE claim_key=?",
                   (now, status, confidence, json.dumps(history), key))
        if prev_status is None:
            return None
        if prev_status != status:
            return {"claim": text, "kind": "status_changed", "from": prev_status, "to": status,
                    "confidence": confidence, "query_id": query_id}
        if prev_conf is not None and abs(confidence - prev_conf) >= 0.15:
            return {"claim": text, "kind": "confidence_changed", "from": round(prev_conf, 3),
                    "to": round(confidence, 3), "status": status, "query_id": query_id}
        return None
