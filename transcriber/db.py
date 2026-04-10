"""SQLite data layer for Aurem relationship intelligence.

Stores the person graph: people, facts (auto-extracted or manual), and
session-to-person speaker mappings. Additive only; session JSON on disk
is untouched.

DB path is `%APPDATA%\\Aurem\\aurem.db` on Windows, `~/.aurem/aurem.db`
elsewhere. The path is set once at import time and can be overridden by
`init_db(path)` for tests.
"""

from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


def get_default_db_path() -> Path:
    """Resolve the user-data DB path. On Windows Aurem installs, this is
    `%APPDATA%\\Aurem\\aurem.db`. Falls back to `~/.aurem/aurem.db`."""
    appdata = os.environ.get("APPDATA")
    if appdata:
        db_dir = Path(appdata) / "Aurem"
    else:
        db_dir = Path.home() / ".aurem"
    db_dir.mkdir(parents=True, exist_ok=True)
    return db_dir / "aurem.db"


SCHEMA = """
CREATE TABLE IF NOT EXISTS people (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    name       TEXT NOT NULL,
    notes      TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS facts (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id     INTEGER REFERENCES people(id) ON DELETE SET NULL,
    session_id    TEXT,
    speaker_label TEXT,
    text          TEXT NOT NULL,
    category      TEXT NOT NULL,
    source        TEXT NOT NULL DEFAULT 'auto',
    confidence    REAL,
    segment_start REAL,
    segment_end   REAL,
    extracted_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS session_speakers (
    session_id    TEXT NOT NULL,
    speaker_label TEXT NOT NULL,
    person_id     INTEGER REFERENCES people(id) ON DELETE SET NULL,
    PRIMARY KEY (session_id, speaker_label)
);

CREATE TABLE IF NOT EXISTS session_extractions (
    session_id TEXT PRIMARY KEY,
    status     TEXT NOT NULL,   -- 'ok', 'pending', 'error', 'skipped'
    error      TEXT,
    ran_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_facts_person ON facts(person_id);
CREATE INDEX IF NOT EXISTS idx_facts_session ON facts(session_id);
CREATE INDEX IF NOT EXISTS idx_facts_session_speaker ON facts(session_id, speaker_label);
CREATE INDEX IF NOT EXISTS idx_session_speakers_person ON session_speakers(person_id);
"""


_DB_PATH: Path | None = None


def init_db(path: Path | None = None) -> Path:
    """Create the schema if needed. Idempotent. Records the resolved path
    globally so subsequent `connect()` calls use it."""
    global _DB_PATH
    resolved = Path(path) if path is not None else get_default_db_path()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(resolved) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.executescript(SCHEMA)
    _DB_PATH = resolved
    return resolved


def get_db_path() -> Path:
    if _DB_PATH is None:
        return init_db()
    return _DB_PATH


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    path = get_db_path()
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


# --- people ---

def list_people() -> list[dict]:
    """Return all people with fact count and last-seen date (max extracted_at
    of their facts). Sorted most-recent-first."""
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT
                p.id,
                p.name,
                p.notes,
                p.created_at,
                COUNT(f.id) AS fact_count,
                MAX(f.extracted_at) AS last_seen
            FROM people p
            LEFT JOIN facts f ON f.person_id = p.id
            GROUP BY p.id
            ORDER BY
                CASE WHEN MAX(f.extracted_at) IS NULL THEN 1 ELSE 0 END,
                MAX(f.extracted_at) DESC,
                p.created_at DESC
            """
        ).fetchall()
    return [dict(r) for r in rows]


def get_person(person_id: int) -> dict | None:
    with connect() as conn:
        row = conn.execute(
            "SELECT id, name, notes, created_at FROM people WHERE id = ?",
            (person_id,),
        ).fetchone()
    return dict(row) if row else None


def create_person(name: str, notes: str | None = None) -> int:
    with connect() as conn:
        cur = conn.execute(
            "INSERT INTO people (name, notes) VALUES (?, ?)",
            (name.strip(), (notes or "").strip() or None),
        )
        return int(cur.lastrowid)


def update_person_notes(person_id: int, notes: str) -> bool:
    with connect() as conn:
        cur = conn.execute(
            "UPDATE people SET notes = ? WHERE id = ?",
            (notes, person_id),
        )
        return cur.rowcount > 0


def delete_person(person_id: int) -> bool:
    with connect() as conn:
        cur = conn.execute("DELETE FROM people WHERE id = ?", (person_id,))
        return cur.rowcount > 0


# --- facts ---

def insert_fact(
    *,
    session_id: str | None,
    speaker_label: str | None,
    text: str,
    category: str,
    source: str = "auto",
    confidence: float | None = None,
    segment_start: float | None = None,
    segment_end: float | None = None,
    person_id: int | None = None,
) -> int:
    with connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO facts (
                person_id, session_id, speaker_label, text, category,
                source, confidence, segment_start, segment_end
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                person_id,
                session_id,
                speaker_label,
                text.strip(),
                category,
                source,
                confidence,
                segment_start,
                segment_end,
            ),
        )
        return int(cur.lastrowid)


def insert_facts(facts: list[dict], *, session_id: str) -> int:
    """Bulk insert extracted facts for a session. Returns number inserted."""
    if not facts:
        return 0
    rows = []
    for f in facts:
        if not isinstance(f, dict):
            continue
        try:
            rows.append(
                (
                    session_id,
                    f.get("speaker_label"),
                    str(f["text"]).strip(),
                    str(f.get("category", "context")),
                    "auto",
                    float(f.get("confidence")) if f.get("confidence") is not None else None,
                    float(f["segment_start"]) if f.get("segment_start") is not None else None,
                    float(f["segment_end"]) if f.get("segment_end") is not None else None,
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    if not rows:
        return 0
    with connect() as conn:
        conn.executemany(
            """
            INSERT INTO facts (
                session_id, speaker_label, text, category,
                source, confidence, segment_start, segment_end
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
    return len(rows)


def delete_fact(fact_id: int) -> bool:
    with connect() as conn:
        cur = conn.execute("DELETE FROM facts WHERE id = ?", (fact_id,))
        return cur.rowcount > 0


def facts_for_person(person_id: int) -> list[dict]:
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT id, person_id, session_id, speaker_label, text, category,
                   source, confidence, segment_start, segment_end, extracted_at
            FROM facts
            WHERE person_id = ?
            ORDER BY extracted_at DESC, id DESC
            """,
            (person_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def facts_for_session(session_id: str) -> list[dict]:
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT id, person_id, session_id, speaker_label, text, category,
                   source, confidence, segment_start, segment_end, extracted_at
            FROM facts
            WHERE session_id = ?
            ORDER BY segment_start, id
            """,
            (session_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def session_has_facts(session_id: str) -> bool:
    with connect() as conn:
        row = conn.execute(
            "SELECT 1 FROM facts WHERE session_id = ? LIMIT 1", (session_id,)
        ).fetchone()
    return row is not None


# --- session speakers ---

def get_session_speaker_mappings(session_id: str) -> dict[str, int | None]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT speaker_label, person_id FROM session_speakers WHERE session_id = ?",
            (session_id,),
        ).fetchall()
    return {r["speaker_label"]: r["person_id"] for r in rows}


def set_session_speaker_mapping(
    session_id: str, speaker_label: str, person_id: int | None
) -> None:
    """Upsert a single mapping and propagate person_id to existing facts
    for (session_id, speaker_label)."""
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO session_speakers (session_id, speaker_label, person_id)
            VALUES (?, ?, ?)
            ON CONFLICT(session_id, speaker_label)
            DO UPDATE SET person_id = excluded.person_id
            """,
            (session_id, speaker_label, person_id),
        )
        conn.execute(
            "UPDATE facts SET person_id = ? WHERE session_id = ? AND speaker_label = ?",
            (person_id, session_id, speaker_label),
        )


def sessions_for_person(person_id: int) -> list[dict]:
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT DISTINCT session_id
            FROM session_speakers
            WHERE person_id = ?
            """,
            (person_id,),
        ).fetchall()
    return [dict(r) for r in rows]


# --- extractions ---

def record_extraction(session_id: str, status: str, error: str | None = None) -> None:
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO session_extractions (session_id, status, error, ran_at)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(session_id)
            DO UPDATE SET status = excluded.status,
                          error = excluded.error,
                          ran_at = excluded.ran_at
            """,
            (session_id, status, error),
        )


def get_extraction_status(session_id: str) -> dict | None:
    with connect() as conn:
        row = conn.execute(
            "SELECT session_id, status, error, ran_at FROM session_extractions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
    return dict(row) if row else None
