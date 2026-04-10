"""Tests for the relationship-intelligence SQLite data layer."""

from __future__ import annotations

import pytest

from transcriber import db


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    """Every test gets a fresh, isolated SQLite file."""
    db_path = tmp_path / "test.db"
    # Reset module-level state so init_db() builds a fresh schema.
    monkeypatch.setattr(db, "_DB_PATH", None)
    db.init_db(db_path)
    yield db_path
    monkeypatch.setattr(db, "_DB_PATH", None)


# --- people ---

def test_create_and_get_person():
    pid = db.create_person("Sarah Chen", "Former colleague")
    assert pid > 0
    person = db.get_person(pid)
    assert person is not None
    assert person["name"] == "Sarah Chen"
    assert person["notes"] == "Former colleague"


def test_create_person_trims_empty_notes_to_none():
    pid = db.create_person("No Notes", "   ")
    person = db.get_person(pid)
    assert person["notes"] is None


def test_list_people_sorts_by_last_seen():
    a = db.create_person("Alice")
    b = db.create_person("Bob")
    # Bob has the more recent fact
    db.insert_fact(
        session_id="s1", speaker_label="Speaker 1",
        text="Bob loves jazz.", category="personal", person_id=b, confidence=0.9,
    )
    people = db.list_people()
    names = [p["name"] for p in people]
    assert names.index("Bob") < names.index("Alice")
    bob = next(p for p in people if p["name"] == "Bob")
    assert bob["fact_count"] == 1


def test_update_person_notes():
    pid = db.create_person("Test")
    assert db.update_person_notes(pid, "new notes") is True
    assert db.get_person(pid)["notes"] == "new notes"
    assert db.update_person_notes(9999, "nope") is False


def test_delete_person_unlinks_facts():
    pid = db.create_person("Temporary")
    fid = db.insert_fact(
        session_id="s1", speaker_label="Speaker 1",
        text="Fact text.", category="context", person_id=pid,
    )
    assert db.delete_person(pid) is True
    assert db.get_person(pid) is None
    # Fact is preserved with person_id NULL (ON DELETE SET NULL)
    facts = db.facts_for_session("s1")
    assert len(facts) == 1
    assert facts[0]["person_id"] is None
    assert facts[0]["id"] == fid


# --- facts ---

def test_insert_facts_bulk_and_filter_malformed():
    facts = [
        {"text": "Valid fact.", "category": "personal", "confidence": 0.9, "speaker_label": "Speaker 1"},
        {"text": "Another valid.", "category": "professional", "confidence": 0.85, "speaker_label": "Speaker 2"},
        {"category": "personal"},  # missing text
        "not a dict",  # not a dict
    ]
    inserted = db.insert_facts(facts, session_id="sess1")
    assert inserted == 2
    rows = db.facts_for_session("sess1")
    assert len(rows) == 2
    assert all(r["source"] == "auto" for r in rows)


def test_insert_facts_empty():
    assert db.insert_facts([], session_id="sess1") == 0


def test_session_has_facts():
    assert db.session_has_facts("s1") is False
    db.insert_fact(
        session_id="s1", speaker_label="Speaker 1",
        text="Hi.", category="context",
    )
    assert db.session_has_facts("s1") is True


def test_facts_for_person_sorted_by_extracted_at():
    pid = db.create_person("Alice")
    db.insert_fact(
        session_id="s1", speaker_label="Speaker 1",
        text="First fact.", category="personal", person_id=pid,
    )
    db.insert_fact(
        session_id="s2", speaker_label="Speaker 1",
        text="Second fact.", category="professional", person_id=pid,
    )
    facts = db.facts_for_person(pid)
    assert len(facts) == 2
    # Most recent first
    assert facts[0]["text"] == "Second fact."


def test_delete_fact():
    fid = db.insert_fact(
        session_id="s1", speaker_label="Speaker 1",
        text="Tmp.", category="context",
    )
    assert db.delete_fact(fid) is True
    assert db.delete_fact(fid) is False


# --- session speakers ---

def test_set_session_speaker_mapping_propagates_to_existing_facts():
    pid = db.create_person("Alice")
    fid = db.insert_fact(
        session_id="s1", speaker_label="Speaker 1",
        text="Facts about alice.", category="personal",
    )
    # No person yet
    assert db.facts_for_session("s1")[0]["person_id"] is None

    db.set_session_speaker_mapping("s1", "Speaker 1", pid)

    mappings = db.get_session_speaker_mappings("s1")
    assert mappings == {"Speaker 1": pid}

    facts = db.facts_for_session("s1")
    assert facts[0]["person_id"] == pid
    assert facts[0]["id"] == fid


def test_set_session_speaker_mapping_upsert_clears():
    pid = db.create_person("Alice")
    db.set_session_speaker_mapping("s1", "Speaker 1", pid)
    db.set_session_speaker_mapping("s1", "Speaker 1", None)
    mappings = db.get_session_speaker_mappings("s1")
    assert mappings == {"Speaker 1": None}


def test_sessions_for_person():
    pid = db.create_person("Alice")
    db.set_session_speaker_mapping("s1", "Speaker 1", pid)
    db.set_session_speaker_mapping("s2", "Speaker 2", pid)
    sessions = db.sessions_for_person(pid)
    session_ids = {s["session_id"] for s in sessions}
    assert session_ids == {"s1", "s2"}


# --- extractions ---

def test_record_and_get_extraction_status():
    db.record_extraction("s1", "ok")
    status = db.get_extraction_status("s1")
    assert status["status"] == "ok"
    assert status["error"] is None


def test_record_extraction_upsert():
    db.record_extraction("s1", "error", "boom")
    db.record_extraction("s1", "ok")
    status = db.get_extraction_status("s1")
    assert status["status"] == "ok"
    assert status["error"] is None
