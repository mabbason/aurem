"""Retroactive fact-extraction CLI.

Scans `config.SESSIONS_DIR` for past sessions and runs the Phase 6 fact
extraction on each one that doesn't already have facts (or whose last
extraction errored). Idempotent: re-runs skip sessions with status 'ok'
unless --force is passed.

Usage:
    python -m transcriber.extract_all [--force] [--session ID]... [--dry-run]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import config  # noqa: E402
from server.ai_config import load_ai_config  # noqa: E402
from transcriber import db  # noqa: E402
from transcriber.extraction import extract_facts, MIN_WORDS, format_transcript  # noqa: E402


def _iter_sessions() -> list[tuple[str, Path]]:
    sessions: list[tuple[str, Path]] = []
    if not config.SESSIONS_DIR.exists():
        return sessions
    for d in sorted(config.SESSIONS_DIR.iterdir()):
        if not d.is_dir():
            continue
        t = d / "transcript.json"
        if t.exists():
            sessions.append((d.name, t))
    return sessions


def _load(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"  ! failed to read: {e}")
        return None


async def _run(session_id: str, data: dict, cfg: dict, dry_run: bool) -> tuple[bool, str]:
    """Returns (success, message)."""
    transcript_text = format_transcript(data)
    word_count = len(transcript_text.split())
    if word_count < MIN_WORDS:
        return True, f"skipped: {word_count} words (<{MIN_WORDS})"

    if dry_run:
        return True, f"would extract from {word_count} words"

    try:
        facts = await extract_facts(data, cfg)
    except Exception as e:
        db.record_extraction(session_id, "error", str(e)[:500])
        return False, f"error: {e}"

    if not facts:
        db.record_extraction(session_id, "ok", None)
        return True, "0 facts"

    inserted = db.insert_facts(facts, session_id=session_id)

    # Propagate existing speaker mappings to the new facts.
    mappings = db.get_session_speaker_mappings(session_id)
    for label, pid in mappings.items():
        if pid is not None:
            db.set_session_speaker_mapping(session_id, label, pid)

    db.record_extraction(session_id, "ok", None)
    return True, f"inserted {inserted} facts"


async def main_async(args) -> int:
    db.init_db()
    cfg = load_ai_config()
    if not cfg.get("provider") and not args.dry_run:
        print("ERROR: No AI provider configured. Set one in the app first.")
        return 2

    sessions = _iter_sessions()
    if args.session:
        wanted = set(args.session)
        sessions = [s for s in sessions if s[0] in wanted]
        missing = wanted - {s[0] for s in sessions}
        for m in missing:
            print(f"! {m}: no transcript.json found")

    if not sessions:
        print("No sessions to process.")
        return 0

    ok_count = 0
    skip_count = 0
    err_count = 0

    for session_id, transcript_path in sessions:
        prior = db.get_extraction_status(session_id)
        if prior and prior["status"] == "ok" and not args.force:
            print(f"- {session_id}: already extracted (use --force to redo)")
            skip_count += 1
            continue

        if db.session_has_facts(session_id) and not args.force:
            print(f"- {session_id}: has existing facts (use --force to redo)")
            skip_count += 1
            continue

        data = _load(transcript_path)
        if data is None:
            err_count += 1
            continue

        title = data.get("title", session_id)
        print(f"* {session_id} ({title})")
        success, msg = await _run(session_id, data, cfg, args.dry_run)
        print(f"    {msg}")
        if success:
            ok_count += 1
        else:
            err_count += 1

    print(
        f"\nDone. ok={ok_count} skipped={skip_count} errors={err_count}"
    )
    return 0 if err_count == 0 else 1


def main():
    parser = argparse.ArgumentParser(description="Retroactive fact extraction")
    parser.add_argument("--force", action="store_true", help="Re-extract even if already done")
    parser.add_argument("--session", action="append", default=[], help="Only run this session (repeatable)")
    parser.add_argument("--dry-run", action="store_true", help="Report what would run without calling the API")
    args = parser.parse_args()
    sys.exit(asyncio.run(main_async(args)))


if __name__ == "__main__":
    main()
