"""
Phase 6 Step 0 validation gate.

Reads real past sessions, formats transcripts as [MM:SS] Speaker N: text,
sends to Claude API with the v1 fact-extraction prompt, and saves the raw
JSON outputs for manual scoring.

Usage:
    python scripts/validate_extraction.py [--sessions-dir DIR] [--session ID]...

Requires either:
    - ai_config.json (same format the app uses) with provider=api, api_provider=anthropic
    - ANTHROPIC_API_KEY env var as fallback

Outputs go to scripts/validation-run-YYYYMMDD/.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import config  # noqa: E402

EXTRACTION_PROMPT = """You are extracting facts about people from a meeting transcript.
The transcript uses speaker labels like "Speaker 1", "Speaker 2" etc.
These labels may not yet be mapped to real names. Extract facts per speaker label.

For each fact, return a JSON object:
{{
  "text": "one clear sentence stating the fact",
  "category": "personal | professional | opinion | commitment | preference | context",
  "speaker_label": "Speaker 1",
  "segment_start": 45.2,
  "segment_end": 52.8,
  "confidence": 0.85
}}

Categories:
- personal: family, health, location, hobbies, life events
- professional: role, company, projects, skills, career moves
- opinion: views on topics, preferences stated as beliefs
- commitment: promises, deadlines, action items
- preference: likes/dislikes, how they want things done
- context: situational info that may be useful later

Rules:
- One fact per object. Don't combine multiple facts.
- Use the exact speaker label from the transcript.
- Confidence: 0.9+ for explicitly stated facts, 0.6-0.8 for inferred, below 0.6 skip.
- Ignore small talk, greetings, filler. Focus on things you'd want to remember.

Return a JSON array of fact objects. Nothing else.

TRANSCRIPT:
{transcript_text}"""


def load_api_credentials() -> tuple[str, str]:
    """Returns (api_key, model). Tries ai_config.json, then env vars."""
    if config.AI_CONFIG_PATH.exists():
        cfg = json.loads(config.AI_CONFIG_PATH.read_text())
        if cfg.get("provider") == "api" and cfg.get("api_provider") == "anthropic":
            key = cfg.get("api_key", "").strip()
            model = cfg.get("api_model", "claude-sonnet-4-20250514").strip()
            if key:
                return key, model

    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if key:
        model = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514").strip()
        return key, model

    raise SystemExit(
        "No Anthropic API key found. Configure via ai_config.json (provider=api, "
        "api_provider=anthropic) or set ANTHROPIC_API_KEY env var."
    )


def format_timestamp(seconds: float) -> str:
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"


def format_transcript(session_data: dict) -> str:
    """Formats as [MM:SS] Speaker N: text, one line per segment."""
    lines = []
    for seg in session_data.get("segments", []):
        ts = format_timestamp(seg.get("start", 0.0))
        speaker = seg.get("speaker", "Speaker")
        text = seg.get("text", "").strip()
        if text:
            lines.append(f"[{ts}] {speaker}: {text}")
    return "\n".join(lines)


def load_session(sessions_dir: Path, session_id: str) -> dict:
    transcript_path = sessions_dir / session_id / "transcript.json"
    if not transcript_path.exists():
        raise FileNotFoundError(f"No transcript.json at {transcript_path}")
    return json.loads(transcript_path.read_text(encoding="utf-8"))


def pick_default_sessions(sessions_dir: Path, want: int = 3) -> list[str]:
    """Pick the N longest sessions with real speakers."""
    candidates = []
    for session_dir in sessions_dir.iterdir():
        if not session_dir.is_dir():
            continue
        t = session_dir / "transcript.json"
        if not t.exists():
            continue
        try:
            data = json.loads(t.read_text(encoding="utf-8"))
        except Exception:
            continue
        dur = float(data.get("duration", 0) or 0)
        segs = data.get("segments", [])
        if dur < 120 or not segs:
            continue
        candidates.append((dur, session_dir.name))
    candidates.sort(reverse=True)
    return [name for _, name in candidates[:want]]


def call_claude(api_key: str, model: str, prompt: str, timeout: float = 120.0) -> dict:
    """Synchronous Claude API call. Returns {'text': ..., 'usage': {...}} or raises."""
    with httpx.Client(timeout=timeout) as client:
        resp = client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": model,
                "max_tokens": 4096,
                "messages": [{"role": "user", "content": prompt}],
            },
        )
        resp.raise_for_status()
        body = resp.json()
        text = body["content"][0]["text"]
        return {"text": text, "usage": body.get("usage", {})}


def parse_facts(text: str) -> list[dict]:
    """Tolerant JSON array parser: strips code fences, handles leading/trailing junk."""
    t = text.strip()
    if t.startswith("```"):
        first_nl = t.find("\n")
        t = t[first_nl + 1 :] if first_nl >= 0 else t[3:]
        if t.endswith("```"):
            t = t[:-3]
        t = t.strip()

    start = t.find("[")
    end = t.rfind("]")
    if start < 0 or end < 0 or end <= start:
        raise ValueError(f"No JSON array found in response: {t[:200]}")

    return json.loads(t[start : end + 1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sessions-dir",
        type=Path,
        default=Path("/mnt/c/Users/mawin/projects/aurem/sessions"),
        help="Directory containing session subdirectories (default: Windows runtime)",
    )
    parser.add_argument(
        "--session",
        action="append",
        default=[],
        help="Session ID to include (repeatable). If omitted, picks 3 longest.",
    )
    parser.add_argument("--out", type=Path, default=None, help="Output directory")
    args = parser.parse_args()

    if not args.sessions_dir.exists():
        raise SystemExit(f"sessions-dir does not exist: {args.sessions_dir}")

    api_key, model = load_api_credentials()
    print(f"Using model: {model}")

    session_ids = args.session or pick_default_sessions(args.sessions_dir, want=3)
    if not session_ids:
        raise SystemExit("No sessions found to validate against.")

    out_dir = args.out or (
        REPO_ROOT / "scripts" / f"validation-run-{datetime.now().strftime('%Y%m%d')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing outputs to: {out_dir}")

    (out_dir / "prompt.txt").write_text(EXTRACTION_PROMPT, encoding="utf-8")

    summary_lines = ["# Validation run", f"Date: {datetime.now().isoformat()}", f"Model: {model}", ""]

    for sid in session_ids:
        print(f"\n=== {sid} ===")
        try:
            data = load_session(args.sessions_dir, sid)
        except FileNotFoundError as e:
            print(f"  SKIP: {e}")
            continue

        title = data.get("title", sid)
        duration = data.get("duration", 0)
        transcript_text = format_transcript(data)
        word_count = len(transcript_text.split())
        print(f"  Title: {title}")
        print(f"  Duration: {duration}s, {word_count} words")

        if word_count < 100:
            print("  SKIP: transcript under 100 words")
            continue

        prompt = EXTRACTION_PROMPT.format(transcript_text=transcript_text)

        (out_dir / f"{sid}_transcript.txt").write_text(transcript_text, encoding="utf-8")

        try:
            result = call_claude(api_key, model, prompt)
        except Exception as e:
            print(f"  ERROR: {e}")
            (out_dir / f"{sid}_error.txt").write_text(str(e), encoding="utf-8")
            summary_lines.append(f"- {sid} ({title}): ERROR {e}")
            continue

        (out_dir / f"{sid}_raw.txt").write_text(result["text"], encoding="utf-8")

        try:
            facts = parse_facts(result["text"])
        except Exception as e:
            print(f"  Parse error: {e}")
            summary_lines.append(f"- {sid} ({title}): PARSE ERROR {e}")
            continue

        (out_dir / f"{sid}_facts.json").write_text(
            json.dumps(facts, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        by_category: dict[str, int] = {}
        for f in facts:
            by_category[f.get("category", "?")] = by_category.get(f.get("category", "?"), 0) + 1

        usage = result.get("usage", {})
        print(f"  Facts extracted: {len(facts)}")
        print(f"  By category: {by_category}")
        print(f"  Usage: {usage}")
        minutes = max(duration / 60.0, 1)
        rate = len(facts) / minutes * 30
        summary_lines.append(
            f"- {sid} ({title}): {len(facts)} facts, "
            f"{rate:.1f} per 30min, categories={by_category}"
        )

    scores_path = out_dir / "scores.md"
    if not scores_path.exists():
        scores_path.write_text(
            "# Manual scoring\n\n"
            "For each fact in *_facts.json, tag as useful / noise / wrong.\n\n"
            "Pass criteria:\n"
            "- 3-5 useful facts per 30 minutes\n"
            "- under 20% noise\n"
            "- zero hallucinations (facts not in transcript)\n\n"
            + "\n".join(summary_lines),
            encoding="utf-8",
        )

    print(f"\nDone. Review JSON files in {out_dir} and record scores in {scores_path}")


if __name__ == "__main__":
    main()
