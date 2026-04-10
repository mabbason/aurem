"""Fact extraction from meeting transcripts using the configured AI provider.

Uses the same AI config as summary/lessons (anthropic API or Ollama).
Returns structured Fact dicts matching the `facts` table schema in db.py.
"""

from __future__ import annotations

import json
import httpx


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


VALID_CATEGORIES = {
    "personal",
    "professional",
    "opinion",
    "commitment",
    "preference",
    "context",
}

MIN_WORDS = 100


def format_transcript(session_data: dict) -> str:
    """Format session segments as [MM:SS] Speaker N: text."""
    lines = []
    for seg in session_data.get("segments", []):
        start = float(seg.get("start", 0.0))
        m = int(start // 60)
        s = int(start % 60)
        speaker = seg.get("speaker", "Speaker")
        text = str(seg.get("text", "")).strip()
        if text:
            lines.append(f"[{m:02d}:{s:02d}] {speaker}: {text}")
    return "\n".join(lines)


def _parse_fact_array(text: str) -> list[dict]:
    """Tolerant parser. Handles bare arrays, code-fenced arrays, and
    objects with a `facts` key (used by Ollama's json mode, which only
    emits objects)."""
    t = text.strip()
    if t.startswith("```"):
        first_nl = t.find("\n")
        t = t[first_nl + 1 :] if first_nl >= 0 else t[3:]
        if t.endswith("```"):
            t = t[:-3]
        t = t.strip()

    # Try bare array first: [ ... ]
    arr_start = t.find("[")
    arr_end = t.rfind("]")
    if arr_start >= 0 and arr_end > arr_start:
        try:
            parsed = json.loads(t[arr_start : arr_end + 1])
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass

    # Fallback: try object with a `facts` key
    obj_start = t.find("{")
    obj_end = t.rfind("}")
    if obj_start >= 0 and obj_end > obj_start:
        try:
            parsed = json.loads(t[obj_start : obj_end + 1])
        except json.JSONDecodeError:
            return []
        if isinstance(parsed, dict):
            for key in ("facts", "items", "results", "extracted"):
                value = parsed.get(key)
                if isinstance(value, list):
                    return value
    return []


def _normalize_facts(raw: list[dict]) -> list[dict]:
    """Drop malformed facts, coerce types, filter out invalid categories
    and confidence below 0.6."""
    cleaned = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        text = item.get("text")
        category = item.get("category")
        if not isinstance(text, str) or not text.strip():
            continue
        if category not in VALID_CATEGORIES:
            continue
        try:
            conf = float(item.get("confidence", 0.0))
        except (TypeError, ValueError):
            conf = 0.0
        if conf < 0.6:
            continue

        fact = {
            "text": text.strip(),
            "category": category,
            "speaker_label": item.get("speaker_label"),
            "confidence": conf,
            "segment_start": None,
            "segment_end": None,
        }
        try:
            if item.get("segment_start") is not None:
                fact["segment_start"] = float(item["segment_start"])
            if item.get("segment_end") is not None:
                fact["segment_end"] = float(item["segment_end"])
        except (TypeError, ValueError):
            pass
        cleaned.append(fact)
    return cleaned


async def _extract_via_anthropic(
    cfg: dict, prompt: str, timeout: float = 60.0
) -> str:
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": cfg.get("api_key", ""),
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": cfg.get("api_model", "claude-sonnet-4-20250514"),
                "max_tokens": 4096,
                "messages": [{"role": "user", "content": prompt}],
            },
        )
        if resp.status_code != 200:
            raise RuntimeError(f"Anthropic API returned {resp.status_code}: {resp.text[:200]}")
        body = resp.json()
        return body["content"][0]["text"]


async def _extract_via_ollama(cfg: dict, prompt: str, timeout: float = 120.0) -> str:
    url = cfg.get("ollama_url", "http://localhost:11434")
    model = cfg.get("ollama_model", "")
    if not model:
        raise RuntimeError("No Ollama model configured")
    # Ollama's format: json only constrains to JSON objects (not arrays),
    # so wrap the request in object form. The parser handles both shapes.
    wrapped_prompt = prompt.replace(
        "Return a JSON array of fact objects. Nothing else.",
        'Return a JSON object like {"facts": [...fact objects...]}. Nothing else.',
    )
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(
            f"{url}/api/generate",
            json={
                "model": model,
                "prompt": wrapped_prompt,
                "format": "json",
                "stream": False,
                "options": {"temperature": 0.2},
            },
        )
        if resp.status_code != 200:
            raise RuntimeError(f"Ollama returned {resp.status_code}: {resp.text[:200]}")
        body = resp.json()
        return body.get("response", "")


async def extract_facts(session_data: dict, cfg: dict) -> list[dict]:
    """Extract structured facts from a session transcript.

    Returns a list of fact dicts ready for `db.insert_facts()`. Short
    transcripts (<MIN_WORDS) return an empty list without hitting the API.
    Raises on API/network errors; caller decides how to record the failure.
    """
    transcript_text = format_transcript(session_data)
    if not transcript_text.strip():
        return []

    word_count = len(transcript_text.split())
    if word_count < MIN_WORDS:
        return []

    prompt = EXTRACTION_PROMPT.format(transcript_text=transcript_text)
    provider = cfg.get("provider", "")

    if provider == "api":
        raw_text = await _extract_via_anthropic(cfg, prompt)
    elif provider == "ollama":
        raw_text = await _extract_via_ollama(cfg, prompt)
    else:
        raise RuntimeError("No AI provider configured")

    parsed = _parse_fact_array(raw_text)
    return _normalize_facts(parsed)
