"""Tests for the Phase 6 fact-extraction module."""

from __future__ import annotations

import pytest

from transcriber.extraction import (
    EXTRACTION_PROMPT,
    MIN_WORDS,
    VALID_CATEGORIES,
    _normalize_facts,
    _parse_fact_array,
    extract_facts,
    format_transcript,
)


# --- format_transcript ---

def test_format_transcript_renders_mm_ss_speaker_text():
    data = {
        "segments": [
            {"start": 0.0, "end": 2.0, "speaker": "Speaker 1", "text": "Hello."},
            {"start": 65.5, "end": 68.0, "speaker": "Speaker 2", "text": "Hi there."},
        ]
    }
    out = format_transcript(data)
    assert "[00:00] Speaker 1: Hello." in out
    assert "[01:05] Speaker 2: Hi there." in out


def test_format_transcript_skips_empty_text():
    data = {
        "segments": [
            {"start": 0.0, "text": "   ", "speaker": "Speaker 1"},
            {"start": 5.0, "text": "real", "speaker": "Speaker 1"},
        ]
    }
    out = format_transcript(data)
    assert "real" in out
    assert out.count("Speaker 1") == 1


# --- _parse_fact_array ---

def test_parse_bare_array():
    text = '[{"text": "A fact.", "category": "personal", "confidence": 0.9}]'
    assert _parse_fact_array(text) == [
        {"text": "A fact.", "category": "personal", "confidence": 0.9}
    ]


def test_parse_code_fenced_array():
    text = '```json\n[{"text": "fact", "category": "personal", "confidence": 0.9}]\n```'
    result = _parse_fact_array(text)
    assert len(result) == 1
    assert result[0]["text"] == "fact"


def test_parse_object_with_facts_key():
    text = '{"facts": [{"text": "wrapped fact", "category": "context", "confidence": 0.8}]}'
    result = _parse_fact_array(text)
    assert len(result) == 1
    assert result[0]["text"] == "wrapped fact"


def test_parse_object_with_items_key():
    text = '{"items": [{"text": "item", "category": "context", "confidence": 0.8}]}'
    result = _parse_fact_array(text)
    assert len(result) == 1


def test_parse_handles_leading_trailing_prose():
    text = 'Here are the facts:\n[{"text": "fact", "category": "personal", "confidence": 0.9}]\nThanks!'
    result = _parse_fact_array(text)
    assert len(result) == 1


def test_parse_invalid_returns_empty():
    assert _parse_fact_array("not json at all") == []
    assert _parse_fact_array("") == []
    assert _parse_fact_array("[ broken json") == []


# --- _normalize_facts ---

def test_normalize_filters_low_confidence():
    raw = [
        {"text": "High conf", "category": "personal", "confidence": 0.9},
        {"text": "Low conf", "category": "personal", "confidence": 0.4},
    ]
    cleaned = _normalize_facts(raw)
    assert len(cleaned) == 1
    assert cleaned[0]["text"] == "High conf"


def test_normalize_filters_invalid_category():
    raw = [
        {"text": "Good", "category": "personal", "confidence": 0.9},
        {"text": "Bad", "category": "gossip", "confidence": 0.9},
    ]
    cleaned = _normalize_facts(raw)
    assert len(cleaned) == 1
    assert cleaned[0]["text"] == "Good"


def test_normalize_drops_empty_text():
    raw = [
        {"text": "  ", "category": "personal", "confidence": 0.9},
        {"text": "real", "category": "personal", "confidence": 0.9},
    ]
    cleaned = _normalize_facts(raw)
    assert len(cleaned) == 1


def test_normalize_coerces_segment_times():
    raw = [{
        "text": "ok", "category": "personal", "confidence": 0.9,
        "segment_start": "12.5", "segment_end": "18.0",
    }]
    cleaned = _normalize_facts(raw)
    assert cleaned[0]["segment_start"] == 12.5
    assert cleaned[0]["segment_end"] == 18.0


def test_normalize_handles_bad_segment_times_gracefully():
    raw = [{
        "text": "ok", "category": "personal", "confidence": 0.9,
        "segment_start": "nope",
    }]
    cleaned = _normalize_facts(raw)
    assert cleaned[0]["segment_start"] is None


def test_normalize_preserves_speaker_label():
    raw = [{
        "text": "ok", "category": "personal",
        "confidence": 0.9, "speaker_label": "Speaker 2",
    }]
    cleaned = _normalize_facts(raw)
    assert cleaned[0]["speaker_label"] == "Speaker 2"


def test_valid_categories_constant():
    expected = {"personal", "professional", "opinion", "commitment", "preference", "context"}
    assert VALID_CATEGORIES == expected


# --- extract_facts (mocked) ---

@pytest.mark.asyncio
async def test_extract_facts_short_transcript_skips_api():
    data = {"segments": [{"start": 0, "end": 1, "text": "too short", "speaker": "Speaker 1"}]}
    result = await extract_facts(data, {"provider": "api", "api_key": "k"})
    assert result == []


@pytest.mark.asyncio
async def test_extract_facts_empty_segments():
    result = await extract_facts({"segments": []}, {"provider": "api"})
    assert result == []


@pytest.mark.asyncio
async def test_extract_facts_calls_anthropic(monkeypatch):
    calls = {}

    async def fake_anthropic(cfg, prompt, timeout=60.0):
        calls["cfg"] = cfg
        calls["prompt"] = prompt
        return '[{"text": "mocked fact", "category": "professional", "confidence": 0.9, "speaker_label": "Speaker 1"}]'

    from transcriber import extraction
    monkeypatch.setattr(extraction, "_extract_via_anthropic", fake_anthropic)

    # Build a transcript with enough words
    segments = [
        {"start": float(i), "end": float(i + 1), "text": "word " * 15, "speaker": "Speaker 1"}
        for i in range(10)
    ]
    data = {"segments": segments}

    result = await extract_facts(data, {"provider": "api", "api_key": "k"})
    assert len(result) == 1
    assert result[0]["text"] == "mocked fact"
    assert result[0]["category"] == "professional"
    assert "[00:00]" in calls["prompt"]


@pytest.mark.asyncio
async def test_extract_facts_calls_ollama(monkeypatch):
    async def fake_ollama(cfg, prompt, timeout=120.0):
        return '{"facts": [{"text": "ollama fact", "category": "context", "confidence": 0.9}]}'

    from transcriber import extraction
    monkeypatch.setattr(extraction, "_extract_via_ollama", fake_ollama)

    segments = [
        {"start": float(i), "end": float(i + 1), "text": "word " * 15, "speaker": "Speaker 1"}
        for i in range(10)
    ]
    result = await extract_facts({"segments": segments}, {"provider": "ollama", "ollama_model": "llama3"})
    assert len(result) == 1
    assert result[0]["text"] == "ollama fact"


@pytest.mark.asyncio
async def test_extract_facts_no_provider_raises():
    segments = [
        {"start": float(i), "end": float(i + 1), "text": "word " * 15, "speaker": "Speaker 1"}
        for i in range(10)
    ]
    with pytest.raises(RuntimeError, match="No AI provider"):
        await extract_facts({"segments": segments}, {})


@pytest.mark.asyncio
async def test_extract_facts_handles_malformed_json(monkeypatch):
    async def fake_anthropic(cfg, prompt, timeout=60.0):
        return "The AI refused to comply. Here is nothing useful."

    from transcriber import extraction
    monkeypatch.setattr(extraction, "_extract_via_anthropic", fake_anthropic)

    segments = [
        {"start": float(i), "end": float(i + 1), "text": "word " * 15, "speaker": "Speaker 1"}
        for i in range(10)
    ]
    result = await extract_facts({"segments": segments}, {"provider": "api"})
    assert result == []
