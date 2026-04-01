"""
AI configuration storage and LLM integration for summary/lessons exports.
"""

import json
import httpx

import config

DEFAULT_CONFIG = {
    "provider": "",       # "ollama" or "api"
    "api_key": "",
    "api_provider": "anthropic",  # "anthropic" or "openai"
    "api_model": "claude-sonnet-4-20250514",
    "ollama_url": "http://localhost:11434",
    "ollama_model": "",
}


def load_ai_config() -> dict:
    if config.AI_CONFIG_PATH.exists():
        with open(config.AI_CONFIG_PATH) as f:
            saved = json.load(f)
        merged = {**DEFAULT_CONFIG, **saved}
        return merged
    return dict(DEFAULT_CONFIG)


def save_ai_config(cfg: dict):
    allowed_keys = set(DEFAULT_CONFIG.keys())
    clean = {k: v for k, v in cfg.items() if k in allowed_keys}
    with open(config.AI_CONFIG_PATH, "w") as f:
        json.dump(clean, f, indent=2)


async def test_ai_connection(cfg: dict) -> dict:
    provider = cfg.get("provider", "")

    if provider == "ollama":
        return await _test_ollama(cfg)
    elif provider == "api":
        return await _test_api(cfg)
    return {"ok": False, "error": "No provider configured"}


async def _test_ollama(cfg: dict) -> dict:
    url = cfg.get("ollama_url", "http://localhost:11434")
    model = cfg.get("ollama_model", "")
    if not model:
        return {"ok": False, "error": "No Ollama model selected"}

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(f"{url}/api/generate", json={
                "model": model,
                "prompt": "Say 'ok' and nothing else.",
                "stream": False,
            })
            if resp.status_code == 200:
                return {"ok": True, "message": f"Connected to {model}"}
            return {"ok": False, "error": f"Ollama returned {resp.status_code}"}
    except httpx.ConnectError:
        return {"ok": False, "error": f"Cannot connect to Ollama at {url}"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


async def _test_api(cfg: dict) -> dict:
    api_key = cfg.get("api_key", "")
    api_provider = cfg.get("api_provider", "anthropic")

    if not api_key:
        return {"ok": False, "error": "No API key provided"}

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            if api_provider == "anthropic":
                resp = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": api_key,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    },
                    json={
                        "model": cfg.get("api_model", "claude-sonnet-4-20250514"),
                        "max_tokens": 16,
                        "messages": [{"role": "user", "content": "Say 'ok'"}],
                    },
                )
            else:  # openai
                resp = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": cfg.get("api_model", "gpt-4o-mini"),
                        "max_tokens": 16,
                        "messages": [{"role": "user", "content": "Say 'ok'"}],
                    },
                )

            if resp.status_code == 200:
                return {"ok": True, "message": "API key is valid"}
            elif resp.status_code == 401:
                return {"ok": False, "error": "Invalid API key"}
            else:
                body = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
                err = body.get("error", {}).get("message", f"HTTP {resp.status_code}")
                return {"ok": False, "error": err}
    except Exception as e:
        return {"ok": False, "error": str(e)}


LESSONS_PROMPT = """You are an expert educator. Given the meeting transcript below, extract the key lessons and knowledge shared. Reframe the discussion as structured learning material in markdown format.

## Structure

1. **Topic Overview** — One paragraph describing what the meeting covered and the main themes
2. **Key Lessons** — Numbered list of the most important concepts, insights, or techniques discussed. Each lesson should:
   - Have a clear, concise title
   - Include 2-3 sentences explaining the concept
   - Note who explained it (using speaker labels from the transcript)
3. **Terminology & Definitions** — Any domain-specific terms or jargon that were explained or used (if none, omit this section)
4. **Discussion Questions** — 3-5 questions that could be used to test understanding of the material
5. **Follow-up Resources** — Topics mentioned that warrant further study (if none, omit this section)

## Rules
- Focus on knowledge transfer, not meeting logistics
- Use speaker labels to attribute ideas and explanations
- Be concise — each lesson should be digestible in under a minute of reading
- Do not hallucinate information not in the transcript
- If the meeting had no educational content, say so honestly

## Transcript

{transcript}"""


SUMMARY_PROMPT = """You are a meeting summarizer. Given the transcript below, produce a structured summary in markdown format.

## Structure

1. **Meeting Summary** — 3-5 bullet points covering the key topics discussed
2. **Key Decisions** — Any decisions that were made (if none, omit this section)
3. **Action Items** — Tasks, follow-ups, or commitments mentioned, with the responsible person if identifiable. Format as a checklist.

## Rules
- Be concise — each bullet should be one sentence
- Use the speakers' names/labels as given in the transcript
- If no action items are mentioned, write "No action items identified"
- Do not hallucinate information not in the transcript

## Transcript

{transcript}"""


def _build_transcript_text(data: dict) -> str:
    lines = []
    for seg in data.get("segments", []):
        speaker = seg.get("speaker", "Speaker")
        lines.append(f"[{speaker}]: {seg['text']}")
    return "\n".join(lines)


async def generate_summary(cfg: dict, session_data: dict) -> dict:
    transcript_text = _build_transcript_text(session_data)
    if not transcript_text.strip():
        return {"error": "No transcript content to summarize"}

    prompt = SUMMARY_PROMPT.format(transcript=transcript_text)
    provider = cfg.get("provider", "")

    try:
        if provider == "ollama":
            return await _generate_ollama(cfg, prompt)
        elif provider == "api":
            return await _generate_api(cfg, prompt)
        return {"error": "No AI provider configured"}
    except Exception as e:
        return {"error": str(e)}


async def _generate_ollama(cfg: dict, prompt: str, key: str = "summary") -> dict:
    url = cfg.get("ollama_url", "http://localhost:11434")
    model = cfg.get("ollama_model", "")

    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(f"{url}/api/generate", json={
            "model": model,
            "prompt": prompt,
            "stream": False,
        })
        if resp.status_code != 200:
            return {"error": f"Ollama returned {resp.status_code}"}
        body = resp.json()
        return {key: body.get("response", ""), "model": model}


async def generate_lessons(cfg: dict, session_data: dict) -> dict:
    transcript_text = _build_transcript_text(session_data)
    if not transcript_text.strip():
        return {"error": "No transcript content to analyze"}

    prompt = LESSONS_PROMPT.format(transcript=transcript_text)
    provider = cfg.get("provider", "")

    try:
        if provider == "ollama":
            return await _generate_ollama(cfg, prompt, key="lessons")
        elif provider == "api":
            return await _generate_api(cfg, prompt, key="lessons")
        return {"error": "No AI provider configured"}
    except Exception as e:
        return {"error": str(e)}


async def _generate_api(cfg: dict, prompt: str, key: str = "summary") -> dict:
    api_key = cfg.get("api_key", "")
    api_provider = cfg.get("api_provider", "anthropic")
    model = cfg.get("api_model", "")

    async with httpx.AsyncClient(timeout=120) as client:
        if api_provider == "anthropic":
            resp = await client.post(
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
            if resp.status_code != 200:
                return {"error": f"Anthropic API returned {resp.status_code}"}
            body = resp.json()
            text = body["content"][0]["text"]
        else:  # openai
            resp = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "max_tokens": 4096,
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
            if resp.status_code != 200:
                return {"error": f"OpenAI API returned {resp.status_code}"}
            body = resp.json()
            text = body["choices"][0]["message"]["content"]

        return {key: text, "model": model}
