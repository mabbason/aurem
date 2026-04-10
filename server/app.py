"""
FastAPI web server with WebSocket support for real-time transcription display.
"""

import json
import os
import threading
import uuid
import httpx
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

import config
from server.ai_config import load_ai_config, save_ai_config, test_ai_connection, generate_summary, generate_lessons

from transcriber.pipeline import TranscriptionPipeline
from transcriber.transcription import Transcriber
from transcriber.diarization import Diarizer
from transcriber import db as person_db
from transcribe_file import (
    load_audio, run_diarization, merge_adjacent_segments,
    format_duration, format_timestamp as tf_format_timestamp,
    format_srt_time as tf_format_srt_time,
)

app = FastAPI(title="Aurem")
pipeline = TranscriptionPipeline()

# Initialize the relationship-intelligence DB (idempotent).
try:
    person_db.init_db()
except Exception as e:
    print(f"Warning: could not initialize person DB: {e}")

# --- File transcription job tracking ---
file_jobs: dict[str, dict] = {}
_file_transcriber: Transcriber | None = None
_file_diarizer: Diarizer | None = None
_models_lock = threading.Lock()

STATIC_DIR = Path(__file__).parent.parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/devices")
async def list_devices():
    default_indices = pipeline._default_devices()
    devices = []
    for d in pipeline.available_devices:
        devices.append({
            "index": d["index"],
            "name": d["name"],
            "type": d["type"],
            "default": d["index"] in default_indices,
        })
    return JSONResponse(devices)


@app.post("/api/session/start")
async def start_session(request: Request):
    device_indices = None
    try:
        body = await request.json()
        device_indices = body.get("devices")
    except Exception:
        pass
    result = pipeline.start_session(device_indices=device_indices)
    return JSONResponse(result)


@app.post("/api/session/stop")
async def stop_session():
    result = pipeline.stop_session()
    if result is None:
        return JSONResponse({"error": "No active session"}, status_code=400)
    return JSONResponse(result)


@app.get("/api/session/status")
async def session_status():
    if pipeline.session:
        capturing = pipeline.capture is not None and pipeline.capture.is_running
        return JSONResponse({
            "active": True,
            "id": pipeline.session["id"],
            "started_at": pipeline.session["started_at"],
            "segment_count": len(pipeline.session["segments"]),
            "audio_capturing": capturing,
        })
    return JSONResponse({"active": False, "audio_capturing": False})


@app.get("/api/sessions")
async def list_sessions():
    return JSONResponse(pipeline.get_sessions())


@app.patch("/api/sessions/{session_id}")
async def rename_session(session_id: str, request: Request):
    body = await request.json()
    title = body.get("title", "").strip()
    if not title:
        return JSONResponse({"error": "Title required"}, status_code=400)
    if pipeline.rename_session(session_id, title):
        return JSONResponse({"ok": True})
    return JSONResponse({"error": "Session not found"}, status_code=404)


@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    data = pipeline.get_session_transcript(session_id)
    if data is None:
        return JSONResponse({"error": "Session not found"}, status_code=404)
    return JSONResponse(data)


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    import shutil
    session_dir = config.SESSIONS_DIR / session_id
    if not session_dir.exists():
        return JSONResponse({"error": "Session not found"}, status_code=404)
    shutil.rmtree(session_dir)
    return JSONResponse({"deleted": session_id})


@app.get("/api/sessions/{session_id}/export/{fmt}")
async def export_session(session_id: str, fmt: str):
    data = pipeline.get_session_transcript(session_id)
    if data is None:
        return JSONResponse({"error": "Session not found"}, status_code=404)

    segments = data.get("segments", [])

    if fmt == "json":
        return JSONResponse(data)

    elif fmt == "txt":
        lines = []
        for seg in segments:
            ts = format_timestamp(seg["start"])
            speaker = seg.get("speaker", "Speaker")
            lines.append(f"[{ts}] {speaker}: {seg['text']}")
        return PlainTextResponse(
            "\n".join(lines),
            headers={"Content-Disposition": f"attachment; filename={session_id}.txt"},
        )

    elif fmt == "srt":
        srt_lines = []
        for i, seg in enumerate(segments, 1):
            start = format_srt_time(seg["start"])
            end = format_srt_time(seg["end"])
            speaker = seg.get("speaker", "Speaker")
            srt_lines.append(f"{i}")
            srt_lines.append(f"{start} --> {end}")
            srt_lines.append(f"{speaker}: {seg['text']}")
            srt_lines.append("")
        return PlainTextResponse(
            "\n".join(srt_lines),
            headers={"Content-Disposition": f"attachment; filename={session_id}.srt"},
        )

    return JSONResponse({"error": f"Unknown format: {fmt}"}, status_code=400)


@app.get("/api/ai-config")
async def get_ai_config():
    cfg = load_ai_config()
    # Never send the full API key to the frontend
    if cfg.get("api_key"):
        key = cfg["api_key"]
        cfg["api_key_preview"] = key[:8] + "..." + key[-4:] if len(key) > 12 else "***"
    cfg.pop("api_key", None)
    return JSONResponse(cfg)


@app.post("/api/ai-config")
async def set_ai_config(request: Request):
    body = await request.json()
    current = load_ai_config()

    # If api_key not provided or empty, keep existing
    if not body.get("api_key"):
        body["api_key"] = current.get("api_key", "")

    save_ai_config(body)
    return JSONResponse({"ok": True})


@app.post("/api/ai-config/test")
async def test_config():
    cfg = load_ai_config()
    result = await test_ai_connection(cfg)
    return JSONResponse(result)


@app.post("/api/sessions/{session_id}/export/summary")
async def export_summary(session_id: str):
    data = pipeline.get_session_transcript(session_id)
    if data is None:
        return JSONResponse({"error": "Session not found"}, status_code=404)

    cfg = load_ai_config()
    if not cfg.get("provider"):
        return JSONResponse({"error": "AI not configured"}, status_code=400)

    result = await generate_summary(cfg, data)
    if "error" in result:
        return JSONResponse(result, status_code=500)
    return JSONResponse(result)


@app.post("/api/sessions/{session_id}/export/lessons")
async def export_lessons(session_id: str):
    data = pipeline.get_session_transcript(session_id)
    if data is None:
        return JSONResponse({"error": "Session not found"}, status_code=404)

    cfg = load_ai_config()
    if not cfg.get("provider"):
        return JSONResponse({"error": "AI not configured"}, status_code=400)

    result = await generate_lessons(cfg, data)
    if "error" in result:
        return JSONResponse(result, status_code=500)
    return JSONResponse(result)


# --- Relationship intelligence: session speakers ---

@app.get("/api/sessions/{session_id}/speakers")
async def get_session_speakers(session_id: str):
    """Return unique speaker labels in a session alongside current mappings
    and the facts extracted per label."""
    data = pipeline.get_session_transcript(session_id)
    if data is None:
        return JSONResponse({"error": "Session not found"}, status_code=404)

    labels: list[str] = []
    seen: set[str] = set()
    for seg in data.get("segments", []):
        label = seg.get("speaker", "Speaker")
        if label not in seen:
            seen.add(label)
            labels.append(label)

    mappings = person_db.get_session_speaker_mappings(session_id)
    facts_by_label: dict[str, int] = {}
    for f in person_db.facts_for_session(session_id):
        lbl = f.get("speaker_label") or "?"
        facts_by_label[lbl] = facts_by_label.get(lbl, 0) + 1

    extraction = person_db.get_extraction_status(session_id)

    return JSONResponse({
        "session_id": session_id,
        "speakers": [
            {
                "label": lbl,
                "person_id": mappings.get(lbl),
                "fact_count": facts_by_label.get(lbl, 0),
            }
            for lbl in labels
        ],
        "extraction": extraction,
    })


@app.post("/api/sessions/{session_id}/speakers")
async def set_session_speakers(session_id: str, request: Request):
    """Upsert speaker-to-person mappings. Body: {"mappings": {"Speaker 1": 3}}.
    A null person_id clears the mapping for that label."""
    body = await request.json()
    mappings = body.get("mappings") or {}
    if not isinstance(mappings, dict):
        return JSONResponse({"error": "mappings must be an object"}, status_code=400)

    for label, pid in mappings.items():
        person_id: int | None
        if pid is None or pid == "":
            person_id = None
        else:
            try:
                person_id = int(pid)
            except (TypeError, ValueError):
                return JSONResponse(
                    {"error": f"Invalid person_id for {label}"}, status_code=400
                )
            if person_db.get_person(person_id) is None:
                return JSONResponse(
                    {"error": f"Person {person_id} not found"}, status_code=400
                )
        person_db.set_session_speaker_mapping(session_id, label, person_id)

    return JSONResponse({"ok": True})


# --- Relationship intelligence: people ---

@app.get("/api/people")
async def list_people_route():
    return JSONResponse(person_db.list_people())


@app.post("/api/people")
async def create_person_route(request: Request):
    body = await request.json()
    name = (body.get("name") or "").strip()
    if not name:
        return JSONResponse({"error": "name required"}, status_code=400)
    notes = (body.get("notes") or "").strip() or None
    pid = person_db.create_person(name, notes)
    return JSONResponse(person_db.get_person(pid))


@app.get("/api/people/{person_id}")
async def get_person_route(person_id: int):
    person = person_db.get_person(person_id)
    if person is None:
        return JSONResponse({"error": "Person not found"}, status_code=404)

    all_facts = person_db.facts_for_person(person_id)

    facts_by_category: dict[str, list[dict]] = {}
    for f in all_facts:
        cat = f.get("category", "context")
        facts_by_category.setdefault(cat, []).append(f)

    session_ids = sorted(
        {f["session_id"] for f in all_facts if f.get("session_id")},
        reverse=True,
    )
    meeting_history = []
    for sid in session_ids:
        session_data = pipeline.get_session_transcript(sid)
        if session_data is None:
            continue
        meeting_history.append({
            "session_id": sid,
            "title": session_data.get("title", sid),
            "started_at": session_data.get("started_at"),
            "duration": session_data.get("duration", 0),
        })

    return JSONResponse({
        "person": person,
        "facts_by_category": facts_by_category,
        "meeting_history": meeting_history,
        "total_facts": len(all_facts),
    })


@app.patch("/api/people/{person_id}")
async def update_person_route(person_id: int, request: Request):
    body = await request.json()
    if "notes" in body:
        notes = body.get("notes") or ""
        if not person_db.update_person_notes(person_id, notes):
            return JSONResponse({"error": "Person not found"}, status_code=404)
    return JSONResponse({"ok": True})


@app.delete("/api/people/{person_id}")
async def delete_person_route(person_id: int):
    if not person_db.delete_person(person_id):
        return JSONResponse({"error": "Person not found"}, status_code=404)
    return JSONResponse({"ok": True})


@app.post("/api/people/{person_id}/facts")
async def add_manual_fact(person_id: int, request: Request):
    """Manual fact entry. No session_id, no segment timestamps."""
    if person_db.get_person(person_id) is None:
        return JSONResponse({"error": "Person not found"}, status_code=404)

    body = await request.json()
    text = (body.get("text") or "").strip()
    category = body.get("category") or "context"
    if not text:
        return JSONResponse({"error": "text required"}, status_code=400)

    from transcriber.extraction import VALID_CATEGORIES
    if category not in VALID_CATEGORIES:
        return JSONResponse({"error": f"Invalid category: {category}"}, status_code=400)

    fact_id = person_db.insert_fact(
        session_id=None,
        speaker_label=None,
        text=text,
        category=category,
        source="manual",
        person_id=person_id,
    )
    return JSONResponse({"id": fact_id})


@app.delete("/api/facts/{fact_id}")
async def delete_fact_route(fact_id: int):
    if not person_db.delete_fact(fact_id):
        return JSONResponse({"error": "Fact not found"}, status_code=404)
    return JSONResponse({"ok": True})


@app.get("/api/diarization/status")
async def diarization_status():
    from transcriber.diarization import DIARIZATION_AVAILABLE
    return JSONResponse({
        "available": DIARIZATION_AVAILABLE,
        "hf_token_set": bool(config.HF_TOKEN),
    })


def _get_file_models() -> tuple[Transcriber, Diarizer]:
    """Lazy-load and cache models for file transcription."""
    global _file_transcriber, _file_diarizer
    with _models_lock:
        if _file_transcriber is None:
            _file_transcriber = Transcriber()
            _file_transcriber.load_model()
        if _file_diarizer is None:
            _file_diarizer = Diarizer()
            _file_diarizer.load_model()
    return _file_transcriber, _file_diarizer


def _process_file_job(job_id: str, filepath: str, filename: str,
                      num_speakers: int | None, label_speakers: bool, language: str):
    """Background worker for file transcription."""
    job = file_jobs[job_id]
    try:
        job["progress"] = "Loading audio..."
        audio = load_audio(filepath)
        duration = len(audio) / config.AUDIO_SAMPLE_RATE
        job["duration"] = round(duration, 1)
        job["progress"] = f"Audio loaded ({format_duration(duration)})"

        transcriber, diarizer = _get_file_models()

        job["progress"] = "Transcribing..."
        segments = transcriber.transcribe(audio, offset_seconds=0.0, language=language)
        for seg in segments:
            seg["speaker"] = "Speaker"
        job["progress"] = f"Transcription complete: {len(segments)} segments"

        # Diarization
        diarization_info = {"num_speakers": 0, "speakers": {}}
        if diarizer.pipeline is not None:
            job["progress"] = "Running speaker diarization..."
            try:
                diarization_info = run_diarization(
                    diarizer, audio, segments,
                    num_speakers=num_speakers,
                    label_speakers=label_speakers,
                )
                n = diarization_info["num_speakers"]
                job["progress"] = f"Diarization complete: {n} speaker(s)"
            except Exception as e:
                job["progress"] = f"Diarization failed ({e}), continuing without it"

        segments = merge_adjacent_segments(segments)
        for seg in segments:
            seg.pop("words", None)

        job["result"] = {
            "source_file": filename,
            "duration": round(duration, 1),
            "language": language,
            "num_speakers": diarization_info.get("num_speakers", 0),
            "speakers": diarization_info.get("speakers", {}),
            "segments": segments,
        }
        job["speakers"] = diarization_info.get("speakers", {})
        job["status"] = "completed"
        job["progress"] = "Done"

    except Exception as e:
        job["status"] = "failed"
        job["progress"] = f"Error: {e}"
    finally:
        try:
            os.unlink(filepath)
        except OSError as cleanup_err:
            print(f"Warning: could not delete temp file {filepath}: {cleanup_err}")


@app.post("/api/transcribe-file")
async def transcribe_file_upload(
    file: UploadFile = File(...),
    num_speakers: int | None = Form(None),
    label_speakers: bool = Form(True),
    language: str = Form("en"),
):
    import tempfile

    job_id = str(uuid.uuid4())
    suffix = Path(file.filename or "audio.mp3").suffix

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    file_jobs[job_id] = {
        "job_id": job_id,
        "status": "processing",
        "progress": "Queued",
        "filename": file.filename,
        "duration": None,
        "speakers": {},
        "result": None,
    }

    thread = threading.Thread(
        target=_process_file_job,
        args=(job_id, tmp_path, file.filename, num_speakers, label_speakers, language),
        daemon=True,
    )
    thread.start()

    return JSONResponse({"job_id": job_id, "status": "processing", "filename": file.filename})


@app.post("/api/transcribe-file-path")
async def transcribe_file_by_path(request: Request):
    """Start transcription from a local file path (no upload needed)."""
    body = await request.json()
    filepath = body.get("filepath")
    num_speakers = body.get("num_speakers")
    label_speakers = body.get("label_speakers", True)
    language = body.get("language", "en")

    if not filepath or not os.path.isfile(filepath):
        return JSONResponse({"error": f"File not found: {filepath}"}, status_code=400)

    job_id = str(uuid.uuid4())
    filename = os.path.basename(filepath)

    file_jobs[job_id] = {
        "job_id": job_id,
        "status": "processing",
        "progress": "Queued",
        "filename": filename,
        "duration": None,
        "speakers": {},
        "result": None,
    }

    def _process_path_job():
        job = file_jobs[job_id]
        try:
            job["progress"] = "Loading audio..."
            audio = load_audio(filepath)
            duration = len(audio) / config.AUDIO_SAMPLE_RATE
            job["duration"] = round(duration, 1)
            job["progress"] = f"Audio loaded ({format_duration(duration)})"

            transcriber, diarizer = _get_file_models()

            job["progress"] = "Transcribing..."
            segments = transcriber.transcribe(audio, offset_seconds=0.0, language=language)
            for seg in segments:
                seg["speaker"] = "Speaker"
            job["progress"] = f"Transcription complete: {len(segments)} segments"

            diarization_info = {"num_speakers": 0, "speakers": {}}
            if diarizer.pipeline is not None:
                job["progress"] = "Running speaker diarization..."
                try:
                    diarization_info = run_diarization(
                        diarizer, audio, segments,
                        num_speakers=num_speakers,
                        label_speakers=label_speakers,
                    )
                    n = diarization_info["num_speakers"]
                    job["progress"] = f"Diarization complete: {n} speaker(s)"
                except Exception as e:
                    job["progress"] = f"Diarization failed ({e}), continuing without it"

            segments = merge_adjacent_segments(segments)
            for seg in segments:
                seg.pop("words", None)

            job["result"] = {
                "source_file": filename,
                "duration": round(duration, 1),
                "language": language,
                "num_speakers": diarization_info.get("num_speakers", 0),
                "speakers": diarization_info.get("speakers", {}),
                "segments": segments,
            }
            job["speakers"] = diarization_info.get("speakers", {})
            job["status"] = "completed"
            job["progress"] = "Done"
        except Exception as e:
            job["status"] = "failed"
            job["progress"] = f"Error: {e}"

    thread = threading.Thread(target=_process_path_job, daemon=True)
    thread.start()

    return JSONResponse({"job_id": job_id, "status": "processing", "filename": filename})


@app.get("/api/transcribe-file/{job_id}/status")
async def transcribe_file_status(job_id: str):
    job = file_jobs.get(job_id)
    if job is None:
        return JSONResponse({"error": "Job not found"}, status_code=404)

    resp = {
        "job_id": job_id,
        "status": job["status"],
        "progress": job["progress"],
    }
    if job["status"] == "completed":
        resp["duration"] = job.get("duration")
        resp["num_speakers"] = job["result"]["num_speakers"] if job["result"] else 0
        resp["speakers"] = job.get("speakers", {})
    return JSONResponse(resp)


@app.get("/api/transcribe-file/{job_id}/result")
async def transcribe_file_result(job_id: str, format: str = "json"):
    job = file_jobs.get(job_id)
    if job is None:
        return JSONResponse({"error": "Job not found"}, status_code=404)
    if job["status"] != "completed" or job["result"] is None:
        return JSONResponse({"error": "Job not completed yet"}, status_code=400)

    result = job["result"]

    if format == "txt":
        lines = []
        for seg in result["segments"]:
            ts = tf_format_timestamp(seg["start"])
            speaker = seg.get("speaker", "Speaker")
            lines.append(f"[{ts}] {speaker}: {seg['text']}")
        return PlainTextResponse(
            "\n".join(lines),
            headers={"Content-Disposition": f"attachment; filename={job_id}.txt"},
        )

    elif format == "srt":
        srt_lines = []
        for i, seg in enumerate(result["segments"], 1):
            start = tf_format_srt_time(seg["start"])
            end = tf_format_srt_time(seg["end"])
            speaker = seg.get("speaker", "Speaker")
            srt_lines.append(f"{i}")
            srt_lines.append(f"{start} --> {end}")
            srt_lines.append(f"{speaker}: {seg['text']}")
            srt_lines.append("")
        return PlainTextResponse(
            "\n".join(srt_lines),
            headers={"Content-Disposition": f"attachment; filename={job_id}.srt"},
        )

    return JSONResponse(result)


@app.post("/api/transcribe-file/{job_id}/save")
async def transcribe_file_save(job_id: str, request: Request):
    """Save transcription results (JSON + TXT + SRT) to a directory on disk."""
    job = file_jobs.get(job_id)
    if job is None:
        return JSONResponse({"error": "Job not found"}, status_code=404)
    if job["status"] != "completed" or job["result"] is None:
        return JSONResponse({"error": "Job not completed yet"}, status_code=400)

    body = await request.json()
    output_dir = body.get("output_dir")
    basename = body.get("basename", job_id)

    if not output_dir:
        return JSONResponse({"error": "output_dir is required"}, status_code=400)

    os.makedirs(output_dir, exist_ok=True)
    result = job["result"]
    saved = []

    json_path = os.path.join(output_dir, f"{basename}_transcript.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    saved.append(json_path)

    txt_path = os.path.join(output_dir, f"{basename}_transcript.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        for seg in result["segments"]:
            ts = tf_format_timestamp(seg["start"])
            speaker = seg.get("speaker", "Speaker")
            f.write(f"[{ts}] {speaker}: {seg['text']}\n")
    saved.append(txt_path)

    srt_path = os.path.join(output_dir, f"{basename}_transcript.srt")
    with open(srt_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(result["segments"], 1):
            start = tf_format_srt_time(seg["start"])
            end = tf_format_srt_time(seg["end"])
            speaker = seg.get("speaker", "Speaker")
            f.write(f"{i}\n{start} --> {end}\n{speaker}: {seg['text']}\n\n")
    saved.append(srt_path)

    return JSONResponse({"saved": saved, "output_dir": output_dir})


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    pipeline.websocket_clients.add(ws)
    print(f"WebSocket client connected ({len(pipeline.websocket_clients)} total)")

    try:
        # Send current session state if active
        if pipeline.session:
            await ws.send_text(json.dumps({
                "type": "session_state",
                "data": {
                    "active": True,
                    "id": pipeline.session["id"],
                    "segments": [
                        {
                            "start": s["start"],
                            "end": s["end"],
                            "text": s["text"],
                            "speaker": s.get("speaker", "Speaker"),
                        }
                        for s in pipeline.session["segments"]
                    ],
                },
            }))

        # Keep connection alive, handle client messages
        while True:
            data = await ws.receive_text()
            msg = json.loads(data)
            if msg.get("type") == "ping":
                await ws.send_text(json.dumps({"type": "pong"}))

    except WebSocketDisconnect:
        pass
    finally:
        pipeline.websocket_clients.discard(ws)
        print(f"WebSocket client disconnected ({len(pipeline.websocket_clients)} total)")


def format_timestamp(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 10)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms}"


def format_srt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
