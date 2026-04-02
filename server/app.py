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
from transcribe_file import (
    load_audio, run_diarization, merge_adjacent_segments,
    format_duration, format_timestamp as tf_format_timestamp,
    format_srt_time as tf_format_srt_time,
)

app = FastAPI(title="Meeting Transcriber")
pipeline = TranscriptionPipeline()

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
        job["progress"] = f"Transcription complete ({len(segments)} segments)"

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
                job["progress"] = f"Diarization complete ({n} speakers)"
            except Exception as e:
                job["progress"] = f"Diarization failed ({e}), transcript only"

        job["progress"] = "Merging segments..."
        segments = merge_adjacent_segments(segments)

        job["result"] = {
            "source_file": filename,
            "duration": round(duration, 1),
            "language": language,
            "num_speakers": diarization_info.get("num_speakers", 0),
            "speakers": diarization_info.get("speakers", {}),
            "segments": segments,
        }
        job["num_speakers"] = diarization_info.get("num_speakers", 0)
        job["speakers"] = diarization_info.get("speakers", {})
        job["status"] = "completed"
        job["progress"] = "Done"

    except Exception as e:
        job["status"] = "failed"
        job["progress"] = f"Error: {e}"
    finally:
        if os.path.exists(filepath):
            os.unlink(filepath)


@app.post("/api/transcribe-file")
async def transcribe_file_upload(
    file: UploadFile = File(...),
    num_speakers: int | None = Form(None),
    label_speakers: bool = Form(True),
    language: str = Form("en"),
):
    import tempfile

    job_id = str(uuid.uuid4())[:8]
    suffix = Path(file.filename or "upload").suffix or ".wav"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False, dir=str(config.BASE_DIR)) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    file_jobs[job_id] = {
        "job_id": job_id,
        "filename": file.filename,
        "status": "processing",
        "progress": "Queued",
        "duration": None,
        "num_speakers": None,
        "speakers": {},
        "result": None,
    }

    thread = threading.Thread(
        target=_process_file_job,
        args=(job_id, tmp_path, file.filename, num_speakers, label_speakers, language),
        daemon=True,
    )
    thread.start()

    return JSONResponse({
        "job_id": job_id,
        "status": "processing",
        "filename": file.filename,
    })


@app.get("/api/transcribe-file/{job_id}/status")
async def transcribe_file_status(job_id: str):
    job = file_jobs.get(job_id)
    if not job:
        return JSONResponse({"error": "Job not found"}, status_code=404)

    resp = {
        "job_id": job["job_id"],
        "status": job["status"],
        "progress": job["progress"],
        "filename": job["filename"],
    }
    if job["status"] == "completed":
        resp["duration"] = job["duration"]
        resp["num_speakers"] = job["num_speakers"]
        resp["speakers"] = job["speakers"]
    return JSONResponse(resp)


@app.get("/api/transcribe-file/{job_id}/result")
async def transcribe_file_result(job_id: str, format: str = "json"):
    job = file_jobs.get(job_id)
    if not job:
        return JSONResponse({"error": "Job not found"}, status_code=404)
    if job["status"] != "completed":
        return JSONResponse({"error": "Job not completed yet"}, status_code=400)

    result = job["result"]
    segments = result["segments"]
    filename = Path(job["filename"]).stem

    if format == "txt":
        lines = []
        for seg in segments:
            ts = tf_format_timestamp(seg["start"])
            speaker = seg.get("speaker", "Speaker")
            lines.append(f"[{ts}] {speaker}: {seg['text']}")
        return PlainTextResponse(
            "\n".join(lines),
            headers={"Content-Disposition": f"attachment; filename={filename}_transcript.txt"},
        )

    elif format == "srt":
        srt_lines = []
        for i, seg in enumerate(segments, 1):
            start = tf_format_srt_time(seg["start"])
            end = tf_format_srt_time(seg["end"])
            speaker = seg.get("speaker", "Speaker")
            srt_lines.append(str(i))
            srt_lines.append(f"{start} --> {end}")
            srt_lines.append(f"{speaker}: {seg['text']}")
            srt_lines.append("")
        return PlainTextResponse(
            "\n".join(srt_lines),
            headers={"Content-Disposition": f"attachment; filename={filename}_transcript.srt"},
        )

    return JSONResponse(result)


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
