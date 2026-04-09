"""
Test fixtures for Aurem.

Mocks heavy dependencies (PyAudioWPatch, faster-whisper, pyannote) so tests
run in WSL without GPU or Windows audio drivers.
"""

import json
import sys
import time
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Mock heavy native modules BEFORE importing any app code
# ---------------------------------------------------------------------------

# PyAudioWPatch (Windows-only WASAPI)
_pyaudio_mock = types.ModuleType("pyaudiowpatch")
_pyaudio_mock.PyAudio = MagicMock
_pyaudio_mock.paFloat32 = 1
_pyaudio_mock.paContinue = 0
_pyaudio_mock.paWASAPI = 0
sys.modules["pyaudiowpatch"] = _pyaudio_mock

# faster-whisper
_fw_mod = types.ModuleType("faster_whisper")
_fw_mod.WhisperModel = MagicMock
sys.modules["faster_whisper"] = _fw_mod

# pyannote.audio (optional but import-time safe)
if "pyannote" not in sys.modules:
    sys.modules["pyannote"] = types.ModuleType("pyannote")
if "pyannote.audio" not in sys.modules:
    _pa_mod = types.ModuleType("pyannote.audio")
    _pa_mod.Pipeline = MagicMock
    _pa_mod.Inference = MagicMock
    sys.modules["pyannote.audio"] = _pa_mod

# torch stub (just enough for the code that references it)
if "torch" not in sys.modules:
    _torch = types.ModuleType("torch")
    _torch.tensor = MagicMock(return_value=MagicMock(unsqueeze=MagicMock(return_value="fake_waveform")))
    _torch.float32 = "float32"
    _torch.Tensor = type("Tensor", (), {})
    _torch.device = MagicMock
    _torch.cuda = MagicMock()
    _torch.cuda.is_available = MagicMock(return_value=False)
    sys.modules["torch"] = _torch

# soundfile may not be installed in test env
if "soundfile" not in sys.modules:
    _sf = types.ModuleType("soundfile")
    _sf.read = MagicMock(return_value=(np.zeros(16000, dtype=np.float32), 16000))
    _sf.write = MagicMock()
    sys.modules["soundfile"] = _sf

# ---------------------------------------------------------------------------
# Now safe to import app code
# ---------------------------------------------------------------------------

from server.app import app, pipeline, file_jobs
from transcriber.pipeline import TranscriptionPipeline

from fastapi.testclient import TestClient


FAKE_SEGMENTS = [
    {
        "start": 0.0,
        "end": 2.5,
        "text": "Hello, this is a test.",
        "speaker": "Speaker",
        "words": [
            {"start": 0.0, "end": 0.5, "word": "Hello,", "probability": 0.99},
            {"start": 0.6, "end": 0.9, "word": " this", "probability": 0.98},
            {"start": 1.0, "end": 1.2, "word": " is", "probability": 0.99},
            {"start": 1.3, "end": 1.5, "word": " a", "probability": 0.99},
            {"start": 1.6, "end": 2.5, "word": " test.", "probability": 0.97},
        ],
    },
    {
        "start": 3.0,
        "end": 5.0,
        "text": "Second segment here.",
        "speaker": "Speaker",
        "words": [
            {"start": 3.0, "end": 3.5, "word": "Second", "probability": 0.95},
            {"start": 3.6, "end": 4.2, "word": " segment", "probability": 0.96},
            {"start": 4.3, "end": 5.0, "word": " here.", "probability": 0.94},
        ],
    },
]


@pytest.fixture(autouse=True)
def isolated_sessions(tmp_path, monkeypatch):
    """Point SESSIONS_DIR to a temp dir so tests don't touch real data."""
    import config
    monkeypatch.setattr(config, "SESSIONS_DIR", tmp_path / "sessions")
    (tmp_path / "sessions").mkdir()
    return tmp_path / "sessions"


@pytest.fixture(autouse=True)
def reset_pipeline():
    """Reset pipeline state between tests."""
    pipeline.session = None
    pipeline.capture = None
    pipeline.websocket_clients = set()
    pipeline._prev_words = []
    pipeline._prev_words_by_source = {}
    pipeline._dual_source_mode = False
    pipeline.available_devices = [
        {"index": 1, "name": "Test Mic", "type": "microphone", "channels": 1},
        {"index": 2, "name": "Test Loopback", "type": "loopback", "channels": 2},
    ]
    file_jobs.clear()
    yield
    # Clean up any lingering session
    pipeline.session = None
    pipeline.capture = None


@pytest.fixture
def mock_transcribe(monkeypatch):
    """Mock transcriber to return fake segments without loading Whisper."""
    def fake_transcribe(audio, offset_seconds=0.0, language="en"):
        import copy
        segs = copy.deepcopy(FAKE_SEGMENTS)
        for s in segs:
            s["start"] += offset_seconds
            s["end"] += offset_seconds
            for w in s.get("words", []):
                w["start"] += offset_seconds
                w["end"] += offset_seconds
        return segs

    monkeypatch.setattr(pipeline.transcriber, "model", MagicMock())
    monkeypatch.setattr(pipeline.transcriber, "transcribe", fake_transcribe)
    return fake_transcribe


@pytest.fixture
def mock_diarizer(monkeypatch):
    """Mock diarizer — pipeline is None (disabled) by default."""
    monkeypatch.setattr(pipeline.diarizer, "pipeline", None)
    monkeypatch.setattr(pipeline.diarizer, "embedding_model", None)
    return pipeline.diarizer


@pytest.fixture
def mock_audio_capture(monkeypatch):
    """Mock AudioCapture so start/stop don't touch real audio hardware."""
    from capture.audio_capture import AudioCapture

    def fake_init(self, on_chunk_ready, devices, all_device_info):
        self.on_chunk_ready = on_chunk_ready
        self._selected_devices = devices
        self._streams = []
        self._stop_event = MagicMock()

    monkeypatch.setattr(AudioCapture, "__init__", fake_init)
    monkeypatch.setattr(AudioCapture, "start", lambda self: None)
    monkeypatch.setattr(AudioCapture, "stop", lambda self: None)


@pytest.fixture
def client(mock_audio_capture, mock_diarizer):
    """Synchronous test client for FastAPI."""
    return TestClient(app)


@pytest.fixture
def session_on_disk(isolated_sessions):
    """Create a saved session on disk for testing reads/exports."""
    session_id = "20260101_120000"
    session_dir = isolated_sessions / session_id
    session_dir.mkdir()

    transcript = {
        "id": session_id,
        "title": "Test Session",
        "started_at": "2026-01-01T12:00:00",
        "ended_at": "2026-01-01T12:05:00",
        "duration": 300.0,
        "segments": FAKE_SEGMENTS,
    }
    (session_dir / "transcript.json").write_text(json.dumps(transcript, indent=2))
    return session_id
