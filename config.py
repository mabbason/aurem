import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent

# Whisper settings
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "large-v3-turbo")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cuda")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8_float32")

# Audio settings
AUDIO_SAMPLE_RATE = int(os.getenv("AUDIO_SAMPLE_RATE", "16000"))

# VAD-based chunking: split on silence pauses instead of fixed intervals
AUDIO_MIN_CHUNK_SECONDS = float(os.getenv("AUDIO_MIN_CHUNK_SECONDS", "1.5"))
AUDIO_MAX_CHUNK_SECONDS = float(os.getenv("AUDIO_MAX_CHUNK_SECONDS", "15"))
AUDIO_SILENCE_THRESHOLD = float(os.getenv("AUDIO_SILENCE_THRESHOLD", "0.005"))
AUDIO_SILENCE_DURATION_MS = int(os.getenv("AUDIO_SILENCE_DURATION_MS", "300"))
AUDIO_OVERLAP_SECONDS = float(os.getenv("AUDIO_OVERLAP_SECONDS", "1.0"))

# Web server
WEB_HOST = os.getenv("WEB_HOST", "0.0.0.0")
WEB_PORT = int(os.getenv("WEB_PORT", "8765"))

# Storage
SESSIONS_DIR = Path(os.getenv("SESSIONS_DIR", str(BASE_DIR / "sessions")))

# HuggingFace token for pyannote speaker diarization
HF_TOKEN = os.getenv("HF_TOKEN", "")

# AI config file (user-configurable via UI)
AI_CONFIG_PATH = BASE_DIR / "ai_config.json"
