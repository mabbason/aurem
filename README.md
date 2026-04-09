# Aurem

**Listen. Learn. Prepare.**

Aurem (Latin for *ear*) is a local-first relationship intelligence tool for Windows 11. It captures meeting audio, transcribes locally with Whisper on GPU, diarizes speakers, and extracts per-person facts into a local person graph. Before your next call, open a person's page and see everything you've ever discussed with them. Meetings are inputs. People are the product. All audio and ML stays on-device.

## Features

- **Verbatim transcription** with timestamps (0.1s precision)
- **Speaker diarization** — distinguishes speakers with consistent labels
- **Platform-agnostic** — captures system audio (works with Zoom, Meet, Teams, etc.)
- **Privacy-first** — all processing runs locally, no cloud APIs
- **Low-latency** — 5-second chunks for near real-time display
- **Export** — JSON, plain text, and SRT subtitle formats

## Architecture

Single Windows Python process:

```
Audio Capture (WASAPI loopback) → faster-whisper (CUDA) → pyannote diarization → Web UI (WebSocket)
```

## Requirements

- Windows 11
- Python 3.11+
- NVIDIA GPU recommended (CUDA for faster transcription)

## Setup

### 1. Install PyTorch with CUDA

```powershell
# For CUDA 12.4 (check your version with nvidia-smi)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Configure

```powershell
copy .env.example .env
# Edit .env — set WHISPER_MODEL, HF_TOKEN (for diarization), etc.
```

**For speaker diarization (optional):**
1. Create a free account at [huggingface.co](https://huggingface.co)
2. Get a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
3. Accept terms at [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
4. Set `HF_TOKEN=your_token` in `.env`

## Usage

### 1. Start the server

```powershell
python main.py
```

### 2. Open the web UI

Navigate to [http://localhost:8765](http://localhost:8765)

Click **Start Recording** to begin capturing and transcribing system audio.

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `WHISPER_MODEL` | `medium` | Whisper model size (tiny/base/small/medium/large-v3) |
| `WHISPER_DEVICE` | `cuda` | Device (cuda/cpu) |
| `WHISPER_COMPUTE_TYPE` | `int8_float32` | Quantization (int8_float32 for GTX, float16 for RTX 30+) |
| `AUDIO_CHUNK_SECONDS` | `5` | Seconds per processing chunk (lower = less latency) |
| `AUDIO_OVERLAP_SECONDS` | `0.5` | Overlap between chunks to avoid cutting words |
| `WEB_PORT` | `8765` | Web UI port |
| `HF_TOKEN` | (empty) | HuggingFace token for speaker diarization |

## Export Formats

- **JSON** — Full metadata including word-level timestamps and confidence scores
- **TXT** — Human-readable: `[HH:MM:SS.s] Speaker N: text`
- **SRT** — Standard subtitle format for video players
