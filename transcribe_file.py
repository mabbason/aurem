"""
Offline audio file transcription with speaker diarization.
Designed for podcast episodes but handles any number of speakers.

Usage:
    python transcribe_file.py -i episode.mp3 --label-speakers --txt
    python transcribe_file.py -i panel.mp3 -n 3 --txt --srt
    python transcribe_file.py -i lecture.wav --no-diarization --txt
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf

import config
from transcriber.transcription import Transcriber
from transcriber.diarization import Diarizer


def load_audio(filepath: str, target_sr: int = 16000) -> np.ndarray:
    """Load any audio file to 16kHz mono float32 numpy array using ffmpeg."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Audio file not found: {filepath}")

    # Try soundfile first (handles WAV, FLAC natively)
    try:
        audio, sr = sf.read(str(filepath), dtype="float32")
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)  # stereo to mono
        if sr != target_sr:
            # Resample via ffmpeg
            raise ValueError("needs resampling")
        return audio
    except Exception:
        pass

    # Fall back to ffmpeg for MP3, M4A, etc.
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        result = subprocess.run(
            [
                "ffmpeg", "-i", str(filepath),
                "-ar", str(target_sr),
                "-ac", "1",
                "-f", "wav",
                "-y", tmp_path,
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {result.stderr[:500]}")

        audio, sr = sf.read(tmp_path, dtype="float32")
        return audio
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def format_duration(seconds: float) -> str:
    """Format seconds as Xh Ym Zs or Ym Zs."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    return f"{m}m {s:02d}s"


def format_timestamp(seconds: float) -> str:
    """Format as HH:MM:SS.d"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    d = int((seconds % 1) * 10)
    return f"{h:02d}:{m:02d}:{s:02d}.{d}"


def format_srt_time(seconds: float) -> str:
    """Format as HH:MM:SS,mmm for SRT."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def run_diarization(diarizer: Diarizer, audio: np.ndarray, segments: list[dict],
                    num_speakers: int | None, label_speakers: bool) -> dict:
    """Run pyannote diarization and assign speaker labels to segments."""
    import torch

    waveform = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
    input_data = {"waveform": waveform, "sample_rate": config.AUDIO_SAMPLE_RATE}

    # Pass num_speakers hint if provided
    kwargs = {}
    if num_speakers is not None:
        kwargs["num_speakers"] = num_speakers

    result = diarizer.pipeline(input_data, **kwargs)

    # pyannote 4.0 returns DiarizeOutput dataclass
    if hasattr(result, "speaker_diarization"):
        annotation = result.speaker_diarization
    else:
        annotation = result

    # Build speaker timeline
    speaker_timeline = []
    label_map = {}
    next_id = 1
    for turn, _, label in annotation.itertracks(yield_label=True):
        speaker_timeline.append({"start": turn.start, "end": turn.end, "label": label})
        if label not in label_map:
            label_map[label] = f"Speaker {next_id}"
            next_id += 1

    num_found = len(label_map)

    # Assign speakers to segments by overlap
    for seg in segments:
        best_overlap = 0
        best_label = "Speaker"
        for t in speaker_timeline:
            overlap_start = max(seg["start"], t["start"])
            overlap_end = min(seg["end"], t["end"])
            overlap = max(0, overlap_end - overlap_start)
            if overlap > best_overlap:
                best_overlap = overlap
                best_label = label_map[t["label"]]
        seg["speaker"] = best_label

    # Calculate speaking time per speaker
    speaker_times = {}
    for seg in segments:
        spk = seg["speaker"]
        dur = seg["end"] - seg["start"]
        speaker_times[spk] = speaker_times.get(spk, 0) + dur

    # Host/Guest labeling for 2-speaker podcasts
    speakers_info = {}
    if label_speakers and num_found == 2:
        sorted_speakers = sorted(speaker_times.items(), key=lambda x: x[1])
        host_key = sorted_speakers[0][0]  # Less speaking time = host
        guest_key = sorted_speakers[1][0]  # More speaking time = guest

        rename_map = {host_key: "Host", guest_key: "Guest"}
        for seg in segments:
            if seg["speaker"] in rename_map:
                seg["speaker"] = rename_map[seg["speaker"]]

        for old_key, new_label in rename_map.items():
            speakers_info[new_label] = {
                "label": new_label,
                "total_speaking_time": round(speaker_times[old_key], 1),
            }
    else:
        for spk, total in sorted(speaker_times.items(), key=lambda x: x[0]):
            speakers_info[spk] = {
                "label": spk,
                "total_speaking_time": round(total, 1),
            }

    return {
        "num_speakers": num_found,
        "speakers": speakers_info,
    }


def merge_adjacent_segments(segments: list[dict]) -> list[dict]:
    """Merge consecutive segments with the same speaker into longer passages."""
    if not segments:
        return segments

    merged = [dict(segments[0])]
    for seg in segments[1:]:
        prev = merged[-1]
        if seg["speaker"] == prev["speaker"]:
            prev["end"] = seg["end"]
            prev["text"] = prev["text"] + " " + seg["text"]
            prev_words = prev.get("words", [])
            seg_words = seg.get("words", [])
            if prev_words or seg_words:
                prev["words"] = prev_words + seg_words
        else:
            merged.append(dict(seg))

    return merged


def save_json(output_path: Path, source_file: str, duration: float,
              language: str, segments: list[dict], diarization_info: dict):
    """Save transcript as JSON."""
    data = {
        "source_file": source_file,
        "duration": round(duration, 1),
        "language": language,
        "num_speakers": diarization_info.get("num_speakers", 0),
        "speakers": diarization_info.get("speakers", {}),
        "segments": segments,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_txt(output_path: Path, segments: list[dict]):
    """Save transcript as plain text."""
    lines = []
    for seg in segments:
        ts = format_timestamp(seg["start"])
        speaker = seg.get("speaker", "Speaker")
        lines.append(f"[{ts}] {speaker}: {seg['text']}")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def save_srt(output_path: Path, segments: list[dict]):
    """Save transcript as SRT subtitles."""
    lines = []
    for i, seg in enumerate(segments, 1):
        start = format_srt_time(seg["start"])
        end = format_srt_time(seg["end"])
        speaker = seg.get("speaker", "Speaker")
        lines.append(str(i))
        lines.append(f"{start} --> {end}")
        lines.append(f"{speaker}: {seg['text']}")
        lines.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio files with speaker diarization."
    )
    parser.add_argument("-i", "--input", required=True, help="Path to audio file")
    parser.add_argument("-o", "--output", help="Output JSON path (default: next to input)")
    parser.add_argument("-n", "--num-speakers", type=int, default=None,
                        help="Expected number of speakers (default: auto-detect)")
    parser.add_argument("-l", "--language", default="en", help="Language code (default: en)")
    parser.add_argument("--model", default=None, help="Override Whisper model")
    parser.add_argument("--label-speakers", action="store_true",
                        help="Label 2-speaker podcasts as Host/Guest")
    parser.add_argument("--txt", action="store_true", help="Also output plain text")
    parser.add_argument("--srt", action="store_true", help="Also output SRT subtitles")
    parser.add_argument("--no-diarization", action="store_true", help="Skip speaker diarization")

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: File not found: {input_path}")
        sys.exit(1)

    # Determine output paths
    if args.output:
        json_path = Path(args.output)
    else:
        json_path = input_path.with_name(f"{input_path.stem}_transcript.json")

    txt_path = json_path.with_suffix(".txt") if args.txt else None
    srt_path = json_path.with_suffix(".srt") if args.srt else None

    # --- Load audio ---
    print(f"Loading audio: {input_path.name}", end="", flush=True)
    t0 = time.time()
    audio = load_audio(str(input_path))
    duration = len(audio) / config.AUDIO_SAMPLE_RATE
    print(f" ({format_duration(duration)}) [{time.time() - t0:.1f}s]")

    # --- Transcribe ---
    if args.model:
        config.WHISPER_MODEL = args.model

    transcriber = Transcriber()
    print(f"Loading Whisper model '{config.WHISPER_MODEL}' on {config.WHISPER_DEVICE}...")
    transcriber.load_model()

    print("Transcribing... (this may take a few minutes)")
    t0 = time.time()
    segments = transcriber.transcribe(audio, offset_seconds=0.0, language=args.language)
    print(f"Transcription complete: {len(segments)} segments [{time.time() - t0:.1f}s]")

    # Add default speaker label
    for seg in segments:
        seg["speaker"] = "Speaker"

    # --- Diarize ---
    diarization_info = {"num_speakers": 0, "speakers": {}}

    if not args.no_diarization:
        diarizer = Diarizer()
        print("Loading diarization pipeline...")
        diarizer.load_model()

        if diarizer.pipeline is not None:
            num_hint = args.num_speakers
            print(f"Running speaker diarization{f' (num_speakers={num_hint})' if num_hint else ''}...")
            t0 = time.time()
            try:
                diarization_info = run_diarization(
                    diarizer, audio, segments,
                    num_speakers=num_hint,
                    label_speakers=args.label_speakers,
                )
                n = diarization_info["num_speakers"]
                print(f"Diarization complete: {n} speaker(s) identified [{time.time() - t0:.1f}s]")
                for spk, info in diarization_info["speakers"].items():
                    t = info["total_speaking_time"]
                    print(f"  {spk}: {format_duration(t)} total speaking time")
            except Exception as e:
                print(f"Diarization failed ({e}), continuing with transcript only")
        else:
            print("Diarization not available (HF_TOKEN not set or model failed to load)")
            print("Continuing with transcript only")

    # --- Merge adjacent same-speaker segments ---
    pre_merge = len(segments)
    segments = merge_adjacent_segments(segments)
    if len(segments) < pre_merge:
        print(f"Merging adjacent same-speaker segments: {pre_merge} -> {len(segments)} segments")

    # --- Save outputs ---
    save_json(json_path, input_path.name, duration, args.language, segments, diarization_info)
    print(f"Saved: {json_path}")

    if txt_path:
        save_txt(txt_path, segments)
        print(f"Saved: {txt_path}")

    if srt_path:
        save_srt(srt_path, segments)
        print(f"Saved: {srt_path}")

    print("Done!")


if __name__ == "__main__":
    main()
