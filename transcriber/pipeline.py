"""
Real-time transcription pipeline.
Orchestrates audio capture, transcription, diarization, and WebSocket broadcasting.
Uses word-level dedup to merge overlapping chunk edges cleanly.
"""

import asyncio
import json
from datetime import datetime

import numpy as np
import soundfile as sf

import config
from capture.audio_capture import AudioCapture
from transcriber.transcription import Transcriber
from transcriber.diarization import Diarizer


class TranscriptionPipeline:
    def __init__(self):
        self.transcriber = Transcriber()
        self.diarizer = Diarizer()
        self.capture = None
        self.session = None
        self.websocket_clients = set()
        self._loop = None
        self._processing_lock = None
        self._prev_words = []  # Word-level output from previous chunk for dedup

    def load_models(self):
        self.transcriber.load_model()
        self.diarizer.load_model()

    def start_session(self) -> dict:
        now = datetime.now()
        session_id = now.strftime("%Y%m%d_%H%M%S")
        session_dir = config.SESSIONS_DIR / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        self.session = {
            "id": session_id,
            "started_at": now.isoformat(),
            "ended_at": None,
            "segments": [],
            "dir": session_dir,
        }

        self._loop = asyncio.get_event_loop()
        self._processing_lock = asyncio.Lock()
        self._prev_words = []

        self.capture = AudioCapture(on_chunk_ready=self._on_chunk_from_thread)
        self.capture.start()

        print(f"Session started: {session_id}")
        return {"id": session_id, "started_at": self.session["started_at"]}

    def stop_session(self) -> dict | None:
        if not self.session:
            return None

        if self.capture:
            self.capture.stop()
            self.capture = None

        self.session["ended_at"] = datetime.now().isoformat()
        self._save_transcript()

        result = {
            "id": self.session["id"],
            "started_at": self.session["started_at"],
            "ended_at": self.session["ended_at"],
            "segment_count": len(self.session["segments"]),
        }

        print(f"Session stopped: {self.session['id']} ({len(self.session['segments'])} segments)")
        self.session = None
        return result

    def _on_chunk_from_thread(self, audio: np.ndarray, chunk_index: int, offset: float):
        if self._loop and self.session:
            peak = np.max(np.abs(audio))
            dur = len(audio) / config.AUDIO_SAMPLE_RATE
            print(f"Chunk {chunk_index}: {dur:.1f}s, peak={peak:.4f}, offset={offset:.1f}s")
            future = asyncio.run_coroutine_threadsafe(
                self._process_chunk(audio, chunk_index, offset),
                self._loop,
            )
            future.add_done_callback(self._chunk_done_callback)

    @staticmethod
    def _chunk_done_callback(future):
        try:
            future.result()
        except Exception as e:
            print(f"Chunk processing error: {e}")
            import traceback
            traceback.print_exc()

    async def _process_chunk(self, audio: np.ndarray, chunk_index: int, offset: float):
        if not self.session:
            return

        async with self._processing_lock:
            chunk_path = self.session["dir"] / f"chunk_{chunk_index:04d}.wav"
            sf.write(str(chunk_path), audio, config.AUDIO_SAMPLE_RATE)

            peak = np.max(np.abs(audio))
            if peak < 0.001:
                return

            loop = asyncio.get_event_loop()

            segments = await loop.run_in_executor(
                None, self.transcriber.transcribe, audio, offset
            )

            if not segments:
                return

            # Collect all words from this chunk's segments
            all_words = []
            for seg in segments:
                all_words.extend(seg.get("words", []))

            # Deduplicate against previous chunk's trailing words
            deduped_words = self._dedup_words(all_words)

            if not deduped_words:
                return

            # Rebuild segments from deduped words
            merged_segments = self._words_to_segments(deduped_words, segments)

            # Diarize
            chunk_segments = []
            for seg in merged_segments:
                chunk_segments.append({
                    **seg,
                    "start": seg["start"] - offset,
                    "end": seg["end"] - offset,
                })

            diarized = await loop.run_in_executor(
                None, self.diarizer.diarize, audio, chunk_segments
            )

            for seg in diarized:
                seg["start"] = round(seg["start"] + offset, 1)
                seg["end"] = round(seg["end"] + offset, 1)
                self.session["segments"].append(seg)
                await self._broadcast(seg)

            # Store trailing words for next chunk's dedup
            self._prev_words = all_words[-20:] if all_words else []

    def _dedup_words(self, new_words: list[dict]) -> list[dict]:
        """Remove words from the start of new_words that overlap with previous chunk."""
        if not self._prev_words or not new_words:
            return new_words

        # Find where the new words stop overlapping with previous words.
        # Compare by normalized word text and approximate timestamp.
        prev_texts = [w["word"].strip().lower().rstrip(".,!?") for w in self._prev_words]

        # Find the best alignment: look for a sequence of new words that matches
        # a suffix of prev_words
        best_skip = 0

        for skip in range(min(len(new_words), 15)):
            new_word = new_words[skip]["word"].strip().lower().rstrip(".,!?")
            # Check if this word matches any of the last prev words
            if new_word in prev_texts[-10:]:
                # Check if subsequent words also match (sequence match)
                match_len = 0
                for j in range(skip, min(skip + 5, len(new_words))):
                    nw = new_words[j]["word"].strip().lower().rstrip(".,!?")
                    if nw in prev_texts:
                        match_len += 1
                    else:
                        break

                if match_len >= 2:
                    # Found an overlap sequence — skip these words
                    best_skip = skip + match_len

        if best_skip > 0:
            skipped = " ".join(w["word"].strip() for w in new_words[:best_skip])
            print(f"  Dedup: skipped {best_skip} overlapping words: '{skipped}'")

        return new_words[best_skip:]

    def _words_to_segments(self, words: list[dict], original_segments: list[dict]) -> list[dict]:
        """
        Rebuild segments from deduped words, preserving natural sentence boundaries
        from the original Whisper segmentation.
        """
        if not words:
            return []

        # Use original segment boundaries where they align with our word list
        segments = []
        word_idx = 0

        for orig_seg in original_segments:
            seg_words = []
            orig_seg_words = orig_seg.get("words", [])

            for ow in orig_seg_words:
                if word_idx >= len(words):
                    break
                # Match by timestamp proximity
                if abs(words[word_idx]["start"] - ow["start"]) < 0.3:
                    seg_words.append(words[word_idx])
                    word_idx += 1

            if seg_words:
                text = "".join(w["word"] for w in seg_words).strip()
                if text:
                    segments.append({
                        "start": seg_words[0]["start"],
                        "end": seg_words[-1]["end"],
                        "text": text,
                        "words": seg_words,
                    })

        # Any remaining words that didn't match original boundaries
        if word_idx < len(words):
            remaining = words[word_idx:]
            text = "".join(w["word"] for w in remaining).strip()
            if text:
                segments.append({
                    "start": remaining[0]["start"],
                    "end": remaining[-1]["end"],
                    "text": text,
                    "words": remaining,
                })

        return segments

    async def _broadcast(self, segment: dict):
        message = json.dumps({
            "type": "segment",
            "data": {
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"],
                "speaker": segment.get("speaker", "Speaker"),
            },
        })

        dead = set()
        for ws in self.websocket_clients:
            try:
                await ws.send_text(message)
            except Exception:
                dead.add(ws)
        self.websocket_clients -= dead

    def _save_transcript(self):
        if not self.session:
            return
        transcript_path = self.session["dir"] / "transcript.json"
        save_data = {
            "id": self.session["id"],
            "started_at": self.session["started_at"],
            "ended_at": self.session["ended_at"],
            "segments": self.session["segments"],
        }
        transcript_path.write_text(json.dumps(save_data, indent=2))

    def get_sessions(self) -> list[dict]:
        sessions = []
        if not config.SESSIONS_DIR.exists():
            return sessions

        for session_dir in sorted(config.SESSIONS_DIR.iterdir(), reverse=True):
            transcript = session_dir / "transcript.json"
            if transcript.exists():
                data = json.loads(transcript.read_text())
                sessions.append({
                    "id": data["id"],
                    "started_at": data["started_at"],
                    "ended_at": data["ended_at"],
                    "segment_count": len(data.get("segments", [])),
                })
        return sessions

    def get_session_transcript(self, session_id: str) -> dict | None:
        transcript = config.SESSIONS_DIR / session_id / "transcript.json"
        if not transcript.exists():
            return None
        return json.loads(transcript.read_text())
