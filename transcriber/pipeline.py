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
from capture.audio_capture import AudioCapture, discover_all_devices
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
        self._prev_words = []
        self._prev_words_by_source = {}
        self._dual_source_mode = False
        self.available_devices = []

    def load_models(self):
        self.transcriber.load_model()
        self.diarizer.load_model()

        print("Discovering audio devices...")
        self.available_devices = discover_all_devices()
        print()

    def start_session(self, device_indices: list[int] | None = None) -> dict:
        now = datetime.now()
        session_id = now.strftime("%Y%m%d_%H%M%S")
        session_dir = config.SESSIONS_DIR / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        # Use provided devices or default to all loopbacks + loudest mic
        if device_indices:
            selected = device_indices
        else:
            selected = self._default_devices()

        # Detect dual-source: both mic and loopback types present
        device_map = {d["index"]: d for d in self.available_devices}
        has_mic = any(device_map.get(i, {}).get("type") == "microphone" for i in selected)
        has_loopback = any(device_map.get(i, {}).get("type") == "loopback" for i in selected)
        self._dual_source_mode = has_mic and has_loopback

        self.session = {
            "id": session_id,
            "started_at": now.isoformat(),
            "ended_at": None,
            "segments": [],
            "dir": session_dir,
            "dual_source": self._dual_source_mode,
        }

        self._loop = asyncio.get_event_loop()
        self._processing_lock = asyncio.Lock()
        self._prev_words = []
        self._prev_words_by_source = {}
        self.diarizer.reset()

        self.capture = AudioCapture(
            on_chunk_ready=self._on_chunk_from_thread,
            devices=selected,
            all_device_info=self.available_devices,
        )
        self.capture.start()

        print(f"Session started: {session_id}")
        return {"id": session_id, "started_at": self.session["started_at"]}

    def _default_devices(self) -> list[int]:
        """Default selection: first loopback + first mic."""
        selected = []
        for d in self.available_devices:
            if d["type"] == "loopback":
                selected.append(d["index"])
                break
        for d in self.available_devices:
            if d["type"] == "microphone":
                selected.append(d["index"])
                break
        return selected

    def stop_session(self) -> dict | None:
        if not self.session:
            return None

        if self.capture:
            self.capture.stop()
            self.capture = None

        self.session["ended_at"] = datetime.now().isoformat()
        segment_count = len(self.session["segments"])

        if segment_count > 0:
            # Save transcript immediately so the session appears in the list
            self._save_transcript()
            # Run diarization in background so the stop response returns fast
            import threading
            session_ref = self.session
            threading.Thread(
                target=self._post_session_diarization_background,
                args=(session_ref,),
                daemon=True,
            ).start()
        else:
            import shutil
            session_dir = self.session["dir"]
            if session_dir.exists():
                shutil.rmtree(session_dir)

        result = {
            "id": self.session["id"],
            "started_at": self.session["started_at"],
            "ended_at": self.session["ended_at"],
            "segment_count": segment_count,
        }

        print(f"Session stopped: {self.session['id']} ({segment_count} segments)")
        self.session = None
        return result

    def _post_session_diarization_background(self, session: dict):
        """Run diarization then fact extraction in background, update the
        saved transcript with speaker labels."""
        try:
            self._run_post_session_diarization(session)
            # Re-save transcript with speaker labels
            transcript_path = session["dir"] / "transcript.json"
            if transcript_path.exists():
                data = json.loads(transcript_path.read_text())
                data["segments"] = session["segments"]
                transcript_path.write_text(json.dumps(data, indent=2))
                print(f"Transcript updated with speaker labels: {session['id']}")
        except Exception as e:
            print(f"Background diarization failed: {e}")
            import traceback
            traceback.print_exc()

        # Fact extraction runs after diarization so facts can be attached
        # to the final speaker labels.
        try:
            self._run_post_session_extraction(session)
        except Exception as e:
            print(f"Background extraction failed: {e}")
            import traceback
            traceback.print_exc()

    def _run_post_session_extraction(self, session: dict):
        """Extract facts from the session transcript and store them in SQLite.
        Runs after diarization so speaker labels are final. Failures are
        recorded in session_extractions but never block the session save."""
        import asyncio
        from server.ai_config import load_ai_config
        from transcriber import db
        from transcriber.extraction import extract_facts

        session_id = session["id"]

        cfg = load_ai_config()
        if not cfg.get("provider"):
            db.record_extraction(session_id, "skipped", "AI provider not configured")
            print(f"Extraction skipped for {session_id}: no AI provider configured")
            return

        # Reload from disk (diarization may have rewritten the speaker labels)
        transcript_path = session["dir"] / "transcript.json"
        if not transcript_path.exists():
            db.record_extraction(session_id, "skipped", "No transcript on disk")
            return
        data = json.loads(transcript_path.read_text())

        if db.session_has_facts(session_id):
            print(f"Extraction skipped for {session_id}: facts already present")
            return

        print(f"Running fact extraction for {session_id}...")
        try:
            facts = asyncio.run(extract_facts(data, cfg))
        except Exception as e:
            db.record_extraction(session_id, "error", str(e)[:500])
            raise

        inserted = db.insert_facts(facts, session_id=session_id)
        db.record_extraction(session_id, "ok")
        print(f"Extraction complete for {session_id}: {inserted} facts stored")

    def _run_post_session_diarization(self, session: dict):
        """Run diarization on the full session audio after recording stops."""
        if self.diarizer.pipeline is None:
            return

        segments = session["segments"]
        if not segments:
            return

        dual_source = session.get("dual_source", False)

        try:
            import glob
            import torch

            # In dual-source mode, only diarize system audio
            if dual_source:
                chunk_files = sorted(glob.glob(str(session["dir"] / "sys_chunk_*.wav")))
                target_segments = [s for s in segments if s.get("source") == "system"]
            else:
                chunk_files = sorted(glob.glob(str(session["dir"] / "chunk_*.wav")))
                target_segments = segments

            if not chunk_files or not target_segments:
                return

            all_audio = []
            for f in chunk_files:
                audio, _ = sf.read(f)
                all_audio.append(audio)
            full_audio = np.concatenate(all_audio)

            src_label = " (system audio only)" if dual_source else ""
            print(f"Post-session diarization{src_label}: {len(full_audio)/config.AUDIO_SAMPLE_RATE:.1f}s audio")

            waveform = torch.tensor(full_audio, dtype=torch.float32).unsqueeze(0)
            result = self.diarizer.pipeline(
                {"waveform": waveform, "sample_rate": config.AUDIO_SAMPLE_RATE}
            )

            # Extract speaker annotation (pyannote 4.0 vs 3.x)
            if hasattr(result, 'speaker_diarization'):
                annotation = result.speaker_diarization
            else:
                annotation = result

            # Build speaker timeline
            speaker_timeline = []
            for turn, _, label in annotation.itertracks(yield_label=True):
                speaker_timeline.append({
                    "start": turn.start,
                    "end": turn.end,
                    "label": label,
                })

            # Build consistent speaker label map
            # In dual-source mode, start at Speaker 2 (Speaker 1 = mic user)
            next_id = 2 if dual_source else 1
            label_map = {}
            for t in speaker_timeline:
                if t["label"] not in label_map:
                    label_map[t["label"]] = f"Speaker {next_id}"
                    next_id += 1

            # Assign speaker labels to target segments by overlap
            for seg in target_segments:
                best_overlap = 0
                best_label = seg.get("speaker", "Speaker")
                for t in speaker_timeline:
                    overlap_start = max(seg["start"], t["start"])
                    overlap_end = min(seg["end"], t["end"])
                    overlap = max(0, overlap_end - overlap_start)
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_label = label_map[t["label"]]
                seg["speaker"] = best_label

            speakers_found = len(label_map)
            mic_note = " (+ Speaker 1 from mic)" if dual_source else ""
            print(f"Post-session diarization complete: {speakers_found} speaker(s) identified{mic_note}")

            # Broadcast updated segments with speaker labels to connected clients
            if self._loop:
                import json
                message = json.dumps({
                    "type": "diarization_complete",
                    "data": {
                        "segments": [
                            {
                                "start": s["start"],
                                "end": s["end"],
                                "text": s["text"],
                                "speaker": s.get("speaker", "Speaker"),
                            }
                            for s in segments
                        ],
                    },
                })
                dead = set()
                for ws in self.websocket_clients:
                    try:
                        asyncio.run_coroutine_threadsafe(
                            ws.send_text(message), self._loop
                        )
                    except Exception:
                        dead.add(ws)
                self.websocket_clients -= dead

        except Exception as e:
            print(f"Post-session diarization error: {e}")
            import traceback
            traceback.print_exc()

    def _on_chunk_from_thread(self, audio: np.ndarray, chunk_index: int, offset: float, source: str | None = None):
        if self._loop and self.session:
            peak = np.max(np.abs(audio))
            dur = len(audio) / config.AUDIO_SAMPLE_RATE
            src_label = f" [{source}]" if source else ""
            print(f"Chunk {chunk_index}{src_label}: {dur:.1f}s, peak={peak:.4f}, offset={offset:.1f}s")
            future = asyncio.run_coroutine_threadsafe(
                self._process_chunk(audio, chunk_index, offset, source),
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

    async def _process_chunk(self, audio: np.ndarray, chunk_index: int, offset: float, source: str | None = None):
        if not self.session:
            return

        async with self._processing_lock:
            # Source-aware chunk filenames
            if self._dual_source_mode and source == "mic":
                chunk_path = self.session["dir"] / f"mic_chunk_{chunk_index:04d}.wav"
            elif self._dual_source_mode and source == "system":
                chunk_path = self.session["dir"] / f"sys_chunk_{chunk_index:04d}.wav"
            else:
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

            # Deduplicate against previous chunk's trailing words (per-source)
            if self._dual_source_mode and source:
                prev_words = self._prev_words_by_source.get(source, [])
            else:
                prev_words = self._prev_words
            deduped_words = self._dedup_words(all_words, prev_words)

            if not deduped_words:
                return

            # Rebuild segments from deduped words
            merged_segments = self._words_to_segments(deduped_words, segments)

            # Assign speaker labels based on mode and source
            if self._dual_source_mode:
                if source == "mic":
                    speaker_label = "Speaker 1"
                else:
                    speaker_label = "Speaker 2"
            else:
                speaker_label = "Speaker"

            for seg in merged_segments:
                seg["speaker"] = speaker_label
                if self._dual_source_mode and source:
                    seg["source"] = source
                self.session["segments"].append(seg)
                await self._broadcast(seg)

            # Store trailing words for next chunk's dedup (per-source)
            trailing = all_words[-20:] if all_words else []
            if self._dual_source_mode and source:
                self._prev_words_by_source[source] = trailing
            else:
                self._prev_words = trailing

    @staticmethod
    def _normalize_word(w: str) -> str:
        return w.strip().lower().rstrip(".,!?;:'\"").lstrip("'\"")

    def _dedup_words(self, new_words: list[dict], prev_words: list[dict] | None = None) -> list[dict]:
        """
        Remove words from the start of new_words that overlap with the
        previous chunk's trailing words.

        Uses two strategies:
        1. Timestamp-based: skip new words whose timestamps fall before the
           last emitted word's end time.
        2. Sequence alignment: find where a contiguous run of new words
           matches a contiguous run in prev_words, and skip past it.
        """
        if prev_words is None:
            prev_words = self._prev_words
        if not prev_words or not new_words:
            return new_words

        prev_texts = [self._normalize_word(w["word"]) for w in prev_words]
        new_texts = [self._normalize_word(w["word"]) for w in new_words]

        last_prev_end = prev_words[-1]["end"]

        # Strategy 1: timestamp — skip new words that fall before the last
        # emitted word's end time
        time_skip = 0
        for i, w in enumerate(new_words):
            if w["start"] < last_prev_end - 0.15:
                time_skip = i + 1
            else:
                break

        # Strategy 2: single-word match — if the first new word matches the
        # last prev word AND its timestamp is near the boundary, skip it
        single_skip = 0
        for i in range(min(len(new_words), 3)):
            nw = new_texts[i]
            # Check against last few prev words
            for pw in prev_texts[-5:]:
                if nw == pw and new_words[i]["start"] <= last_prev_end + 0.5:
                    single_skip = i + 1
                    break
            if single_skip <= i:
                break  # Stop if this word didn't match

        # Strategy 3: sequence alignment — find longest contiguous match
        # between a suffix of prev_texts and a prefix of new_texts
        seq_skip = 0
        for start_p in range(max(0, len(prev_texts) - 15), len(prev_texts)):
            match_len = 0
            for j in range(min(len(prev_texts) - start_p, len(new_texts))):
                if prev_texts[start_p + j] == new_texts[j]:
                    match_len += 1
                else:
                    break
            if match_len >= 2:
                seq_skip = max(seq_skip, match_len)

        # Also check offset alignments
        for start_n in range(1, min(len(new_texts), 8)):
            for start_p in range(max(0, len(prev_texts) - 10), len(prev_texts)):
                match_len = 0
                for j in range(min(len(prev_texts) - start_p, len(new_texts) - start_n)):
                    if prev_texts[start_p + j] == new_texts[start_n + j]:
                        match_len += 1
                    else:
                        break
                if match_len >= 2:
                    seq_skip = max(seq_skip, start_n + match_len)

        best_skip = max(time_skip, single_skip, seq_skip)

        if best_skip > 0:
            skipped = " ".join(w["word"].strip() for w in new_words[:best_skip])
            print(f"  Dedup: skipped {best_skip} words (time={time_skip}, single={single_skip}, seq={seq_skip}): '{skipped}'")

        return new_words[best_skip:]

    def _words_to_segments(self, words: list[dict], original_segments: list[dict]) -> list[dict]:
        """
        Rebuild segments from deduped words. Uses original segment boundaries
        where possible — a word belongs to a segment if its timestamp falls
        within that segment's time range.
        """
        if not words:
            return []

        # Build a map: for each original segment, collect deduped words that fall within it
        segments = []
        used = set()

        for orig_seg in original_segments:
            seg_start = orig_seg["start"]
            seg_end = orig_seg["end"]
            seg_words = []

            for i, w in enumerate(words):
                if i in used:
                    continue
                # Word falls within this segment's time range (with tolerance)
                if w["start"] >= seg_start - 0.2 and w["start"] <= seg_end + 0.2:
                    seg_words.append(w)
                    used.add(i)

            if seg_words:
                text = "".join(w["word"] for w in seg_words).strip()
                if text:
                    segments.append({
                        "start": seg_words[0]["start"],
                        "end": seg_words[-1]["end"],
                        "text": text,
                        "words": seg_words,
                    })

        # Any remaining unmatched words
        remaining = [w for i, w in enumerate(words) if i not in used]
        if remaining:
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

    def _next_untitled_name(self) -> str:
        """Generate 'Untitled 001', 'Untitled 002', etc."""
        max_num = 0
        if config.SESSIONS_DIR.exists():
            for session_dir in config.SESSIONS_DIR.iterdir():
                transcript = session_dir / "transcript.json"
                if transcript.exists():
                    data = json.loads(transcript.read_text())
                    title = data.get("title", "")
                    if title.startswith("Untitled"):
                        parts = title.split()
                        if len(parts) == 2 and parts[1].isdigit():
                            max_num = max(max_num, int(parts[1]))
        return f"Untitled {max_num + 1:03d}"

    def _save_transcript(self):
        if not self.session:
            return
        transcript_path = self.session["dir"] / "transcript.json"

        # Calculate duration from segments
        segments = self.session["segments"]
        duration = 0.0
        if segments:
            duration = segments[-1]["end"] - segments[0]["start"]

        title = self.session.get("title") or self._next_untitled_name()

        save_data = {
            "id": self.session["id"],
            "title": title,
            "started_at": self.session["started_at"],
            "ended_at": self.session["ended_at"],
            "duration": round(duration, 1),
            "dual_source": self.session.get("dual_source", False),
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
                    "title": data.get("title", data["id"]),
                    "started_at": data["started_at"],
                    "ended_at": data["ended_at"],
                    "duration": data.get("duration", 0),
                    "segment_count": len(data.get("segments", [])),
                })
        return sessions

    def rename_session(self, session_id: str, title: str) -> bool:
        transcript = config.SESSIONS_DIR / session_id / "transcript.json"
        if not transcript.exists():
            return False
        data = json.loads(transcript.read_text())
        data["title"] = title.strip()
        transcript.write_text(json.dumps(data, indent=2))
        return True

    def get_session_transcript(self, session_id: str) -> dict | None:
        transcript = config.SESSIONS_DIR / session_id / "transcript.json"
        if not transcript.exists():
            return None
        return json.loads(transcript.read_text())
