"""
Speaker diarization using pyannote.audio.
Assigns speaker labels to transcription segments.
Uses speaker embeddings for cross-chunk speaker consistency.
"""

import numpy as np
import torch

import config

DIARIZATION_AVAILABLE = False


class Diarizer:
    def __init__(self):
        self.pipeline = None
        self.embedding_model = None
        self.speaker_map = {}  # Map pyannote labels to consistent Speaker N labels
        self.speaker_embeddings = {}  # Speaker N -> average embedding vector
        self.next_speaker_id = 1
        self._similarity_threshold = 0.75  # Cosine similarity threshold for same speaker

    def load_model(self):
        global DIARIZATION_AVAILABLE

        if not config.HF_TOKEN:
            print("WARNING: HF_TOKEN not set — speaker diarization disabled")
            print("  Set HF_TOKEN in .env and accept terms at:")
            print("  https://huggingface.co/pyannote/speaker-diarization-3.1")
            return

        try:
            from pyannote.audio import Pipeline, Inference

            print("Loading pyannote diarization pipeline...")
            self.pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                token=config.HF_TOKEN,
            )

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.pipeline.to(device)

            try:
                print("Loading pyannote speaker embedding model...")
                self.embedding_model = Inference(
                    "pyannote/embedding",
                    window="whole",
                )
                try:
                    self.embedding_model.to(device)
                except (AttributeError, TypeError):
                    pass
                print(f"Diarization pipeline loaded with embeddings ({device})")
            except Exception as e:
                print(f"  Embedding model failed ({e}), cross-chunk speaker matching disabled")
                self.embedding_model = None
                print(f"Diarization pipeline loaded without embeddings ({device})")

            DIARIZATION_AVAILABLE = True
        except Exception as e:
            print(f"WARNING: Could not load diarization pipeline: {e}")
            print("  Speaker diarization will be disabled")

    def diarize(self, audio: np.ndarray, segments: list[dict]) -> list[dict]:
        """
        Assign speaker labels to transcription segments.
        Modifies segments in-place, adding 'speaker' field.
        Uses embeddings to maintain consistent speaker labels across chunks.
        """
        if self.pipeline is None:
            for seg in segments:
                seg["speaker"] = "Speaker"
            return segments

        try:
            waveform = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
            input_data = {"waveform": waveform, "sample_rate": config.AUDIO_SAMPLE_RATE}
            diarization = self.pipeline(input_data)

            # Build timeline of speaker segments with embeddings
            speaker_timeline = []
            chunk_speaker_embeddings = {}  # pyannote label -> embedding

            for turn, _, speaker_label in diarization.itertracks(yield_label=True):
                speaker_timeline.append({
                    "start": turn.start,
                    "end": turn.end,
                    "label": speaker_label,
                })

                # Extract embedding for this speaker's audio segment
                if speaker_label not in chunk_speaker_embeddings and self.embedding_model:
                    try:
                        start_sample = int(turn.start * config.AUDIO_SAMPLE_RATE)
                        end_sample = int(turn.end * config.AUDIO_SAMPLE_RATE)
                        segment_audio = audio[start_sample:end_sample]

                        # Need at least 0.5s of audio for a useful embedding
                        if len(segment_audio) >= config.AUDIO_SAMPLE_RATE * 0.5:
                            seg_waveform = torch.tensor(segment_audio, dtype=torch.float32).unsqueeze(0)
                            embedding = self.embedding_model(
                                {"waveform": seg_waveform, "sample_rate": config.AUDIO_SAMPLE_RATE}
                            )
                            chunk_speaker_embeddings[speaker_label] = embedding
                    except Exception:
                        pass

            # Map chunk-local pyannote labels to session-consistent Speaker N labels
            chunk_label_map = {}
            for pyannote_label, embedding in chunk_speaker_embeddings.items():
                matched = self._match_speaker(embedding)
                if matched:
                    chunk_label_map[pyannote_label] = matched
                else:
                    new_label = f"Speaker {self.next_speaker_id}"
                    self.next_speaker_id += 1
                    chunk_label_map[pyannote_label] = new_label
                    self.speaker_embeddings[new_label] = embedding

            # For labels without embeddings, fall back to the old sequential mapping
            for turn_info in speaker_timeline:
                label = turn_info["label"]
                if label not in chunk_label_map:
                    if label not in self.speaker_map:
                        self.speaker_map[label] = f"Speaker {self.next_speaker_id}"
                        self.next_speaker_id += 1
                    chunk_label_map[label] = self.speaker_map[label]

            # Assign speakers to transcription segments by overlap
            for seg in segments:
                seg["speaker"] = self._find_speaker(
                    seg["start"], seg["end"], speaker_timeline, chunk_label_map
                )

        except Exception as e:
            print(f"Diarization error: {e}")
            for seg in segments:
                seg["speaker"] = "Speaker"

        return segments

    def _match_speaker(self, embedding) -> str | None:
        """Match embedding against known speakers using cosine similarity."""
        if not self.speaker_embeddings:
            return None

        best_score = -1
        best_label = None

        for label, known_embedding in self.speaker_embeddings.items():
            score = self._cosine_similarity(embedding, known_embedding)
            if score > best_score:
                best_score = score
                best_label = label

        if best_score >= self._similarity_threshold:
            # Update the stored embedding with running average
            old = self.speaker_embeddings[best_label]
            self.speaker_embeddings[best_label] = (old + embedding) / 2
            # Re-normalize
            norm = np.linalg.norm(self.speaker_embeddings[best_label])
            if norm > 0:
                self.speaker_embeddings[best_label] = self.speaker_embeddings[best_label] / norm
            return best_label

        return None

    @staticmethod
    def _cosine_similarity(a, b) -> float:
        """Compute cosine similarity between two vectors."""
        if isinstance(a, torch.Tensor):
            a = a.cpu().numpy()
        if isinstance(b, torch.Tensor):
            b = b.cpu().numpy()
        a = a.flatten()
        b = b.flatten()
        dot = np.dot(a, b)
        norm = np.linalg.norm(a) * np.linalg.norm(b)
        if norm == 0:
            return 0.0
        return float(dot / norm)

    def _find_speaker(self, seg_start, seg_end, timeline, label_map):
        """Find the speaker with most overlap for a given time range."""
        best_overlap = 0
        best_label = None

        for turn in timeline:
            overlap_start = max(seg_start, turn["start"])
            overlap_end = min(seg_end, turn["end"])
            overlap = max(0, overlap_end - overlap_start)

            if overlap > best_overlap:
                best_overlap = overlap
                best_label = turn["label"]

        if best_label is None:
            return "Speaker"

        return label_map.get(best_label, "Speaker")

    def reset(self):
        self.speaker_map = {}
        self.speaker_embeddings = {}
        self.next_speaker_id = 1
