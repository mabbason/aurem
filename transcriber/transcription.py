"""
Speech-to-text using faster-whisper with CUDA acceleration.
Produces verbatim transcription with segment-level timestamps.
"""

import numpy as np
from faster_whisper import WhisperModel

import config


class Transcriber:
    def __init__(self):
        self.model = None

    def load_model(self):
        print(f"Loading Whisper model '{config.WHISPER_MODEL}' on {config.WHISPER_DEVICE}...")
        self.model = WhisperModel(
            config.WHISPER_MODEL,
            device=config.WHISPER_DEVICE,
            compute_type=config.WHISPER_COMPUTE_TYPE,
        )
        print("Whisper model loaded")

    def transcribe(self, audio: np.ndarray, offset_seconds: float = 0.0, language: str = "en") -> list[dict]:
        """
        Transcribe audio array and return segments with absolute timestamps.

        Returns list of:
            {
                "start": float,  # absolute start time in seconds
                "end": float,    # absolute end time in seconds
                "text": str,     # verbatim text
            }
        """
        if self.model is None:
            self.load_model()

        segments_iter, info = self.model.transcribe(
            audio,
            language=language,
            beam_size=5,
            word_timestamps=True,
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=300,
                speech_pad_ms=100,
            ),
        )

        results = []
        for seg in segments_iter:
            results.append({
                "start": round(offset_seconds + seg.start, 1),
                "end": round(offset_seconds + seg.end, 1),
                "text": seg.text.strip(),
                "words": [
                    {
                        "start": round(offset_seconds + w.start, 1),
                        "end": round(offset_seconds + w.end, 1),
                        "word": w.word,
                        "probability": round(w.probability, 3),
                    }
                    for w in (seg.words or [])
                ],
            })

        return results
