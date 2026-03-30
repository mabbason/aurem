"""
Audio capture using PyAudioWPatch (WASAPI).
Captures system audio loopback and microphone, mixes into mono.
Uses VAD-based chunking: splits on natural speech pauses instead of fixed intervals.
"""

import threading
import numpy as np
import pyaudiowpatch as pyaudio

import config

NATIVE_RATE = 48000
FRAME_SIZE = 4800  # 100ms at 48kHz


class AudioCapture:
    def __init__(self, on_chunk_ready):
        self.on_chunk_ready = on_chunk_ready
        self.target_rate = config.AUDIO_SAMPLE_RATE

        # VAD chunking parameters
        self.min_samples = int(self.target_rate * config.AUDIO_MIN_CHUNK_SECONDS)
        self.max_samples = int(self.target_rate * config.AUDIO_MAX_CHUNK_SECONDS)
        self.silence_threshold = config.AUDIO_SILENCE_THRESHOLD
        self.silence_samples = int(self.target_rate * config.AUDIO_SILENCE_DURATION_MS / 1000)
        self.overlap_samples = int(self.target_rate * config.AUDIO_OVERLAP_SECONDS)

        # Window for RMS energy calculation (50ms)
        self.energy_window = int(self.target_rate * 0.05)

        self._pa = None
        self._lb_stream = None
        self._mic_stream = None
        self._stop_event = threading.Event()
        self._chunk_index = 0
        self._total_emitted_samples = 0

        self._lock = threading.Lock()
        self._lb_buf = np.array([], dtype=np.float32)
        self._mic_buf = np.array([], dtype=np.float32)
        self._has_mic = False

        self._mixer_thread = None

    def start(self):
        self._pa = pyaudio.PyAudio()
        self._stop_event.clear()
        self._chunk_index = 0
        self._total_emitted_samples = 0

        wasapi = self._pa.get_host_api_info_by_type(pyaudio.paWASAPI)

        # Find and open loopback stream
        for i in range(self._pa.get_device_count()):
            dev = self._pa.get_device_info_by_index(i)
            if dev["hostApi"] == wasapi["index"] and dev.get("isLoopbackDevice", False):
                lb_ch = dev["maxInputChannels"]
                print(f"Loopback: {dev['name']} ({lb_ch}ch)")
                self._lb_stream = self._pa.open(
                    format=pyaudio.paFloat32,
                    channels=lb_ch,
                    rate=NATIVE_RATE,
                    input=True,
                    input_device_index=i,
                    frames_per_buffer=FRAME_SIZE,
                    stream_callback=self._make_lb_callback(lb_ch),
                )
                break

        if not self._lb_stream:
            raise RuntimeError("No WASAPI loopback device found")

        # Find and open microphone stream
        for i in range(self._pa.get_device_count()):
            dev = self._pa.get_device_info_by_index(i)
            if (dev["hostApi"] == wasapi["index"]
                    and not dev.get("isLoopbackDevice", False)
                    and dev["maxInputChannels"] > 0):
                print(f"Microphone: {dev['name']}")
                try:
                    self._mic_stream = self._pa.open(
                        format=pyaudio.paFloat32,
                        channels=1,
                        rate=NATIVE_RATE,
                        input=True,
                        input_device_index=i,
                        frames_per_buffer=FRAME_SIZE,
                        stream_callback=self._mic_callback,
                    )
                    self._has_mic = True
                except Exception as e:
                    print(f"Could not open mic: {e}")
                break

        self._mixer_thread = threading.Thread(target=self._mix_loop, daemon=True)
        self._mixer_thread.start()
        print("Audio capture started (VAD chunking)")

    def stop(self):
        self._stop_event.set()

        if self._lb_stream:
            self._lb_stream.stop_stream()
            self._lb_stream.close()
            self._lb_stream = None

        if self._mic_stream:
            self._mic_stream.stop_stream()
            self._mic_stream.close()
            self._mic_stream = None

        if self._mixer_thread:
            self._mixer_thread.join(timeout=3)
            self._mixer_thread = None

        if self._pa:
            self._pa.terminate()
            self._pa = None

        print("Audio capture stopped")

    @property
    def is_running(self):
        return self._lb_stream is not None and self._lb_stream.is_active()

    def _make_lb_callback(self, channels):
        def callback(in_data, frame_count, time_info, status):
            raw = np.frombuffer(in_data, dtype=np.float32).reshape(-1, channels)
            mono = np.mean(raw[:, :2], axis=1)
            mono16k = mono[::3]  # 48kHz -> 16kHz
            with self._lock:
                self._lb_buf = np.concatenate([self._lb_buf, mono16k])
            return (None, pyaudio.paContinue)
        return callback

    def _mic_callback(self, in_data, frame_count, time_info, status):
        mono = np.frombuffer(in_data, dtype=np.float32)
        mono16k = mono[::3]
        with self._lock:
            self._mic_buf = np.concatenate([self._mic_buf, mono16k])
        return (None, pyaudio.paContinue)

    def _find_silence_boundary(self, audio):
        """
        Find the best split point in audio by looking for silence gaps.
        Scans from the end of the min_chunk region backward to find the latest
        silence gap. Returns the sample index to split at, or -1 if no silence found.
        """
        if len(audio) < self.min_samples:
            return -1

        # Compute RMS energy in windows across the audio
        n_windows = len(audio) // self.energy_window
        if n_windows == 0:
            return -1

        energies = np.array([
            np.sqrt(np.mean(audio[i * self.energy_window:(i + 1) * self.energy_window] ** 2))
            for i in range(n_windows)
        ])

        # Find runs of silence (energy below threshold)
        is_silent = energies < self.silence_threshold
        silence_windows_needed = max(1, self.silence_samples // self.energy_window)

        # Scan from the end backward to find the latest silence gap after min_chunk
        min_window = self.min_samples // self.energy_window
        best_split = -1

        for i in range(n_windows - 1, min_window - 1, -1):
            # Check if there's a silence run ending at or around window i
            start = max(0, i - silence_windows_needed + 1)
            if np.all(is_silent[start:i + 1]):
                # Split at the middle of the silence gap
                split_window = (start + i) // 2
                best_split = split_window * self.energy_window
                break

        return best_split

    def _emit_chunk(self, chunk):
        offset = self._total_emitted_samples / self.target_rate
        self.on_chunk_ready(chunk, self._chunk_index, offset)
        # Advance by (chunk_length - overlap) to track absolute position
        self._total_emitted_samples += len(chunk) - self.overlap_samples
        self._chunk_index += 1

    def _mix_loop(self):
        mixed_buffer = np.array([], dtype=np.float32)

        while not self._stop_event.is_set():
            self._stop_event.wait(timeout=0.1)

            with self._lock:
                n_lb = len(self._lb_buf)
                n_mic = len(self._mic_buf)

                if self._has_mic and n_lb > 0 and n_mic > 0:
                    n = min(n_lb, n_mic)
                    lb = self._lb_buf[:n].copy()
                    mc = self._mic_buf[:n].copy()
                    self._lb_buf = self._lb_buf[n:]
                    self._mic_buf = self._mic_buf[n:]
                    new_audio = (lb + mc) * 0.5
                elif n_lb > 0 and not self._has_mic:
                    new_audio = self._lb_buf.copy()
                    self._lb_buf = np.array([], dtype=np.float32)
                else:
                    continue

            mixed_buffer = np.concatenate([mixed_buffer, new_audio])

            # VAD-based chunking: look for silence boundaries
            if len(mixed_buffer) >= self.min_samples:
                split = self._find_silence_boundary(mixed_buffer)

                if split > 0:
                    # Found a silence gap — split there
                    chunk = mixed_buffer[:split].copy()
                    # Keep overlap for context in next chunk
                    keep_from = max(0, split - self.overlap_samples)
                    mixed_buffer = mixed_buffer[keep_from:]
                    self._emit_chunk(chunk)

                elif len(mixed_buffer) >= self.max_samples:
                    # Hit max without finding silence — force split at best point
                    # Try to find any brief dip in energy near the end
                    split = self._find_best_split_near_end(mixed_buffer)
                    chunk = mixed_buffer[:split].copy()
                    keep_from = max(0, split - self.overlap_samples)
                    mixed_buffer = mixed_buffer[keep_from:]
                    self._emit_chunk(chunk)

        # Flush remaining buffer on stop
        if len(mixed_buffer) > self.target_rate:  # At least 1 second
            self._emit_chunk(mixed_buffer)

        print("Audio capture stopped")

    def _find_best_split_near_end(self, audio):
        """When forced to split at max_samples, find the lowest energy point
        in the last 3 seconds to minimize mid-word cuts."""
        search_start = max(self.min_samples, len(audio) - self.target_rate * 3)
        n_windows = (len(audio) - search_start) // self.energy_window

        if n_windows <= 0:
            return self.max_samples

        energies = np.array([
            np.sqrt(np.mean(
                audio[search_start + i * self.energy_window:
                      search_start + (i + 1) * self.energy_window] ** 2
            ))
            for i in range(n_windows)
        ])

        # Find the lowest energy window
        best_window = np.argmin(energies)
        return search_start + best_window * self.energy_window
