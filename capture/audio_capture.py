"""
Audio capture using PyAudioWPatch (WASAPI).
Supports N audio sources (any mix of loopback + microphones).
Uses VAD-based chunking: splits on natural speech pauses.

Device discovery happens once at server startup.
Session start/stop opens/closes streams instantly.
"""

import threading
import numpy as np
import pyaudiowpatch as pyaudio

import config

NATIVE_RATE = 48000
FRAME_SIZE = 4800  # 100ms at 48kHz


def discover_all_devices():
    """
    Discover all WASAPI audio devices at startup. No probing — just enumerates.
    Returns list of device dicts with keys: index, name, type, channels.
    """
    p = pyaudio.PyAudio()
    wasapi = p.get_host_api_info_by_type(pyaudio.paWASAPI)
    devices = []

    for i in range(p.get_device_count()):
        dev = p.get_device_info_by_index(i)
        if dev["hostApi"] != wasapi["index"]:
            continue

        is_loopback = dev.get("isLoopbackDevice", False)
        has_input = dev["maxInputChannels"] > 0

        if is_loopback:
            devices.append({
                "index": i,
                "name": dev["name"],
                "type": "loopback",
                "channels": dev["maxInputChannels"],
            })
        elif has_input:
            devices.append({
                "index": i,
                "name": dev["name"],
                "type": "microphone",
                "channels": 1,
            })

    p.terminate()

    for d in devices:
        print(f"  [{d['index']}] {d['type']:10s} {d['name']}")

    return devices


class AudioCapture:
    def __init__(self, on_chunk_ready, devices, all_device_info):
        """
        devices: list of device index ints to capture from
        all_device_info: full device list from discover_all_devices()
        """
        self.on_chunk_ready = on_chunk_ready
        self.target_rate = config.AUDIO_SAMPLE_RATE
        self._selected_devices = devices
        self._device_info = {d["index"]: d for d in all_device_info}

        # VAD chunking parameters
        self.min_samples = int(self.target_rate * config.AUDIO_MIN_CHUNK_SECONDS)
        self.max_samples = int(self.target_rate * config.AUDIO_MAX_CHUNK_SECONDS)
        self.silence_threshold = config.AUDIO_SILENCE_THRESHOLD
        self.silence_samples = int(self.target_rate * config.AUDIO_SILENCE_DURATION_MS / 1000)
        self.overlap_samples = int(self.target_rate * config.AUDIO_OVERLAP_SECONDS)
        self.energy_window = int(self.target_rate * 0.05)

        self._pa = None
        self._streams = []
        self._stop_event = threading.Event()
        self._chunk_index = 0
        self._total_emitted_samples = 0

        self._lock = threading.Lock()
        # One buffer per source device
        self._buffers = {}

        self._mixer_thread = None

    def start(self):
        self._pa = pyaudio.PyAudio()
        self._stop_event.clear()
        self._chunk_index = 0
        self._total_emitted_samples = 0
        self._streams = []
        self._buffers = {idx: np.array([], dtype=np.float32) for idx in self._selected_devices}

        for dev_idx in self._selected_devices:
            info = self._device_info.get(dev_idx)
            if not info:
                print(f"WARNING: Device {dev_idx} not found, skipping")
                continue

            channels = info["channels"]
            is_loopback = info["type"] == "loopback"

            try:
                stream = self._pa.open(
                    format=pyaudio.paFloat32,
                    channels=channels,
                    rate=NATIVE_RATE,
                    input=True,
                    input_device_index=dev_idx,
                    frames_per_buffer=FRAME_SIZE,
                    stream_callback=self._make_callback(dev_idx, channels, is_loopback),
                )
                self._streams.append(stream)
                label = "loopback" if is_loopback else "mic"
                print(f"  Opened [{dev_idx}] {info['name']} ({label})")
            except Exception as e:
                print(f"  Could not open [{dev_idx}] {info['name']}: {e}")

        if not self._streams:
            raise RuntimeError("No audio streams could be opened")

        self._mixer_thread = threading.Thread(target=self._mix_loop, daemon=True)
        self._mixer_thread.start()
        print(f"Audio capture started ({len(self._streams)} source(s))")

    def stop(self):
        self._stop_event.set()

        for stream in self._streams:
            try:
                stream.stop_stream()
                stream.close()
            except Exception:
                pass
        self._streams = []

        if self._mixer_thread:
            self._mixer_thread.join(timeout=3)
            self._mixer_thread = None

        if self._pa:
            self._pa.terminate()
            self._pa = None

        print("Audio capture stopped")

    @property
    def is_running(self):
        return any(s.is_active() for s in self._streams)

    def _make_callback(self, dev_idx, channels, is_loopback):
        def callback(in_data, frame_count, time_info, status):
            raw = np.frombuffer(in_data, dtype=np.float32)
            if channels > 1:
                raw = raw.reshape(-1, channels)
                mono = np.mean(raw[:, :2], axis=1)
            else:
                mono = raw
            mono16k = mono[::3]  # 48kHz -> 16kHz
            with self._lock:
                self._buffers[dev_idx] = np.concatenate([self._buffers[dev_idx], mono16k])
            return (None, pyaudio.paContinue)
        return callback

    def _mix_loop(self):
        mixed_buffer = np.array([], dtype=np.float32)
        n_sources = len(self._selected_devices)

        while not self._stop_event.is_set():
            self._stop_event.wait(timeout=0.1)

            with self._lock:
                # Collect all sources that have data
                active_chunks = []
                for idx in self._selected_devices:
                    if idx in self._buffers and len(self._buffers[idx]) > 0:
                        active_chunks.append((idx, len(self._buffers[idx])))

                if not active_chunks:
                    continue

                if len(active_chunks) == 1:
                    # Only one source has data — use it directly
                    idx = active_chunks[0][0]
                    new_audio = self._buffers[idx].copy()
                    self._buffers[idx] = np.array([], dtype=np.float32)
                else:
                    # Multiple sources have data — take min available and mix
                    n = min(length for _, length in active_chunks)
                    chunks = []
                    for idx, _ in active_chunks:
                        chunks.append(self._buffers[idx][:n].copy())
                        self._buffers[idx] = self._buffers[idx][n:]
                    new_audio = np.sum(chunks, axis=0)
                    peak = np.max(np.abs(new_audio))
                    if peak > 1.0:
                        new_audio /= peak

            mixed_buffer = np.concatenate([mixed_buffer, new_audio])

            # VAD-based chunking
            if len(mixed_buffer) >= self.min_samples:
                split = self._find_silence_boundary(mixed_buffer)

                if split > 0:
                    chunk = mixed_buffer[:split].copy()
                    keep_from = max(0, split - self.overlap_samples)
                    mixed_buffer = mixed_buffer[keep_from:]
                    self._emit_chunk(chunk)

                elif len(mixed_buffer) >= self.max_samples:
                    split = self._find_best_split_near_end(mixed_buffer)
                    chunk = mixed_buffer[:split].copy()
                    keep_from = max(0, split - self.overlap_samples)
                    mixed_buffer = mixed_buffer[keep_from:]
                    self._emit_chunk(chunk)

        if len(mixed_buffer) > self.target_rate:
            self._emit_chunk(mixed_buffer)

        print("Audio capture stopped")

    def _emit_chunk(self, chunk):
        offset = self._total_emitted_samples / self.target_rate
        self.on_chunk_ready(chunk, self._chunk_index, offset)
        self._total_emitted_samples += len(chunk) - self.overlap_samples
        self._chunk_index += 1

    def _find_silence_boundary(self, audio):
        if len(audio) < self.min_samples:
            return -1

        n_windows = len(audio) // self.energy_window
        if n_windows == 0:
            return -1

        energies = np.array([
            np.sqrt(np.mean(audio[i * self.energy_window:(i + 1) * self.energy_window] ** 2))
            for i in range(n_windows)
        ])

        is_silent = energies < self.silence_threshold
        silence_windows_needed = max(1, self.silence_samples // self.energy_window)

        min_window = self.min_samples // self.energy_window

        for i in range(n_windows - 1, min_window - 1, -1):
            start = max(0, i - silence_windows_needed + 1)
            if np.all(is_silent[start:i + 1]):
                split_window = (start + i) // 2
                return split_window * self.energy_window

        return -1

    def _find_best_split_near_end(self, audio):
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

        best_window = np.argmin(energies)
        return search_start + best_window * self.energy_window
