from __future__ import annotations

from pathlib import Path

import numpy as np

from voicetyper.audio import (
    StreamingDownsampler,
    is_effectively_silent,
    peak_dbfs,
    read_audio,
    save_wav,
    trim_silence,
)


def _tone(seconds: float = 1.0, rate: int = 16000, amplitude: float = 0.4) -> np.ndarray:
    time = np.arange(int(seconds * rate), dtype=np.float32) / rate
    return (amplitude * np.sin(2 * np.pi * 220 * time)).astype(np.float32)


def test_wav_roundtrip_and_peak(tmp_path: Path) -> None:
    path = tmp_path / "tone.wav"
    data = _tone()
    save_wav(path, data, 16000)
    loaded, rate = read_audio(path)
    assert rate == 16000
    assert loaded.shape == data.shape
    assert peak_dbfs(loaded) < 0


def test_silence_detection_and_trimming() -> None:
    rate = 16000
    silence = np.zeros(rate, dtype=np.float32)
    assert is_effectively_silent(silence, rate)
    trimmed = trim_silence(np.concatenate([silence[:4000], _tone(), silence[:4000]]), rate)
    assert 0 < trimmed.size < rate * 3


def test_resampling_changes_length() -> None:
    source = _tone(seconds=0.5, rate=48000)
    converted = ReadableResampler().run(source, 48000, 16000)
    assert abs(converted.size - 8000) <= 2


def test_streaming_downsampler_matches_chunking() -> None:
    downsampler = StreamingDownsampler(48000, 8000)
    source = (np.random.default_rng(1).integers(-2000, 2000, 48000)).astype(np.int16)
    output = np.concatenate([downsampler.feed(source[i : i + 4800]) for i in range(0, source.size, 4800)])
    assert output.dtype == np.int16
    assert abs(output.size - 8000) <= 2


class ReadableResampler:
    def run(self, data: np.ndarray, source: int, target: int) -> np.ndarray:
        from voicetyper.audio import _resample_float

        return _resample_float(data, source, target)
