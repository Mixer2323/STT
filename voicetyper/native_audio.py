"""Minimal Media Foundation/WASAPI recorder built on soundcard's cffi bindings.

``soundcard``'s own recorder asserts that the device mix format is
``WAVE_FORMAT_EXTENSIBLE`` with an IEEE-float subformat. Many endpoints —
including USB headsets and the RDP "Remote Audio" device — report plain
``WAVE_FORMAT_IEEE_FLOAT`` or PCM instead, so that recorder refuses to open
them. This recorder keeps the device mix format and converts PCM/float samples
to float32; the caller resamples to the speech rate afterwards.
"""

from __future__ import annotations

import logging
import time

import numpy as np

logger = logging.getLogger(__name__)

try:
    from soundcard import mediafoundation as _mf

    HAS_MEDIA_FOUNDATION = True
except Exception:
    _mf = None
    HAS_MEDIA_FOUNDATION = False

_FLOAT_GUID = bytes.fromhex("0300000000001000800000aa00389b71")
_PCM_GUID = bytes.fromhex("0100000000001000800000aa00389b71")


def _detect_sample_format(raw: bytes) -> str:
    tag = int.from_bytes(raw[0:2], "little")
    bits = int.from_bytes(raw[14:16], "little")
    if tag == 0x0003:
        return "float32"
    if tag == 0x0001:
        return {16: "pcm16", 24: "pcm24", 32: "pcm32"}.get(bits, "pcm16")
    if tag == 0xFFFE and len(raw) >= 40:
        subformat = raw[24:40]
        if subformat == _FLOAT_GUID:
            return "float32"
        if subformat == _PCM_GUID:
            return {16: "pcm16", 24: "pcm24", 32: "pcm32"}.get(bits, "pcm16")
    if bits == 32:
        return "float32"
    if bits == 24:
        return "pcm24"
    return "pcm16"


if HAS_MEDIA_FOUNDATION:

    class _Recorder(_mf._Recorder):
        """Recorder that accepts the device mix format instead of forcing float."""

        def __init__(self, ptr, seconds: float = 0.1):
            self._ptr = ptr
            format_pointer = _mf._ffi.new("WAVEFORMATEXTENSIBLE**")
            hr = self._ptr[0][0].lpVtbl.GetMixFormat(self._ptr[0], format_pointer)
            _mf._com.check_error(hr)
            fmt = format_pointer[0][0]
            tag = int(fmt.Format.wFormatTag)
            raw_size = 40 if tag == 0xFFFE else 16
            raw = bytes(_mf._ffi.buffer(_mf._ffi.addressof(fmt), raw_size))
            self._sample_format = _detect_sample_format(raw)
            self._bytes_per_frame = int(fmt.Format.nBlockAlign)
            self._mix_channels = max(1, int(fmt.Format.nChannels))
            self.samplerate = int(fmt.Format.nSamplesPerSec)
            self.channelmap = list(range(self._mix_channels))

            blocksize = max(1, int(self.samplerate * seconds))
            buffer_duration = int(blocksize / self.samplerate * 10_000_000)
            hr = self._ptr[0][0].lpVtbl.Initialize(
                self._ptr[0],
                0,  # AUDCLNT_SHAREMODE_SHARED
                0x00080000,  # AUDCLNT_STREAMFLAGS_NOPERSIST
                buffer_duration,
                0,
                format_pointer[0],
                _mf._ffi.NULL,
            )
            _mf._com.check_error(hr)
            _mf._ole32.CoTaskMemFree(format_pointer[0])
            self._idle_start_time = None

        @property
        def sample_format(self) -> str:
            return self._sample_format

        def _convert(self, raw: bytes) -> np.ndarray:
            if self._sample_format == "float32":
                return np.frombuffer(raw, dtype="<f4").astype(np.float32, copy=True)
            if self._sample_format == "pcm16":
                return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
            if self._sample_format == "pcm24":
                data = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
                values = (
                    data[:, 0].astype(np.int32)
                    | (data[:, 1].astype(np.int32) << 8)
                    | (data[:, 2].astype(np.int32) << 16)
                )
                values = np.where(values & 0x800000, values - 0x1000000, values)
                return values.astype(np.float32) / 8388608.0
            if self._sample_format == "pcm32":
                return (
                    np.frombuffer(raw, dtype="<i4").astype(np.float32)
                    / 2147483648.0
                )
            raise RuntimeError(f"Неизвестный формат аудио: {self._sample_format}")

        def _record_chunk(self) -> np.ndarray:
            while self._capture_available_frames() == 0:
                if self._idle_start_time is None:
                    self._idle_start_time = time.perf_counter_ns()
                default_block_length, minimum_block_length = self.deviceperiod
                time.sleep(minimum_block_length / 4)
                elapsed = time.perf_counter_ns() - self._idle_start_time
                if elapsed / 1_000_000_000 > default_block_length * 4:
                    frames = int(self.samplerate * elapsed / 1_000_000_000)
                    self._idle_start_time += elapsed
                    return np.zeros(
                        frames * self._mix_channels, dtype=np.float32
                    )

            self._idle_start_time = None
            data_pointer, frames, flags = self._capture_buffer()
            if data_pointer == _mf._ffi.NULL:
                raise RuntimeError("Could not create capture buffer")
            raw = _mf._ffi.buffer(
                data_pointer, frames * self._bytes_per_frame
            )[:]
            chunk = self._convert(raw)
            if flags & _mf._ole32.AUDCLNT_BUFFERFLAGS_SILENT:
                chunk[:] = 0
            if self._is_first_frame:
                flags &= ~_mf._ole32.AUDCLNT_BUFFERFLAGS_DATA_DISCONTINUITY
                self._is_first_frame = False
            if flags & _mf._ole32.AUDCLNT_BUFFERFLAGS_DATA_DISCONTINUITY:
                logger.debug("Разрыв в потоке записи Windows Media Foundation")
            if frames > 0:
                self._capture_release(frames)
                return chunk
            return np.zeros([0], dtype=np.float32)

    def open_recorder(microphone, *, seconds: float = 0.1) -> "_Recorder":
        return _Recorder(microphone._audio_client(), seconds)

else:

    def open_recorder(microphone, *, seconds: float = 0.1):
        raise RuntimeError("soundcard не установлен — запись через Media Foundation недоступна")
