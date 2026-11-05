"""Windows WASAPI capture built on soundcard's cffi bindings.

``soundcard``'s own recorder asserts that the device mix format is
``WAVE_FORMAT_EXTENSIBLE`` with an IEEE-float subformat. Many endpoints —
including USB headsets and the RDP "Remote Audio" device — report plain
``WAVE_FORMAT_IEEE_FLOAT`` or PCM instead, so that recorder refuses to open
them.

This module keeps the device mix format, and when a device refuses it, retries
with a handful of standard formats plus ``AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM``
so the Windows audio engine does the conversion. Every failure carries the
failing step and HRESULT, which makes ``--audio-test`` reports actionable.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

try:
    from soundcard import mediafoundation as _mf

    HAS_MEDIA_FOUNDATION = True
except Exception:  # pragma: no cover - only on non-Windows
    _mf = None
    HAS_MEDIA_FOUNDATION = False

_FLOAT_GUID = bytes.fromhex("0300000000001000800000aa00389b71")
_PCM_GUID = bytes.fromhex("0100000000001000800000aa00389b71")

# AUDCLNT_STREAMFLAGS_*
_NOPERSIST = 0x00080000
_RATEADJUST = 0x00100000
_SRC_DEFAULT_QUALITY = 0x08000000
_AUTOCONVERTPCM = 0x80000000
_AUTOCONVERT = _AUTOCONVERTPCM | _SRC_DEFAULT_QUALITY | _RATEADJUST | _NOPERSIST

_SHARED_MODE = 0
_EXCLUSIVE_MODE = 1

_HRESULT_TEXT = {
    0x80004002: "E_NOINTERFACE",
    0x80004003: "E_POINTER",
    0x8007000E: "E_OUTOFMEMORY",
    0x80070057: "E_INVALIDARG",
    0x88890001: "AUDCLNT_E_NOT_INITIALIZED",
    0x88890002: "AUDCLNT_E_ALREADY_INITIALIZED",
    0x88890003: "AUDCLNT_E_WRONG_ENDPOINT_TYPE",
    0x88890004: "AUDCLNT_E_DEVICE_INVALIDATED",
    0x88890005: "AUDCLNT_E_NOT_STOPPED",
    0x88890006: "AUDCLNT_E_BUFFER_TOO_LARGE",
    0x88890008: "AUDCLNT_E_UNSUPPORTED_FORMAT",
    0x8889000A: "AUDCLNT_E_DEVICE_IN_USE",
    0x8889000B: "AUDCLNT_E_BUFFER_OPERATION_PENDING",
    0x8889000E: "AUDCLNT_E_EXCLUSIVE_MODE_NOT_ALLOWED",
    0x8889000F: "AUDCLNT_E_ENDPOINT_CREATE_FAILED",
    0x88890010: "AUDCLNT_E_SERVICE_NOT_RUNNING",
    0x88890012: "AUDCLNT_E_EXCLUSIVE_MODE_ONLY",
    0x88890013: "AUDCLNT_E_BUFDURATION_PERIOD_NOT_EQUAL",
    0x8889001E: "AUDCLNT_E_INVALID_STREAM_FLAG",
    0x88890020: "AUDCLNT_E_NONOFFLOAD_MODE_ONLY",
}


def _hresult_code(value: int) -> int:
    return int(value) & 0xFFFFFFFF


def _hresult_text(value: int) -> str:
    code = _hresult_code(value)
    name = _HRESULT_TEXT.get(code)
    if name:
        return f"HRESULT 0x{code:08X} ({name})"
    return f"HRESULT 0x{code:08X}"


class CaptureError(RuntimeError):
    """Capture failed with an optional step name, HRESULT and format variant."""

    def __init__(
        self,
        message: str,
        *,
        step: str | None = None,
        hresult: int | None = None,
        variant: str | None = None,
    ) -> None:
        self.step = step
        self.hresult = hresult
        self.variant = variant
        super().__init__(message)


def _attempt_details(errors: list[BaseException]) -> str:
    """Group every failed variant by HRESULT — none of them may be hidden.

    The RDP "Remote Audio" endpoint answers GetMixFormat but refuses every
    Initialize call; showing only the last error made a blocked endpoint look
    like a format problem.
    """
    groups: dict[str, list[str]] = {}
    loose: list[str] = []
    for error in errors:
        code = getattr(error, "hresult", None)
        if code is None:
            loose.append(str(error))
            continue
        variant = getattr(error, "variant", None) or str(error)
        groups.setdefault(_hresult_text(code), []).append(variant)
    parts = [
        f"{hresult} → {', '.join(variants)}" for hresult, variants in groups.items()
    ]
    parts.extend(loose)
    return "; ".join(parts)


def _check(hr: int, step: str) -> None:
    code = _hresult_code(hr)
    if code == 0:
        return
    text = f"{step}: {_hresult_text(code)}"
    raise CaptureError(text, step=step, hresult=code)


def _u16(raw: bytes, offset: int) -> int:
    return int.from_bytes(raw[offset : offset + 2], "little")


def _u32(raw: bytes, offset: int) -> int:
    return int.from_bytes(raw[offset : offset + 4], "little")


def _detect_sample_format(raw: bytes) -> str:
    tag = _u16(raw, 0)
    bits = _u16(raw, 14)
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


@dataclass(frozen=True)
class MixFormat:
    """Parsed ``GetMixFormat`` result plus its raw bytes."""

    raw: bytes
    tag: int
    channels: int
    rate: int
    avg_bytes_per_sec: int
    block_align: int
    bits: int
    cb_size: int
    valid_bits: int
    channel_mask: int
    subformat: bytes
    sample_format: str

    def describe(self) -> str:
        kind = {
            0x0001: "PCM",
            0x0003: "IEEE float",
            0xFFFE: "WAVE_FORMAT_EXTENSIBLE",
        }.get(self.tag, f"tag 0x{self.tag:04X}")
        parts = [
            kind,
            f"{self.rate} Гц",
            f"{self.channels} кан.",
            f"{self.bits} бит",
            f"block={self.block_align}",
            f"avg={self.avg_bytes_per_sec}",
            f"cbSize={self.cb_size}",
            f"данные={self.sample_format}",
        ]
        if self.valid_bits and self.valid_bits != self.bits:
            parts.append(f"valid={self.valid_bits}")
        if self.channel_mask:
            parts.append(f"mask=0x{self.channel_mask:X}")
        if self.subformat:
            parts.append(f"sub={self._guid_text(self.subformat)}")
        return ", ".join(parts)

    @staticmethod
    def _guid_text(raw: bytes) -> str:
        data1 = int.from_bytes(raw[0:4], "little")
        data2 = int.from_bytes(raw[4:6], "little")
        data3 = int.from_bytes(raw[6:8], "little")
        data4 = raw[8:16]
        if data4:
            tail = (
                f"{data4[0]:02X}{data4[1]:02X}-"
                f"{data4[2]:02X}{data4[3]:02X}{data4[4]:02X}"
                f"{data4[5]:02X}{data4[6]:02X}{data4[7]:02X}"
            )
        else:
            tail = ""
        return f"{{{data1:08X}-{data2:04X}-{data3:04X}-{tail}}}"


def parse_mix_format(raw: bytes) -> MixFormat:
    tag = _u16(raw, 0)
    channels = _u16(raw, 2)
    rate = _u32(raw, 4)
    avg_bytes = _u32(raw, 8)
    block_align = _u16(raw, 12)
    bits = _u16(raw, 14)
    cb_size = _u16(raw, 16) if len(raw) >= 18 else 0
    valid_bits = bits
    channel_mask = 0
    subformat = b""
    if tag == 0xFFFE and cb_size >= 22 and len(raw) >= 40:
        valid_bits = _u16(raw, 18)
        channel_mask = _u32(raw, 20)
        subformat = raw[24:40]
    return MixFormat(
        raw=raw,
        tag=tag,
        channels=max(1, channels),
        rate=rate,
        avg_bytes_per_sec=avg_bytes,
        block_align=max(1, block_align),
        bits=bits,
        cb_size=cb_size,
        valid_bits=valid_bits,
        channel_mask=channel_mask,
        subformat=subformat,
        sample_format=_detect_sample_format(raw),
    )


@dataclass(frozen=True)
class _CaptureFormat:
    label: str
    rate: int
    channels: int
    bits: int
    block_align: int
    tag: int
    extensible: bool
    valid_bits: int
    channel_mask: int
    subformat: bytes
    flags: int
    buffer_hns: int
    sample_format: str
    share_mode: int = _SHARED_MODE
    raw: bytes | None = None

    def summary(self) -> str:
        flags = []
        if self.flags & _AUTOCONVERTPCM:
            flags.append("autoconvert")
        if self.flags & _NOPERSIST:
            flags.append("nopersist")
        flag_text = "+".join(flags) if flags else "без флагов"
        return (
            f"{self.label}: {self.rate} Гц, {self.channels} кан., "
            f"{self.bits} бит, {self.sample_format}, {flag_text}"
        )


def _from_mix(
    mix: MixFormat,
    *,
    label: str,
    flags: int,
    buffer_hns: int,
    share_mode: int = _SHARED_MODE,
) -> _CaptureFormat:
    return _CaptureFormat(
        label=label,
        rate=mix.rate,
        channels=mix.channels,
        bits=mix.bits,
        block_align=mix.block_align,
        tag=mix.tag,
        extensible=bool(mix.subformat),
        valid_bits=mix.valid_bits,
        channel_mask=mix.channel_mask,
        subformat=mix.subformat,
        flags=flags,
        buffer_hns=buffer_hns,
        sample_format=mix.sample_format,
        share_mode=share_mode,
        raw=mix.raw,
    )


def _pcm16(
    label: str,
    rate: int,
    channels: int,
    flags: int,
    buffer_hns: int,
    share_mode: int = _SHARED_MODE,
) -> _CaptureFormat:
    return _CaptureFormat(
        label=label,
        rate=rate,
        channels=channels,
        bits=16,
        block_align=channels * 2,
        tag=0x0001,
        extensible=False,
        valid_bits=16,
        channel_mask=0,
        subformat=_PCM_GUID,
        flags=flags,
        buffer_hns=buffer_hns,
        sample_format="pcm16",
        share_mode=share_mode,
    )


def _float32(
    label: str,
    rate: int,
    channels: int,
    flags: int,
    buffer_hns: int,
    share_mode: int = _SHARED_MODE,
) -> _CaptureFormat:
    return _CaptureFormat(
        label=label,
        rate=rate,
        channels=channels,
        bits=32,
        block_align=channels * 4,
        tag=0xFFFE,
        extensible=True,
        valid_bits=32,
        channel_mask=(1 << channels) - 1,
        subformat=_FLOAT_GUID,
        flags=flags,
        buffer_hns=buffer_hns,
        sample_format="float32",
        share_mode=share_mode,
    )


def format_candidates(
    mix: MixFormat | None,
    *,
    preferred_rate: int = 16000,
) -> list[_CaptureFormat]:
    """Return the initialization variants to try, most likely first."""
    candidates: list[_CaptureFormat] = []
    seen: set[tuple[object, ...]] = set()

    def add(spec: _CaptureFormat) -> None:
        if spec.raw is not None:
            key: tuple[object, ...] = ("mix", spec.raw, spec.flags, spec.buffer_hns)
        else:
            key = (
                spec.tag,
                spec.rate,
                spec.channels,
                spec.bits,
                spec.flags,
                spec.buffer_hns,
            )
        if key in seen:
            return
        seen.add(key)
        candidates.append(spec)

    if mix is not None:
        add(_from_mix(mix, label="формат устройства", flags=_NOPERSIST, buffer_hns=1_000_000))
        add(_from_mix(mix, label="формат устройства, буфер 0", flags=0, buffer_hns=0))
        add(
            _from_mix(
                mix,
                label="формат устройства + автоконвертация",
                flags=_AUTOCONVERT,
                buffer_hns=0,
            )
        )

    add(
        _pcm16(
            "PCM16 16 кГц, моно + автоконвертация",
            preferred_rate,
            1,
            _AUTOCONVERT,
            0,
        )
    )
    add(
        _pcm16(
            "PCM16 16 кГц, моно",
            preferred_rate,
            1,
            _NOPERSIST,
            1_000_000,
        )
    )

    if mix is not None:
        channels = min(mix.channels, 2)
        add(
            _pcm16(
                f"PCM16 {mix.rate} Гц, {channels} кан.",
                mix.rate,
                channels,
                _NOPERSIST,
                1_000_000,
            )
        )
        add(
            _pcm16(
                f"PCM16 {mix.rate} Гц, {channels} кан. + автоконвертация",
                mix.rate,
                channels,
                _AUTOCONVERT,
                0,
            )
        )

    for rate in (44100, 48000):
        add(_pcm16(f"PCM16 {rate} Гц, стерео", rate, 2, _NOPERSIST, 1_000_000))
        add(
            _pcm16(
                f"PCM16 {rate} Гц, стерео + автоконвертация",
                rate,
                2,
                _AUTOCONVERT,
                0,
            )
        )
        add(
            _float32(
                f"float32 {rate} Гц, стерео + автоконвертация",
                rate,
                2,
                _AUTOCONVERT,
                0,
            )
        )

    return candidates


def _build_format(spec: _CaptureFormat):
    """Create a cffi-backed ``WAVEFORMATEXTENSIBLE`` for ``spec``."""
    if spec.raw is not None:
        buffer = _mf._ffi.new(f"char[{len(spec.raw)}]")
        _mf._ffi.memmove(buffer, spec.raw, len(spec.raw))
        return _mf._ffi.cast("WAVEFORMATEXTENSIBLE*", buffer)

    fmt = _mf._ffi.new("WAVEFORMATEXTENSIBLE*")
    fmt.Format.wFormatTag = spec.tag
    fmt.Format.nChannels = spec.channels
    fmt.Format.nSamplesPerSec = spec.rate
    fmt.Format.nAvgBytesPerSec = spec.rate * spec.block_align
    fmt.Format.nBlockAlign = spec.block_align
    fmt.Format.wBitsPerSample = spec.bits
    fmt.Format.cbSize = 22 if spec.extensible else 0
    if spec.extensible:
        fmt.Samples.wValidBitsPerSample = spec.valid_bits or spec.bits
        fmt.dwChannelMask = spec.channel_mask
        fmt.SubFormat.Data1 = int.from_bytes(spec.subformat[0:4], "little")
        fmt.SubFormat.Data2 = int.from_bytes(spec.subformat[4:6], "little")
        fmt.SubFormat.Data3 = int.from_bytes(spec.subformat[6:8], "little")
        for index, value in enumerate(spec.subformat[8:16]):
            fmt.SubFormat.Data4[index] = value
    return fmt


if HAS_MEDIA_FOUNDATION:

    class _Recorder(_mf._Recorder):
        """Recorder that accepts the negotiated stream format."""

        def __init__(self, ptr, spec: _CaptureFormat, seconds: float = 0.1):
            self._ptr = ptr
            self._spec = spec
            self._format_memory = _build_format(spec)
            hr = self._ptr[0][0].lpVtbl.Initialize(
                self._ptr[0],
                _SHARED_MODE,
                spec.flags,
                spec.buffer_hns,
                0,
                self._format_memory,
                _mf._ffi.NULL,
            )
            if _hresult_code(hr) != 0:
                code = _hresult_code(hr)
                raise CaptureError(
                    f"Initialize [{spec.summary()}]: {_hresult_text(code)}",
                    step="Initialize",
                    hresult=code,
                    variant=spec.label,
                )
            self.samplerate = spec.rate
            self.channelmap = list(range(spec.channels))
            self._sample_format = spec.sample_format
            self._bytes_per_frame = spec.block_align
            self._mix_channels = spec.channels
            self._idle_start_time = None
            self.opened_with = spec.label
            self.attempts: list[BaseException] = []

        @property
        def sample_format(self) -> str:
            return self._sample_format

        def _capture_client(self):
            iid = _mf._guidof("{C8ADBD64-E71E-48a0-A4DE-185C395CD317}")
            pp_capture_client = _mf._ffi.new("IAudioCaptureClient**")
            hr = self._ptr[0][0].lpVtbl.GetService(
                self._ptr[0], iid, _mf._ffi.cast("void**", pp_capture_client)
            )
            _check(hr, "GetService(IAudioCaptureClient)")
            return pp_capture_client

        def _capture_buffer(self):
            data = _mf._ffi.new("BYTE**")
            to_read = _mf._ffi.new("UINT32*")
            flags = _mf._ffi.new("DWORD*")
            hr = self._ppCaptureClient[0][0].lpVtbl.GetBuffer(
                self._ppCaptureClient[0],
                data,
                to_read,
                flags,
                _mf._ffi.NULL,
                _mf._ffi.NULL,
            )
            _check(hr, "IAudioCaptureClient::GetBuffer")
            return data[0], to_read[0], flags[0]

        def _capture_release(self, numframes: int) -> None:
            hr = self._ppCaptureClient[0][0].lpVtbl.ReleaseBuffer(
                self._ppCaptureClient[0], numframes
            )
            _check(hr, "IAudioCaptureClient::ReleaseBuffer")

        def _capture_available_frames(self) -> int:
            size = _mf._ffi.new("UINT32*")
            hr = self._ppCaptureClient[0][0].lpVtbl.GetNextPacketSize(
                self._ppCaptureClient[0], size
            )
            _check(hr, "IAudioCaptureClient::GetNextPacketSize")
            return size[0]

        def __enter__(self):
            self._ppCaptureClient = self._capture_client()
            hr = self._ptr[0][0].lpVtbl.Start(self._ptr[0])
            _check(hr, "Start")
            self._pending_chunk = np.zeros([0], dtype=np.float32)
            self._is_first_frame = True
            return self

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
            raise CaptureError(f"Неизвестный формат аудио: {self._sample_format}")

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
                raise CaptureError("IAudioCaptureClient::GetBuffer: пустой буфер")
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

    def _open_audio_client(microphone):
        try:
            return microphone._audio_client()
        except Exception as exc:
            raise CaptureError(f"Activate(IAudioClient): {exc}") from exc

    def read_mix_format(microphone) -> MixFormat:
        """Read and parse the endpoint mix format without opening a stream."""
        ptr = _open_audio_client(microphone)
        format_pointer = _mf._ffi.new("WAVEFORMATEXTENSIBLE**")
        try:
            hr = ptr[0][0].lpVtbl.GetMixFormat(ptr[0], format_pointer)
            _check(hr, "GetMixFormat")
            fmt = format_pointer[0][0]
            size = 18 + int(fmt.Format.cbSize)
            raw = bytes(_mf._ffi.buffer(_mf._ffi.addressof(fmt), size))
        finally:
            if format_pointer[0] != _mf._ffi.NULL:
                _mf._ole32.CoTaskMemFree(format_pointer[0])
            _mf._com.release(ptr)
        return parse_mix_format(raw)

    def open_recorder(
        microphone,
        *,
        seconds: float = 0.1,
        preferred_rate: int = 16000,
    ) -> "_Recorder":
        """Open a capture stream, trying several formats if needed."""
        errors: list[BaseException] = []
        mix = None
        try:
            mix = read_mix_format(microphone)
        except Exception as exc:
            errors.append(exc)
            logger.debug("GetMixFormat не удался: %s", exc)

        for spec in format_candidates(mix, preferred_rate=preferred_rate):
            try:
                ptr = _open_audio_client(microphone)
            except Exception as exc:
                errors.append(exc)
                break
            try:
                recorder = _Recorder(ptr, spec, seconds)
                # Validate GetService before the caller starts the stream.
                client = recorder._capture_client()
                _mf._com.release(client)
                recorder.attempts = list(errors)
                return recorder
            except Exception as exc:
                errors.append(exc)
                try:
                    _mf._com.release(ptr)
                except Exception:
                    pass

        details = _attempt_details(errors)
        raise CaptureError(
            "не удалось открыть поток"
            + (f": {details}" if details else "")
        )

    def probe_microphone(microphone, *, seconds: float = 0.2) -> list[str]:
        """Describe an endpoint and try to capture from it."""
        name = str(getattr(microphone, "name", "?"))
        lines: list[str] = [f"  {name}:"]
        try:
            mix = read_mix_format(microphone)
            lines.append(f"    формат устройства: {mix.describe()}")
        except Exception as exc:
            lines.append(f"    формат устройства: ошибка: {exc}")
        try:
            with open_recorder(microphone, seconds=0.1) as recorder:
                frames = max(256, int(recorder.samplerate * seconds))
                data = recorder.record(numframes=frames)
            lines.append(
                "    захват: OK — "
                f"{recorder.opened_with}, {recorder.samplerate} Гц, "
                f"{recorder._mix_channels} кан., {recorder.sample_format}, "
                f"{len(data)} отсчётов"
            )
        except Exception as exc:
            lines.append(f"    захват: ошибка: {exc}")
        return lines

else:

    def read_mix_format(microphone):  # pragma: no cover
        raise CaptureError("soundcard не установлен")

    def open_recorder(  # pragma: no cover
        microphone,
        *,
        seconds: float = 0.1,
        preferred_rate: int = 16000,
    ):
        raise CaptureError("soundcard не установлен — запись через Media Foundation недоступна")

    def probe_microphone(microphone, *, seconds: float = 0.2) -> list[str]:  # pragma: no cover
        return [f"  {microphone}: soundcard не установлен"]
