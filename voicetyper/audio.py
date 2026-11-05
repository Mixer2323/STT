"""Microphone capture, WAV I/O, resampling and silence trimming."""

from __future__ import annotations

import logging
import math
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from . import paths
from .config import Config
from .constants import APP_VERSION

logger = logging.getLogger(__name__)

try:
    import sounddevice as sd

    HAS_SOUNDDEVICE = True
except Exception:
    sd = None
    HAS_SOUNDDEVICE = False

try:
    import soundfile as sf

    HAS_SOUNDFILE = True
except Exception:
    sf = None
    HAS_SOUNDFILE = False

try:
    import soundcard as sc

    HAS_SOUNDCARD = True
except Exception:
    sc = None
    HAS_SOUNDCARD = False

try:
    from .native_audio import (
        HAS_MEDIA_FOUNDATION,
        open_recorder,
        probe_microphone,
    )
except Exception:
    HAS_MEDIA_FOUNDATION = False
    open_recorder = None
    probe_microphone = None

try:
    import audioop

    HAS_AUDIOOP = True
except Exception:
    audioop = None
    HAS_AUDIOOP = False

try:
    import soxr

    HAS_SOXR = True
except Exception:
    soxr = None
    HAS_SOXR = False


class AudioError(RuntimeError):
    pass


@dataclass(frozen=True)
class InputDevice:
    index: int
    name: str
    channels: int
    default_samplerate: float
    hostapi: str = ""

    def label(self) -> str:
        rate = int(round(self.default_samplerate))
        suffix = f" · {self.hostapi}" if self.hostapi else ""
        return f"{self.name} ({rate} Гц){suffix}"


@dataclass
class RecordingResult:
    path: Optional[Path] = None
    duration: float = 0.0
    capture_rate: int = 16000
    overflowed: bool = False
    silent: bool = False
    error: Optional[str] = None
    peak_dbfs: Optional[float] = None

    @property
    def ok(self) -> bool:
        return self.path is not None and self.path.exists() and not self.error


def list_input_devices() -> list[InputDevice]:
    if not HAS_SOUNDDEVICE:
        return []
    devices: list[InputDevice] = []
    try:
        hostapis = sd.query_hostapis()
        for index, info in enumerate(sd.query_devices()):
            channels = int(info.get("max_input_channels", 0))
            if channels <= 0:
                continue
            api_index = int(info.get("hostapi", -1))
            api_name = ""
            if 0 <= api_index < len(hostapis):
                api_name = str(hostapis[api_index].get("name", ""))
            devices.append(
                InputDevice(
                    index=index,
                    name=str(info.get("name", f"Device {index}")),
                    channels=channels,
                    default_samplerate=float(info.get("default_samplerate", 0) or 0),
                    hostapi=api_name,
                )
            )
    except Exception:
        logger.warning("Не удалось перечислить микрофоны", exc_info=True)
    return devices


def default_input_device() -> Optional[InputDevice]:
    if not HAS_SOUNDDEVICE:
        return None
    try:
        index = sd.default.device[0]
    except Exception:
        return None
    for device in list_input_devices():
        if device.index == index:
            return device
        if index is None and device.name == "default":
            return device
    return None


def is_remote_session() -> bool:
    """True inside a Remote Desktop / terminal-services session."""
    if os.name != "nt":
        return False
    if os.environ.get("SESSIONNAME", "").strip().upper().startswith("RDP-"):
        return True
    try:
        import ctypes

        return bool(ctypes.windll.user32.GetSystemMetrics(0x1000))  # SM_REMOTESESSION
    except Exception:
        return False


def refresh_input_devices() -> bool:
    """Re-enumerate PortAudio devices.

    PortAudio caches the device list on initialization. In an RDP session the
    set of audio devices changes when the client connects or disconnects, so
    the cached list (and the default device index) can become stale.
    """
    if not HAS_SOUNDDEVICE:
        return False
    terminate = getattr(sd, "_terminate", None)
    initialize = getattr(sd, "_initialize", None)
    if terminate is None or initialize is None:
        return False
    try:
        terminate()
        initialize()
    except Exception:
        logger.warning("Не удалось обновить список аудиоустройств", exc_info=True)
        return False
    logger.debug("Список аудиоустройств обновлён")
    return True


def _system_default_input() -> Optional[int]:
    if not HAS_SOUNDDEVICE:
        return None
    try:
        default = sd.default.device[0]
    except Exception:
        return None
    try:
        index = int(default)
    except (TypeError, ValueError):
        return None
    return index if index >= 0 else None


def _wasapi_default_input() -> Optional[int]:
    """PortAudio index of the default WASAPI input device, if available.

    Windows apps normally capture through WASAPI (shared mode). PortAudio's
    default input, however, is usually the MME "Sound Mapper", and MME cannot
    open the RDP "Remote Audio" endpoint at all (MMSYSERR_INVALPARAM / error
    11), even though other applications record from it without problems.
    """
    if not HAS_SOUNDDEVICE:
        return None
    try:
        for hostapi in sd.query_hostapis():
            if "wasapi" not in str(hostapi.get("name", "")).casefold():
                continue
            index = int(hostapi.get("default_input_device", -1))
            return index if index >= 0 else None
    except Exception:
        logger.debug("Не удалось определить устройство WASAPI", exc_info=True)
    return None


def _same_device_name(first: str, second: str) -> bool:
    """Compare device names across host APIs, tolerating MME truncation."""
    a = first.strip().casefold()
    b = second.strip().casefold()
    if not a or not b:
        return False
    if a == b:
        return True
    short, long = (a, b) if len(a) <= len(b) else (b, a)
    return len(short) >= 12 and long.startswith(short)


def _wasapi_equivalent(device_index: Optional[int]) -> Optional[int]:
    """Find the WASAPI entry for the same physical device."""
    if not HAS_SOUNDDEVICE or device_index is None:
        return None
    try:
        name = str(sd.query_devices(device_index, "input").get("name", ""))
    except Exception:
        return None
    if not name:
        return None
    for device in list_input_devices():
        if device.index == device_index:
            continue
        if "wasapi" not in device.hostapi.casefold():
            continue
        if _same_device_name(device.name, name):
            return device.index
    return None


def _hostapi_name(device_index: Optional[int]) -> str:
    """Name of the host API that owns ``device_index``."""
    if not HAS_SOUNDDEVICE or device_index is None:
        return ""
    try:
        info = sd.query_devices(device_index, "input")
        api_index = int(info.get("hostapi", -1))
        hostapis = sd.query_hostapis()
        if 0 <= api_index < len(hostapis):
            return str(hostapis[api_index].get("name", ""))
    except Exception:
        pass
    return ""


def _is_wasapi_index(device_index: Optional[int]) -> bool:
    return "wasapi" in _hostapi_name(device_index).casefold()


def _device_name(device_index: Optional[int]) -> str:
    if not HAS_SOUNDDEVICE or device_index is None:
        return ""
    try:
        return str(sd.query_devices(device_index, "input").get("name", ""))
    except Exception:
        return ""


_MIXER_ENDPOINT_HINTS = ("микшер", "stereo mix", "what u hear", "mixer", "loopback")
_MICROPHONE_HINTS = (
    "микрофон",
    "microphone",
    "mic",
    "гарнитур",
    "headset",
    "головн",
)
# WDM-KS talks to the hardware directly, so it still sees the microphones
# attached to the server when an RDP session's WASAPI only exposes the
# redirected endpoint. MME and DirectSound mirror the session endpoints.
_LAST_RESORT_HOST_APIS = ("wdm-ks", "directsound", "mme")


def _is_mixer_endpoint(name: str) -> bool:
    lowered = name.casefold()
    return any(hint in lowered for hint in _MIXER_ENDPOINT_HINTS)


def _looks_like_microphone(name: str) -> bool:
    lowered = name.casefold()
    return any(hint in lowered for hint in _MICROPHONE_HINTS)


def _other_hostapi_devices() -> list[int]:
    """Non-WASAPI input indices, most likely to record voice first."""

    def rank(device: InputDevice) -> tuple[int, int]:
        api = device.hostapi.casefold()
        position = next(
            (index for index, name in enumerate(_LAST_RESORT_HOST_APIS) if name in api),
            len(_LAST_RESORT_HOST_APIS),
        )
        return position, 0 if _looks_like_microphone(device.name) else 1

    return [
        device.index
        for device in sorted(
            (
                device
                for device in list_input_devices()
                if "wasapi" not in device.hostapi.casefold()
                and not _is_mixer_endpoint(device.name)
            ),
            key=rank,
        )
    ]


def _is_server_side_fallback(device_index: Optional[int]) -> bool:
    """True when an RDP session ended up recording from the server's own mic."""
    if not is_remote_session():
        return False
    name = _device_name(device_index)
    return bool(name) and not _is_redirected_endpoint(name)


def _wasapi_exclusive_settings() -> object:
    """``WasapiSettings(exclusive=True)`` or ``None`` when unavailable."""
    if not HAS_SOUNDDEVICE:
        return None
    factory = getattr(sd, "WasapiSettings", None)
    if factory is None:
        return None
    try:
        return factory(exclusive=True)
    except Exception:
        return None


def _format_candidates(device_index: Optional[int], requested: int) -> list[tuple[int, int]]:
    """Return (samplerate, channels) pairs to try, most likely first.

    MME (the Windows host API PortAudio picks by default) can report a format
    as supported through ``check_input_settings`` yet still fail ``waveInOpen``
    with ``MMSYSERR_INVALPARAM`` (PortAudio's "MME error 11") for sample rates
    the driver does not really accept — typical for the RDP audio redirection
    device and Bluetooth hands-free mics. So we prefer the device's own
    default rate and also try mono/stereo.

    WASAPI devices are tried with stereo first: some virtual endpoints (the RDP
    "Remote Audio" device in particular) report a single channel but only open
    in shared mode when stereo is requested.
    """
    if not HAS_SOUNDDEVICE:
        return []
    default_rate = 0
    try:
        info = sd.query_devices(device_index, "input")
        default_rate = int(round(float(info.get("default_samplerate", 0) or 0)))
    except Exception:
        pass

    rates = [
        rate
        for rate in (default_rate, 48000, 44100, requested, 16000, 8000)
        if rate > 0
    ]
    if _is_wasapi_index(device_index):
        channel_options = [2, 1]
    else:
        channel_options = [1, 2]

    formats: list[tuple[int, int]] = []
    for rate in dict.fromkeys(rates):
        for channels in channel_options:
            formats.append((rate, channels))
    return formats


def _soundcard_microphones() -> list[object]:
    if not HAS_SOUNDCARD or os.name != "nt":
        return []
    try:
        return list(sc.all_microphones(include_loopback=False))
    except Exception:
        logger.debug("Не удалось перечислить микрофоны Windows Media Foundation", exc_info=True)
        return []


def _find_soundcard_microphone(names: list[str]) -> Optional[object]:
    """Pick a Media Foundation microphone matching one of ``names``."""
    if not HAS_SOUNDCARD or os.name != "nt":
        return None
    microphones = _soundcard_microphones()
    if not microphones:
        return None
    wanted = [name.strip() for name in names if name and name.strip()]
    for name in wanted:
        for microphone in microphones:
            microphone_name = str(getattr(microphone, "name", ""))
            if not microphone_name:
                continue
            if _same_device_name(microphone_name, name):
                return microphone
            if name.casefold() in microphone_name.casefold():
                return microphone
    try:
        return sc.default_microphone()
    except Exception:
        logger.debug("Не удалось выбрать микрофон по умолчанию", exc_info=True)
        return None


def resolve_input_device(value: int | str | None) -> Optional[int]:
    """Map a config value (index or name) to a sounddevice index.

    Device numbering is not stable across sessions: after an RDP client
    connects or disconnects the list is rebuilt and the saved index may point
    to another device or disappear. In that case fall back to the system
    default instead of making recording impossible.
    """
    if not HAS_SOUNDDEVICE:
        raise AudioError("sounddevice не установлен — запись недоступна")
    if value is None or value == "":
        return _system_default_input()

    devices = list_input_devices()
    if isinstance(value, int) or (isinstance(value, str) and value.strip().isdigit()):
        index = int(value)
        if any(device.index == index for device in devices):
            return index
        logger.warning(
            "Микрофон с индексом %s не найден — используется системный по умолчанию",
            index,
        )
        return _system_default_input()

    needle = str(value).strip().casefold()
    exact = [device for device in devices if device.name.casefold() == needle]
    if exact:
        return exact[0].index
    partial = [device for device in devices if needle and needle in device.name.casefold()]
    if len(partial) == 1:
        return partial[0].index
    if len(partial) > 1:
        names = ", ".join(device.name for device in partial[:4])
        raise AudioError(f"Название «{value}» подходит к нескольким микрофонам: {names}")
    logger.warning(
        "Микрофон «%s» не найден — используется системный по умолчанию", value
    )
    return _system_default_input()


def _no_input_device_message() -> str:
    message = "Микрофоны не найдены"
    if is_remote_session():
        message += (
            ". В сеансе RDP включите перенаправление микрофона: «Локальные ресурсы» → "
            "«Звук» → «Параметры» → «Запись с этого компьютера», затем перезапустите приложение"
        )
    return message + "."


def _capture_blockers() -> list[str]:
    """Windows rules that make capture impossible, if any (policy/privacy/service)."""
    if os.name != "nt":
        return []
    try:
        from .capture_env import collect

        return list(collect().blockers)
    except Exception:
        logger.debug("Не удалось проверить ограничения записи Windows", exc_info=True)
        return []


def _open_microphone_message(error: Optional[BaseException]) -> str:
    detail = str(error) if error else "устройство недоступно"
    message = f"Не удалось открыть микрофон: {detail}"
    blockers = _capture_blockers()
    if blockers:
        message += " " + blockers[0]
    elif is_remote_session():
        message += (
            " Проверьте перенаправление микрофона в RDP: «Локальные ресурсы» → "
            "«Звук» → «Параметры» → «Запись с этого компьютера»."
        )
    return message


def _resample_float(data: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr or data.size == 0:
        return data.astype(np.float32, copy=False)
    if HAS_SOXR:
        try:
            return np.asarray(soxr.resample(data, src_sr, dst_sr, quality="HQ"), dtype=np.float32)
        except Exception:
            logger.debug("soxr не смог выполнить ресэмплинг", exc_info=True)
    # Linear interpolation fallback. Adequate for speech at 16 kHz; soxr is
    # recommended in requirements for best quality.
    n_out = int(round(len(data) * dst_sr / src_sr))
    if n_out <= 1:
        return np.zeros(0, dtype=np.float32)
    x_old = np.arange(len(data), dtype=np.float64)
    x_new = np.arange(n_out, dtype=np.float64) * (src_sr / dst_sr)
    return np.interp(x_new, x_old, data.astype(np.float64)).astype(np.float32)


def _downmix(data: np.ndarray) -> np.ndarray:
    if data.ndim == 1:
        return data.astype(np.float32, copy=False)
    if data.shape[1] == 1:
        return data[:, 0].astype(np.float32, copy=False)
    return data.astype(np.float32, copy=False).mean(axis=1)


def read_audio(
    path: Path | str,
    target_sr: Optional[int] = None,
) -> tuple[np.ndarray, int]:
    if not HAS_SOUNDFILE:
        raise AudioError("soundfile не установлен — чтение аудио недоступно")
    try:
        data, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    except Exception as exc:
        raise AudioError(f"Не удалось прочитать {path}: {exc}") from exc
    mono = _downmix(data)
    rate = int(sample_rate)
    if target_sr and target_sr != rate:
        mono = _resample_float(mono, rate, int(target_sr))
        rate = int(target_sr)
    return mono, rate


def load_audio_int32_8k(path: Path | str) -> np.ndarray:
    mono, _rate = read_audio(path, target_sr=8000)
    return np.clip(np.rint(mono * 32768.0), -32768, 32767).astype(np.int32)


def save_wav(path: Path | str, data: np.ndarray, sample_rate: int) -> Path:
    if not HAS_SOUNDFILE:
        raise AudioError("soundfile не установлен — сохранение аудио недоступно")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(data)
    if array.dtype == np.int16:
        pass
    else:
        array = np.clip(array, -1.0, 1.0).astype(np.float32)
    tmp = destination.with_name(destination.name + ".tmp")
    try:
        sf.write(str(tmp), array, sample_rate, subtype="PCM_16", format="WAV")
        os.replace(tmp, destination)
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass
    return destination


def _frame_rms(data: np.ndarray, frame_size: int) -> np.ndarray:
    if data.size == 0:
        return np.zeros(0, dtype=np.float32)
    count = max(1, data.size // frame_size)
    trimmed = data[: count * frame_size].reshape(count, frame_size)
    return np.sqrt(np.mean(np.square(trimmed, dtype=np.float32), axis=1))


def peak_dbfs(data: np.ndarray) -> float:
    if data.size == 0:
        return -120.0
    peak = float(np.max(np.abs(data)))
    if peak <= 1e-9:
        return -120.0
    return 20.0 * math.log10(min(1.0, peak))


def trim_silence(
    data: np.ndarray,
    sample_rate: int,
    *,
    threshold_dbfs: float = -48.0,
    padding_ms: int = 120,
) -> np.ndarray:
    """Trim leading/trailing silence. Returns an empty array if no speech."""
    if data.size == 0:
        return data
    frame_size = max(1, int(sample_rate * 0.02))
    rms = _frame_rms(data, frame_size)
    if rms.size == 0:
        return data
    absolute = 10 ** (threshold_dbfs / 20.0)
    relative = max(float(np.max(rms)) * 0.03, absolute)
    active = np.flatnonzero(rms >= relative)
    if active.size == 0:
        return np.zeros(0, dtype=np.float32)
    padding = max(0, int(sample_rate * padding_ms / 1000.0))
    start = max(0, int(active[0]) * frame_size - padding)
    end = min(data.size, (int(active[-1]) + 1) * frame_size + padding)
    return data[start:end]


def is_effectively_silent(data: np.ndarray, sample_rate: int) -> bool:
    if data.size < int(sample_rate * 0.15):
        return True
    trimmed = trim_silence(data, sample_rate, threshold_dbfs=-50.0)
    return trimmed.size < int(sample_rate * 0.12)


class StreamingDownsampler:
    """Streaming int16 resampler used for T-one previews."""

    def __init__(self, source_rate: int, target_rate: int):
        self.source_rate = int(source_rate)
        self.target_rate = int(target_rate)
        self._state = None
        self._pending = b""

    def feed(self, pcm16: np.ndarray) -> np.ndarray:
        if pcm16.size == 0 or self.source_rate == self.target_rate:
            return np.asarray(pcm16, dtype=np.int16)
        raw = np.asarray(pcm16, dtype=np.int16).tobytes()
        if HAS_AUDIOOP:
            converted, self._state = audioop.ratecv(
                raw, 2, 1, self.source_rate, self.target_rate, self._state
            )
            return np.frombuffer(converted, dtype=np.int16).copy()
        # audioop was removed in Python 3.13; fall back to a numeric resampler.
        floats = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        output = _resample_float(floats, self.source_rate, self.target_rate)
        return np.clip(np.rint(output * 32767.0), -32768, 32767).astype(np.int16)

    def flush(self) -> np.ndarray:
        return np.zeros(0, dtype=np.int16)


class RecordingSession:
    """One push-to-talk recording. ``stop`` is idempotent and safe to call twice."""

    def __init__(
        self,
        cfg: Config,
        *,
        on_auto_stop: Optional[Callable[["RecordingSession"], None]] = None,
        on_level: Optional[Callable[[float], None]] = None,
        on_chunk: Optional[Callable[[np.ndarray, int], None]] = None,
    ):
        self.cfg = cfg
        self._on_auto_stop = on_auto_stop
        self._on_level = on_level
        self._on_chunk = on_chunk
        self._lock = threading.RLock()
        self._finalize_lock = threading.Lock()
        self._chunks: list[np.ndarray] = []
        self._frames = 0
        self._stream = None
        self._started = False
        self._stopped = False
        self._result: Optional[RecordingResult] = None
        self._capture_rate = int(cfg.sample_rate)
        self._capture_channels = 1
        self._auto_stop_fired = False
        self._last_level_at = 0.0
        self._overflowed = False
        self._native_thread: Optional[threading.Thread] = None
        self._native_stop: Optional[threading.Event] = None
        self._native_error: Optional[BaseException] = None
        self._native_mic_name = ""

    @property
    def is_recording(self) -> bool:
        with self._lock:
            return self._started and not self._stopped

    @property
    def capture_rate(self) -> int:
        return self._capture_rate

    @property
    def result(self) -> Optional[RecordingResult]:
        with self._lock:
            return self._result

    def _candidate_formats(self, device_index: Optional[int]) -> list[tuple[int, int]]:
        """Return (samplerate, channels) pairs to try, most likely first."""
        return _format_candidates(device_index, int(self.cfg.sample_rate))

    def _open_stream(
        self,
        device_index: Optional[int],
        rate: int,
        channels: int,
        *,
        exclusive: bool = False,
    ):
        assert sd is not None
        extra_settings = _wasapi_exclusive_settings() if exclusive else None
        if exclusive and extra_settings is None:
            raise RuntimeError("WASAPI exclusive mode недоступен")
        stream = sd.InputStream(
            device=device_index,
            samplerate=rate,
            channels=channels,
            dtype="int16",
            blocksize=0,
            callback=self._callback,
            extra_settings=extra_settings,
        )
        try:
            stream.start()
        except Exception:
            try:
                stream.close()
            except Exception:
                pass
            raise
        return stream

    def _candidate_devices(self, device_index: Optional[int]) -> list[Optional[int]]:
        """Device indices to try, best first.

        On Windows the WASAPI endpoint is preferred: it is what other
        applications use, and MME/DirectSound cannot open the RDP "Remote
        Audio" device. An explicitly selected device is tried first, with its
        WASAPI equivalent as a fallback. All WASAPI input devices are tried
        before that, because the default endpoint reported by PortAudio is not
        always the one that actually carries audio. Last come the devices of the
        other host APIs — in an RDP session those are the microphones physically
        attached to the server, which is better than not recording at all.
        """
        wasapi_devices = [
            device.index
            for device in list_input_devices()
            if "wasapi" in device.hostapi.casefold()
        ]
        wasapi_default = _wasapi_default_input()
        explicit = self.cfg.input_device not in (None, "")
        ordered: list[Optional[int]] = []

        def add(index: Optional[int]) -> None:
            if index is not None and index not in ordered:
                ordered.append(index)

        if explicit:
            add(device_index)
            add(_wasapi_equivalent(device_index))
            add(wasapi_default)
            for index in wasapi_devices:
                add(index)
        else:
            add(wasapi_default)
            for index in wasapi_devices:
                add(index)
            add(device_index)
            add(_system_default_input())
        for index in _other_hostapi_devices():
            add(index)
        if not ordered:
            ordered.append(None)
        return ordered

    def _open_first_available(self) -> tuple[object, int, int, Optional[int]]:
        assert sd is not None
        device_index = resolve_input_device(self.cfg.input_device)
        last_error: Optional[BaseException] = None

        for attempt in range(2):
            if attempt:
                if not refresh_input_devices():
                    break
                # Device indices can shift after the PortAudio restart.
                device_index = resolve_input_device(self.cfg.input_device)

            candidates = self._candidate_devices(device_index)
            logger.debug(
                "Кандидаты микрофона: %s",
                ", ".join("default" if c is None else str(c) for c in candidates),
            )
            for candidate in candidates:
                # WASAPI shared mode is the normal path; exclusive mode is the
                # last resort for endpoints that refuse shared-mode formats.
                exclusive_modes = [False]
                if candidate is not None and _is_wasapi_index(candidate):
                    exclusive_modes.append(True)
                candidate_error: Optional[BaseException] = None
                for exclusive in exclusive_modes:
                    for rate, channels in self._candidate_formats(candidate):
                        try:
                            stream = self._open_stream(
                                candidate, rate, channels, exclusive=exclusive
                            )
                        except Exception as exc:
                            last_error = exc
                            candidate_error = exc
                            logger.debug(
                                "Не удалось открыть микрофон (%s, %d Гц, %d кан., %s): %s",
                                candidate if candidate is not None else "default",
                                rate,
                                channels,
                                "exclusive" if exclusive else "shared",
                                exc,
                            )
                            continue
                        if attempt:
                            logger.info(
                                "Список аудиоустройств обновлён — микрофон открыт заново"
                            )
                        logger.info(
                            "Микрофон открыт: устройство=%s (%s), %d Гц, каналов=%d, %s",
                            candidate if candidate is not None else "default",
                            _hostapi_name(candidate) or "default host API",
                            rate,
                            channels,
                            "exclusive" if exclusive else "shared",
                        )
                        if _is_server_side_fallback(candidate):
                            logger.warning(
                                "Перенаправленный микрофон RDP не открылся — запись идёт "
                                "с устройства, подключённого к серверу: %s",
                                _device_name(candidate),
                            )
                        return stream, rate, channels, candidate
                logger.warning(
                    "Микрофон %s (%s) не открылся: %s",
                    candidate if candidate is not None else "default",
                    _hostapi_name(candidate) or "default host API",
                    candidate_error,
                )

        if not list_input_devices():
            raise AudioError(_no_input_device_message())
        for device in list_input_devices():
            logger.warning("Микрофон %s: %s", device.index, device.label())
        for blocker in _capture_blockers():
            logger.warning("Ограничение записи Windows: %s", blocker)
        logger.warning(
            "Не удалось открыть ни один микрофон; последняя ошибка: %s", last_error
        )
        raise AudioError(_open_microphone_message(last_error))

    def _native_capture_names(self, device_index: Optional[int]) -> list[str]:
        names: list[str] = []
        for index in self._candidate_devices(device_index):
            if index is None:
                continue
            try:
                name = str(sd.query_devices(index, "input").get("name", ""))
            except Exception:
                continue
            if name and name not in names:
                names.append(name)
        return names

    def _start_native_capture(self) -> bool:
        """Fall back to Windows Media Foundation capture.

        PortAudio's WASAPI backend refuses to open some virtual endpoints,
        notably the RDP "Remote Audio" device, even though Media Foundation —
        the stack Windows applications use — records from it without problems.
        """
        if not HAS_SOUNDCARD or not HAS_MEDIA_FOUNDATION or os.name != "nt":
            return False
        device_index = resolve_input_device(self.cfg.input_device)
        microphone = _find_soundcard_microphone(self._native_capture_names(device_index))
        if microphone is None:
            logger.warning("Windows Media Foundation не нашёл ни одного микрофона")
            return False

        stop_event = threading.Event()
        ready = threading.Event()
        self._native_stop = stop_event
        self._native_error = None
        self._capture_rate = int(self.cfg.sample_rate)
        self._capture_channels = 1
        self._native_mic_name = str(getattr(microphone, "name", "default"))
        thread = threading.Thread(
            target=self._native_capture_loop,
            args=(microphone, stop_event, ready),
            name="native-audio-capture",
            daemon=True,
        )
        self._native_thread = thread
        thread.start()
        if not ready.wait(timeout=10.0):
            logger.warning("Windows Media Foundation не успел открыть микрофон")
            stop_event.set()
            self._native_thread = None
            return False
        if self._native_error is not None:
            logger.warning(
                "Windows Media Foundation не смог открыть микрофон %s: %s",
                self._native_mic_name,
                self._native_error,
            )
            self._native_thread = None
            return False
        return True

    def _native_capture_loop(
        self,
        microphone,
        stop_event: threading.Event,
        ready: threading.Event,
    ) -> None:
        com_initialized = False
        try:
            import ctypes

            ctypes.windll.ole32.CoInitializeEx(None, 0)
            com_initialized = True
        except Exception:
            logger.debug("Не удалось инициализировать COM в потоке записи", exc_info=True)

        try:
            with open_recorder(
                microphone,
                seconds=0.1,
                preferred_rate=int(self.cfg.sample_rate),
            ) as recorder:
                self._capture_rate = int(recorder.samplerate)
                rate = self._capture_rate
                blocksize = max(256, int(rate * 0.1))
                for attempt in getattr(recorder, "attempts", []):
                    logger.info(
                        "Media Foundation: вариант не подошёл: %s", attempt
                    )
                logger.info(
                    "Media Foundation открыл поток: %s, %d Гц, %d кан., %s",
                    getattr(recorder, "opened_with", "?"),
                    recorder.samplerate,
                    recorder._mix_channels,
                    recorder.sample_format,
                )
                ready.set()
                while not stop_event.is_set():
                    data = recorder.record(numframes=blocksize)
                    if data is None or len(data) == 0:
                        continue
                    if getattr(data, "ndim", 1) > 1:
                        data = data[:, 0]
                    samples = np.clip(
                        np.asarray(data, dtype=np.float32), -1.0, 1.0
                    )
                    chunk = np.rint(samples * 32767.0).astype(np.int16)
                    with self._lock:
                        if self._stopped:
                            break
                        self._chunks.append(chunk)
                        self._frames += len(chunk)
                        total_seconds = self._frames / max(1, self._capture_rate)
                        should_auto_stop = (
                            total_seconds >= self.cfg.max_recording_seconds
                            and not self._auto_stop_fired
                        )
                        if should_auto_stop:
                            self._auto_stop_fired = True
                    if self._on_chunk is not None:
                        try:
                            self._on_chunk(chunk, self._capture_rate)
                        except Exception:
                            logger.debug("Ошибка потокового колбэка", exc_info=True)
                    now = time.monotonic()
                    if self._on_level is not None and now - self._last_level_at > 0.1:
                        self._last_level_at = now
                        try:
                            self._on_level(
                                peak_dbfs(chunk.astype(np.float32) / 32768.0)
                            )
                        except Exception:
                            pass
                    if should_auto_stop:
                        logger.info(
                            "Достигнут лимит записи %d с",
                            self.cfg.max_recording_seconds,
                        )
                        target = self._on_auto_stop or (
                            lambda session: session.stop()
                        )
                        threading.Thread(
                            target=target,
                            args=(self,),
                            name="auto-stop-recording",
                            daemon=True,
                        ).start()
        except Exception as exc:
            self._native_error = exc
            ready.set()
            if not stop_event.is_set():
                logger.warning(
                    "Ошибка записи через Windows Media Foundation: %s", exc
                )
        finally:
            if com_initialized:
                try:
                    ctypes.windll.ole32.CoUninitialize()
                except Exception:
                    pass

    def start(self) -> None:
        if not HAS_SOUNDDEVICE:
            raise AudioError("sounddevice не установлен — запись недоступна")
        with self._lock:
            if self._started:
                return
            try:
                stream, rate, channels, device_index = self._open_first_available()
            except AudioError:
                if not self._start_native_capture():
                    raise
                self._started = True
                self._stopped = False
                logger.info(
                    "Запись начата через Windows Media Foundation: устройство=%s, "
                    "%d Гц, каналов=%d",
                    self._native_mic_name,
                    self._capture_rate,
                    self._capture_channels,
                )
                return
            self._stream = stream
            self._capture_rate = rate
            self._capture_channels = channels
            self._started = True
            self._stopped = False
        logger.info(
            "Запись начата: устройство=%s, %d Гц, каналов=%d",
            device_index if device_index is not None else "default",
            self._capture_rate,
            self._capture_channels,
        )

    def _callback(self, indata, frames, _time_info, status) -> None:
        if status:
            self._overflowed = True
            logger.debug("Статус аудиопотока: %s", status)
        chunk = indata[:, 0].copy() if indata.ndim > 1 else indata.copy()
        with self._lock:
            if self._stopped:
                return
            self._chunks.append(chunk)
            self._frames += len(chunk)
            total_seconds = self._frames / max(1, self._capture_rate)
            should_auto_stop = (
                total_seconds >= self.cfg.max_recording_seconds and not self._auto_stop_fired
            )
            if should_auto_stop:
                self._auto_stop_fired = True
        if self._on_chunk is not None:
            try:
                self._on_chunk(chunk, self._capture_rate)
            except Exception:
                logger.debug("Ошибка потокового колбэка", exc_info=True)
        now = time.monotonic()
        if self._on_level is not None and now - self._last_level_at > 0.1:
            self._last_level_at = now
            try:
                self._on_level(peak_dbfs(chunk.astype(np.float32) / 32768.0))
            except Exception:
                pass
        if should_auto_stop:
            logger.info("Достигнут лимит записи %d с", self.cfg.max_recording_seconds)
            target = self._on_auto_stop or (lambda session: session.stop())
            threading.Thread(
                target=target, args=(self,), name="auto-stop-recording", daemon=True
            ).start()

    def stop(self) -> RecordingResult:
        with self._finalize_lock:
            return self._stop_locked()

    def _stop_locked(self) -> RecordingResult:
        with self._lock:
            if self._result is not None:
                return self._result
            if not self._started:
                result = RecordingResult(error="Запись не была начата")
                self._result = result
                return result
            self._stopped = True
            stream = self._stream
            self._stream = None
            native_thread = self._native_thread
            native_stop = self._native_stop
            self._native_thread = None
            self._native_stop = None

        if native_stop is not None:
            native_stop.set()
        if stream is not None:
            try:
                stream.stop()
            except Exception:
                logger.debug("Ошибка остановки потока", exc_info=True)
            try:
                stream.close()
            except Exception:
                logger.debug("Ошибка закрытия потока", exc_info=True)
        if native_thread is not None:
            native_thread.join(timeout=5.0)
            if native_thread.is_alive():
                logger.warning("Поток Windows Media Foundation не завершился")

        with self._lock:
            if self._result is not None:
                return self._result
            chunks = self._chunks
            self._chunks = []
            frames = self._frames
            capture_rate = self._capture_rate
            overflowed = self._overflowed

        result = RecordingResult(
            capture_rate=capture_rate,
            overflowed=overflowed,
            duration=frames / max(1, capture_rate),
        )
        try:
            if not chunks or frames == 0:
                result.error = "Нет записанного аудио"
                result.silent = True
                with self._lock:
                    self._result = result
                return result

            pcm = np.concatenate(chunks).astype(np.int16, copy=False)
            data = pcm.astype(np.float32) / 32768.0
            result.peak_dbfs = peak_dbfs(data)
            if capture_rate != int(self.cfg.sample_rate):
                data = _resample_float(data, capture_rate, int(self.cfg.sample_rate))
                capture_rate = int(self.cfg.sample_rate)
                result.capture_rate = capture_rate

            if self.cfg.trim_silence:
                data = trim_silence(data, capture_rate)
            if data.size < int(capture_rate * 0.12) or is_effectively_silent(data, capture_rate):
                result.silent = True
                result.error = "Тишина — распознавание пропущено"
                with self._lock:
                    self._result = result
                return result

            stem = f"rec_{datetime.now():%Y%m%d_%H%M%S}_{os.getpid()}_{int(time.time() * 1000) % 1000}"
            path = paths.temp_dir() / f"{stem}.wav"
            paths.temp_dir().mkdir(parents=True, exist_ok=True)
            save_wav(path, data, capture_rate)
            result.path = path
            result.duration = len(data) / capture_rate
            logger.info(
                "Запись сохранена: %.2f с, %s, пик %.1f dBFS",
                result.duration,
                path.name,
                result.peak_dbfs if result.peak_dbfs is not None else -120.0,
            )
        except Exception as exc:
            logger.exception("Не удалось сохранить запись")
            result.error = f"Не удалось сохранить запись: {exc}"
        finally:
            with self._lock:
                self._result = result
        return result

    def abort(self) -> None:
        result = self.stop()
        if result.path:
            try:
                result.path.unlink(missing_ok=True)
            except OSError:
                pass
            result.path = None


def _grouped_error_lines(errors: list[tuple[int, int, str]]) -> list[str]:
    """Every refused format, grouped by the message it produced.

    PortAudio repeats the same host error for each rate/channel attempt, so the
    previous "last error only" summary hid the fact that some variants failed
    for a different reason.
    """
    if not errors:
        return ["        форматы не проверялись"]
    groups: dict[str, list[str]] = {}
    for rate, channels, message in errors:
        groups.setdefault(message, []).append(f"{rate} Гц/{channels} кан.")
    return [
        f"        {', '.join(variants)}: {message}"
        for message, variants in groups.items()
    ]


def _capture_env():
    """The Windows capture environment snapshot (policy, privacy, services)."""
    from .capture_env import collect

    return collect()


_REDIRECTED_KEYWORDS = ("удаленн", "remote audio", "перенаправл", "rdp")


def _is_redirected_endpoint(name: str) -> bool:
    lowered = name.casefold()
    return any(keyword in lowered for keyword in _REDIRECTED_KEYWORDS)


def _verdict_lines(
    *,
    is_rdp: bool,
    opened: list[str],
    failed_redirect: list[str],
    local_opened: list[str],
    redirect_seen: bool,
    mf_text: str,
    blockers: list[str],
) -> list[str]:
    """Say in plain words what the device table means, so the user knows what to fix."""
    lines = ["== Вывод =="]
    if blockers:
        lines.append(
            "  Причина в ограничениях Windows — см. строки ВНИМАНИЕ в разделе "
            "«Ограничения записи Windows»."
        )
        return lines
    if is_rdp and not redirect_seen:
        lines.append(
            "  Перенаправленного микрофона RDP в сеансе нет — сервер не создал "
            "конечную точку записи, потому что клиент не запросил переадресацию: "
            "в клиенте RDP «Локальные ресурсы» → «Звук» → «Параметры» → «Запись с "
            "этого компьютера» = «Этот компьютер» (в .rdp-файле audiocapturemode:i:1)."
        )
        if opened:
            lines.append(
                "  С микрофона, подключённого к серверу, запись доступна: "
                + "; ".join((local_opened or opened)[:3])
            )
        return lines
    if failed_redirect and not opened:
        lines.append(
            "  Не открывается ни одно устройство записи. Проверьте, что в сеансе "
            "работает служба Windows Audio и что микрофон разрешён в параметрах "
            "конфиденциальности Windows."
        )
        return lines
    if is_rdp and failed_redirect:
        lines.append(
            "  Перенаправленный микрофон RDP не открывается ("
            + "; ".join(failed_redirect)
            + "), при этом другие устройства работают ("
            + "; ".join(opened)
            + ")."
        )
        if "Initialize" in mf_text and "E_INVALIDARG" in mf_text:
            lines.append(
                "  Конечная точка существует и отвечает на GetMixFormat, но "
                "IAudioClient::Initialize возвращает E_INVALIDARG даже для её "
                "собственного формата: сервер точку создал, а звук от клиента не "
                "приходит."
            )
        lines.append(
            "  Проверять нужно клиент, а не сервер — на нём ограничения записи не "
            "найдены. На машине, с которой выполняется подключение: разрешите доступ "
            "к микрофону для «Подключение к удалённому рабочему столу» (Параметры → "
            "Конфиденциальность → Микрофон), включите «Запись с этого компьютера» = "
            "«Этот компьютер» (audiocapturemode:i:1) и убедитесь, что микрофон на "
            "клиенте работает. Клиенты для macOS, iOS, Android и браузер переадресацию "
            "записи не поддерживают."
        )
        if local_opened:
            lines.append(
                "  До настройки перенаправления приложение будет записывать с "
                "микрофона сервера: "
                + "; ".join(local_opened[:3])
                + " — говорить нужно в него."
            )
        return lines
    if opened:
        lines.append(
            "  Запись работает хотя бы на одном устройстве: " + "; ".join(opened[:3])
        )
    else:
        lines.append("  Очевидной причины не видно — пришлите этот отчёт разработчику.")
    return lines


def probe_input_devices(cfg: Optional[Config] = None) -> str:
    """Try to open every input device and describe the result.

    Written for ``VoiceTyper.exe --audio-test``. The windowed PyInstaller build
    has no console, so the report is written to a file and the user can send it
    back. Each device is opened in shared mode and, for WASAPI devices, in
    exclusive mode; the first working format is reported.
    """
    cfg = cfg or Config()
    lines = [
        f"Voice Typer {APP_VERSION} — проверка микрофонов",
        f"Время: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"Сеанс RDP: {'да' if is_remote_session() else 'нет'}",
        f"Частота из настроек: {int(cfg.sample_rate)} Гц",
        f"Микрофон из настроек: {cfg.input_device or 'системный по умолчанию'}",
        "",
    ]
    if not HAS_SOUNDDEVICE:
        lines.append("sounddevice не установлен — проверка невозможна.")
        return "\n".join(lines) + "\n"

    refresh_input_devices()
    devices = list_input_devices()
    default = default_input_device()
    lines.append(f"Микрофонов найдено: {len(devices)}")
    lines.append(f"Системный по умолчанию: {default.label() if default else 'не определён'}")
    wasapi_default = _wasapi_default_input()
    lines.append(
        "WASAPI по умолчанию: "
        + (str(wasapi_default) if wasapi_default is not None else "не определён")
    )
    lines.append("")

    session = RecordingSession(cfg)
    try:
        device_index = resolve_input_device(cfg.input_device)
        candidates = session._candidate_devices(device_index)
        lines.append("Порядок, в котором приложение пробует устройства:")
        lines.append(
            "  "
            + " → ".join(
                "default" if c is None else f"{c} ({_hostapi_name(c) or '?'})"
                for c in candidates
            )
        )
    except Exception as exc:
        lines.append(f"Не удалось построить список кандидатов: {exc}")
    lines.append("")

    env = _capture_env() if os.name == "nt" else None
    blockers = list(env.blockers) if env is not None else []
    if env is not None:
        lines.extend(env.report_lines())
        lines.append("")

    opened_labels: list[str] = []
    local_opened: list[str] = []
    failed_redirect: list[str] = []
    redirect_seen = False
    for device in devices:
        redirected = _is_redirected_endpoint(device.name)
        redirect_seen = redirect_seen or redirected
        lines.append(f"[{device.index}] {device.label()} | каналов: {device.channels}")
        wasapi = "wasapi" in device.hostapi.casefold()
        modes = (False, True) if wasapi else (False,)
        device_ok = False
        for exclusive in modes:
            mode = "exclusive" if exclusive else "shared"
            errors: list[tuple[int, int, str]] = []
            opened: Optional[tuple[int, int]] = None
            for rate, channels in session._candidate_formats(device.index):
                try:
                    stream = session._open_stream(
                        device.index, rate, channels, exclusive=exclusive
                    )
                except Exception as exc:
                    errors.append((rate, channels, str(exc).strip()))
                    continue
                opened = (rate, channels)
                try:
                    stream.stop()
                    stream.close()
                except Exception:
                    logger.debug("Ошибка закрытия тестового потока", exc_info=True)
                break
            if opened is not None:
                lines.append(
                    f"    {mode}: OK — {opened[0]} Гц, {opened[1]} каналов"
                )
                device_ok = True
                opened_labels.append(f"[{device.index}] {device.name}")
                if not redirected:
                    local_opened.append(f"[{device.index}] {device.name}")
                break
            count = len(errors)
            label = "формат" if count == 1 else "форматов"
            lines.append(f"    {mode}: не удалось — {count} {label}:")
            lines.extend(_grouped_error_lines(errors))
            with session._lock:
                session._chunks.clear()
                session._frames = 0
        if not device_ok:
            lines.append("    ИТОГ: устройство не открывается")
            if redirected:
                failed_redirect.append(f"[{device.index}] {device.name}")
        lines.append("")

    mf_lines: list[str] = []
    if os.name == "nt":
        mf_lines.append("== Windows Media Foundation (WASAPI) ==")
        microphones = _soundcard_microphones()
        if not microphones or not HAS_MEDIA_FOUNDATION or probe_microphone is None:
            mf_lines.append("  Микрофоны не найдены или модуль soundcard недоступен")
        else:
            for microphone in microphones:
                mf_lines.extend(probe_microphone(microphone, seconds=0.2))
        mf_lines.append("")
        lines.extend(mf_lines)

    remote = is_remote_session()
    lines.extend(
        _verdict_lines(
            is_rdp=remote,
            opened=opened_labels,
            failed_redirect=failed_redirect,
            local_opened=local_opened,
            redirect_seen=redirect_seen,
            mf_text="\n".join(mf_lines),
            blockers=blockers,
        )
    )
    lines.append("")
    if not opened_labels and not blockers and not (remote and failed_redirect):
        lines.append(
            "Если ни одно устройство не открылось, проверьте перенаправление микрофона "
            "в клиенте RDP («Локальные ресурсы» → «Звук» → «Параметры» → «Запись с "
            "этого компьютера») и разрешение доступа к микрофону в параметрах Windows."
        )
    return "\n".join(lines) + "\n"

