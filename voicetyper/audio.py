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


def _open_microphone_message(error: Optional[BaseException]) -> str:
    detail = str(error) if error else "устройство недоступно"
    message = f"Не удалось открыть микрофон: {detail}"
    if is_remote_session():
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
        """Return (samplerate, channels) pairs to try, most likely first.

        MME (the Windows host API PortAudio picks by default) can report a
        format as supported through ``check_input_settings`` yet still fail
        ``waveInOpen`` with ``MMSYSERR_INVALPARAM`` (PortAudio's "MME error
        11") for sample rates the driver does not really accept — typical for
        the RDP audio redirection device and Bluetooth hands-free mics. So we
        prefer the device's own default rate and also try mono/stereo.
        """
        assert sd is not None
        requested = int(self.cfg.sample_rate)
        default_rate = 0
        max_channels = 1
        try:
            info = sd.query_devices(device_index, "input")
            default_rate = int(round(float(info.get("default_samplerate", 0) or 0)))
            max_channels = max(1, int(info.get("max_input_channels", 1) or 1))
        except Exception:
            pass

        rates = [
            rate
            for rate in (default_rate, 48000, 44100, requested, 16000, 8000)
            if rate > 0
        ]
        channel_options = [1]
        if max_channels >= 2:
            channel_options.append(2)

        formats: list[tuple[int, int]] = []
        for rate in dict.fromkeys(rates):
            for channels in channel_options:
                formats.append((rate, channels))
        return formats

    def _open_stream(self, device_index: Optional[int], rate: int, channels: int):
        assert sd is not None
        stream = sd.InputStream(
            device=device_index,
            samplerate=rate,
            channels=channels,
            dtype="int16",
            blocksize=0,
            callback=self._callback,
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

            for rate, channels in self._candidate_formats(device_index):
                try:
                    stream = self._open_stream(device_index, rate, channels)
                except Exception as exc:
                    last_error = exc
                    logger.debug(
                        "Не удалось открыть микрофон (%s, %d Гц, %d кан.): %s",
                        device_index if device_index is not None else "default",
                        rate,
                        channels,
                        exc,
                    )
                    continue
                if attempt:
                    logger.info(
                        "Список аудиоустройств обновлён — микрофон открыт заново"
                    )
                return stream, rate, channels, device_index

        if not list_input_devices():
            raise AudioError(_no_input_device_message())
        raise AudioError(_open_microphone_message(last_error))

    def start(self) -> None:
        if not HAS_SOUNDDEVICE:
            raise AudioError("sounddevice не установлен — запись недоступна")
        with self._lock:
            if self._started:
                return
            stream, rate, channels, device_index = self._open_first_available()
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

        if stream is not None:
            try:
                stream.stop()
            except Exception:
                logger.debug("Ошибка остановки потока", exc_info=True)
            try:
                stream.close()
            except Exception:
                logger.debug("Ошибка закрытия потока", exc_info=True)

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
