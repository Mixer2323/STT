"""Typed configuration with validation, migration and atomic persistence."""

from __future__ import annotations

import copy
import json
import logging
import os
import threading
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Callable

from .constants import (
    COMPUTE_TYPES,
    CONFIG_VERSION,
    DEFAULT_EXCLUDE_PHRASES,
    DEFAULT_HOTKEY,
    DEVICES,
    HOTKEY_MODES,
    INSERT_MODES,
    LANGUAGES,
    MULTILINGUAL_ENGINES,
    TONE_DECODERS,
    WHISPER_MODELS,
)

logger = logging.getLogger(__name__)

MODEL_TYPES = ("auto", "gigaam", "faster_whisper", "tone")


class ConfigError(ValueError):
    """Raised when a configuration value cannot be validated."""


@dataclass
class Config:
    version: int = CONFIG_VERSION
    hotkey: str = DEFAULT_HOTKEY
    hotkey_mode: str = "hold"  # hold | toggle
    language: str = "ru"
    model_type: str = "auto"
    whisper_model: str = "large-v3-turbo"
    tone_decoder: str = "auto"
    device: str = "auto"  # auto | cpu | cuda
    compute_type: str = "auto"
    beam_size: int = 5
    vad_filter: bool = True
    sample_rate: int = 16000
    input_device: int | str | None = None
    max_recording_seconds: int = 300
    exclude_phrases: list[str] = field(default_factory=lambda: list(DEFAULT_EXCLUDE_PHRASES))
    insert_mode: str = "paste"  # paste | clipboard | type
    restore_clipboard: bool = True
    paste_delay_ms: int = 80
    restore_target_window: bool = True
    trim_silence: bool = True
    capitalize: bool = True
    remove_fillers: bool = False
    history_enabled: bool = True
    history_limit: int = 200
    log_transcripts: bool = False
    notifications: bool = True
    streaming_preview: bool = True
    autostart: bool = False
    single_instance: bool = True


def _as_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("1", "true", "yes", "on"):
            return True
        if lowered in ("0", "false", "no", "off"):
            return False
    return default


def _as_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, number))


def _as_choice(value: Any, allowed: tuple[str, ...], default: str) -> str:
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in allowed:
            return lowered
    return default


def _normalize_phrases(value: Any) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple)):
        return list(DEFAULT_EXCLUDE_PHRASES)
    result: list[str] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, (str, int, float)):
            continue
        phrase = str(item).strip()
        if not phrase or len(phrase) > 200:
            continue
        key = phrase.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(phrase)
        if len(result) >= 100:
            break
    return result


def _normalize_device(value: Any) -> int | str | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.isdigit():
            return int(text)
        return text[:200]
    return None


def validate_mapping(data: dict[str, Any]) -> tuple[Config, list[str]]:
    """Build a config from raw JSON, coercing invalid values and reporting them."""
    warnings: list[str] = []
    raw = copy.deepcopy(data) if isinstance(data, dict) else {}
    if not isinstance(data, dict):
        warnings.append("Файл конфигурации повреждён: ожидался объект JSON")

    # Migration from the pre-1.1 flat schema.
    if "force_cpu" in raw:
        legacy_cpu = _as_bool(raw.get("force_cpu"), False)
        legacy_only = set(raw) <= {"force_cpu"}
        if legacy_cpu and legacy_only:
            warnings.append(
                "Старый флаг force_cpu сброшен: он записывался автоматическим "
                "fallback при ошибке CUDA"
            )
            raw["device"] = "auto"
        elif legacy_cpu:
            raw.setdefault("device", "cpu")
        del raw["force_cpu"]

    if "exclude_phrases" in raw:
        phrases = _normalize_phrases(raw["exclude_phrases"])
        if phrases != raw["exclude_phrases"]:
            raw["exclude_phrases"] = phrases

    cfg = Config()
    if "version" in raw:
        try:
            raw_version = int(raw["version"])
        except (TypeError, ValueError):
            raw_version = 0
        if raw_version > CONFIG_VERSION:
            warnings.append(
                f"Конфигурация новее приложения (v{raw_version}); "
                "неизвестные параметры будут проигнорированы"
            )

    def check(field_name: str, normalizer, message: str):
        if field_name not in raw:
            return
        current = getattr(cfg, field_name)
        normalized = normalizer(raw[field_name], current)
        if normalized != raw[field_name]:
            warnings.append(message)
        setattr(cfg, field_name, normalized)

    def choice(name: str, allowed: tuple[str, ...]):
        check(
            name,
            lambda value, default: _as_choice(value, allowed, default),
            f"Недопустимое значение {name!r}; использовано значение по умолчанию",
        )

    choice("hotkey_mode", HOTKEY_MODES)
    choice("language", LANGUAGES)
    choice("model_type", MODEL_TYPES)
    choice("whisper_model", WHISPER_MODELS)
    choice("tone_decoder", TONE_DECODERS)
    choice("device", DEVICES)
    choice("compute_type", COMPUTE_TYPES)
    choice("insert_mode", INSERT_MODES)

    # Only Whisper can recognise other languages; the Russian-only engines
    # ignore the setting, so keep it pinned to "ru" for them.
    if cfg.model_type not in MULTILINGUAL_ENGINES and cfg.language != "ru":
        if "language" in raw:
            warnings.append(
                "Выбор языка доступен только для Faster-Whisper; установлен русский"
            )
        cfg.language = "ru"

    if "hotkey" in raw:
        hotkey = str(raw["hotkey"]).strip() if raw["hotkey"] is not None else ""
        if not hotkey or len(hotkey) > 60:
            warnings.append("Некорректная горячая клавиша; оставлена по умолчанию")
        else:
            cfg.hotkey = hotkey

    check("beam_size", lambda v, d: _as_int(v, d, 1, 10), "beam_size вне диапазона 1..10")
    check(
        "sample_rate",
        lambda v, d: _as_int(v, d, 8000, 48000),
        "sample_rate вне диапазона 8000..48000",
    )
    check(
        "max_recording_seconds",
        lambda v, d: _as_int(v, d, 10, 3600),
        "max_recording_seconds вне диапазона 10..3600",
    )
    check(
        "paste_delay_ms",
        lambda v, d: _as_int(v, d, 0, 2000),
        "paste_delay_ms вне диапазона 0..2000",
    )
    check(
        "history_limit",
        lambda v, d: _as_int(v, d, 0, 10000),
        "history_limit вне диапазона 0..10000",
    )

    for name in (
        "vad_filter",
        "restore_clipboard",
        "restore_target_window",
        "trim_silence",
        "capitalize",
        "remove_fillers",
        "history_enabled",
        "log_transcripts",
        "notifications",
        "streaming_preview",
        "autostart",
        "single_instance",
    ):
        if name in raw:
            setattr(cfg, name, _as_bool(raw[name], getattr(cfg, name)))

    if "input_device" in raw:
        cfg.input_device = _normalize_device(raw["input_device"])
    if "exclude_phrases" in raw:
        cfg.exclude_phrases = _normalize_phrases(raw["exclude_phrases"])

    if cfg.history_limit == 0:
        cfg.history_enabled = False

    cfg.version = CONFIG_VERSION

    known = {item.name for item in fields(Config)}
    unknown = sorted(set(raw) - known)
    if unknown:
        warnings.append("Неизвестные параметры проигнорированы: " + ", ".join(unknown[:10]))
    return cfg, warnings


def _atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        with open(tmp, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass


class ConfigStore:
    """Thread-safe configuration holder with change notifications."""

    def __init__(self, path: Path | None = None, legacy_path: Path | None = None):
        from . import paths

        self.path = Path(path) if path else paths.config_path()
        self._legacy_path = Path(legacy_path) if legacy_path else None
        self._lock = threading.RLock()
        self._config = Config()
        self._listeners: list[Callable[[Config], None]] = []

    @property
    def config(self) -> Config:
        with self._lock:
            return self._config

    def get(self) -> Config:
        with self._lock:
            return copy.deepcopy(self._config)

    def snapshot(self) -> dict[str, Any]:
        return asdict(self.get())

    def subscribe(self, listener: Callable[[Config], None]) -> None:
        with self._lock:
            if listener not in self._listeners:
                self._listeners.append(listener)

    def _notify(self, cfg: Config) -> None:
        with self._lock:
            listeners = list(self._listeners)
        for listener in listeners:
            try:
                listener(copy.deepcopy(cfg))
            except Exception:
                logger.exception("Ошибка подписчика конфигурации")

    def load(self) -> list[str]:
        warnings: list[str] = []
        source = self.path
        migrated_from_legacy = False
        if not source.exists() and self._legacy_path and self._legacy_path.exists():
            source = self._legacy_path
            migrated_from_legacy = True

        raw: dict[str, Any] = {}
        if source.exists():
            try:
                with open(source, "r", encoding="utf-8-sig") as handle:
                    loaded = json.load(handle)
                if not isinstance(loaded, dict):
                    raise ValueError("корневой элемент не объект")
                raw = loaded
            except Exception as exc:
                warnings.append(f"Конфигурация повреждена: {exc}")
                backup = source.with_name(
                    f"{source.stem}.corrupt-{time.strftime('%Y%m%d-%H%M%S')}{source.suffix}"
                )
                try:
                    os.replace(source, backup)
                    warnings.append(f"Повреждённый файл сохранён как {backup.name}")
                except OSError:
                    logger.warning("Не удалось сохранить копию повреждённого конфига", exc_info=True)

        cfg, validation_warnings = validate_mapping(raw)
        warnings.extend(validation_warnings)

        with self._lock:
            self._config = cfg
        if source == self.path or migrated_from_legacy or not self.path.exists():
            self.save(cfg)
        if migrated_from_legacy:
            warnings.append(f"Настройки перенесены из {source}")
        return warnings

    def save(self, cfg: Config | None = None) -> bool:
        with self._lock:
            current = copy.deepcopy(cfg) if cfg is not None else copy.deepcopy(self._config)
            if cfg is not None:
                self._config = copy.deepcopy(cfg)
        payload = asdict(current)
        try:
            _atomic_json_write(self.path, payload)
            return True
        except Exception:
            logger.exception("Не удалось сохранить конфигурацию в %s", self.path)
            return False

    def update(self, **changes: Any) -> Config:
        with self._lock:
            merged = asdict(self._config)
        merged.update(changes)
        cfg, warnings = validate_mapping(merged)
        for warning in warnings:
            logger.warning("Конфигурация: %s", warning)
        with self._lock:
            self._config = cfg
        self.save(cfg)
        self._notify(cfg)
        return copy.deepcopy(cfg)

    def replace(self, cfg: Config) -> Config:
        validated, warnings = validate_mapping(asdict(cfg))
        for warning in warnings:
            logger.warning("Конфигурация: %s", warning)
        with self._lock:
            self._config = validated
        self.save(validated)
        self._notify(validated)
        return copy.deepcopy(validated)
