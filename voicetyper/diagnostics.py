"""Human-readable diagnostics without transcript content."""

from __future__ import annotations

import os
import platform
import sys
from datetime import datetime
from pathlib import Path

from . import paths
from .config import ConfigStore
from .constants import APP_VERSION
from .downloads import directory_size, human_size
from .engines import describe_engines
from .hardware import detect_hardware, format_bytes
from .history import HistoryStore
from .models import ModelManager


def _yes(value: bool) -> str:
    return "да" if value else "нет"


def collect_diagnostics(
    config_store: ConfigStore,
    models: ModelManager,
    history: HistoryStore,
) -> str:
    cfg = config_store.get()
    hw = detect_hardware()
    availability = describe_engines()
    lines: list[str] = []
    lines.append(f"Voice Typer {APP_VERSION}")
    lines.append(f"Сформировано: {datetime.now().astimezone().isoformat(timespec='seconds')}")
    lines.append("")
    lines.append("== Система ==")
    lines.append(f"ОС: {platform.platform()}")
    lines.append(f"Python: {platform.python_version()} ({sys.executable})")
    lines.append(f"Сборка PyInstaller: {_yes(getattr(sys, 'frozen', False))}")
    lines.append(f"Железо: {hw.summary()}")
    lines.append(f"Свободно на диске моделей: {format_bytes(hw.free_disk_gb * 1024**3) if hw.free_disk_gb else '?'}")
    lines.append("")

    lines.append("== Распознавание ==")
    lines.append(f"Конфигурация: движок={cfg.model_type}, язык={cfg.language}, "
                 f"устройство={cfg.device}, compute={cfg.compute_type}")
    lines.append(f"Текущая модель: {models.current_name or 'не загружена'}")
    runtime = models.current_runtime()
    if runtime:
        lines.append(f"Текущий runtime: {runtime}")
    status = models.status
    lines.append(f"Статус: {status.state} · {status.message}")
    for name, info in availability.items():
        lines.append(
            f"  {name}: установлен={_yes(info.installed)}, версия={info.version or '?'}"
        )
    lines.append("")

    lines.append("== Модели на диске ==")
    lines.append(f"Каталог: {paths.models_dir()}")
    try:
        if paths.models_dir().exists():
            for child in sorted(paths.models_dir().iterdir()):
                if child.is_dir():
                    lines.append(f"  {child.name}: {human_size(directory_size(child))}")
        else:
            lines.append("  каталог отсутствует")
    except OSError as exc:
        lines.append(f"  ошибка чтения: {exc}")
    lines.append("")

    lines.append("== Данные ==")
    lines.append(f"Данные: {paths.data_dir()}")
    lines.append(f"Логи: {paths.logs_dir()}")
    lines.append(f"Конфиг: {paths.config_path()}")
    lines.append(f"История: {paths.history_path()} ({history.count()} записей)")
    lines.append(f"Портативный режим: {_yes(paths.is_portable())}")
    lines.append(f"Логирование текста: {_yes(cfg.log_transcripts)}")
    lines.append("")

    lines.append("== Аудио ==")
    try:
        from .audio import default_input_device, list_input_devices

        devices = list_input_devices()
        default = default_input_device()
        lines.append(f"Микрофонов найдено: {len(devices)}")
        lines.append(f"По умолчанию: {default.label() if default else 'не определён'}")
        lines.append(f"Выбрано в конфиге: {cfg.input_device or 'системное по умолчанию'}")
        for device in devices[:8]:
            lines.append(f"  {device.index}: {device.label()}")
    except Exception as exc:
        lines.append(f"Ошибка опроса устройств: {exc}")
    lines.append("")

    lines.append("== Параметры ==")
    for field, value in sorted(cfg.__dict__.items()):
        if field == "exclude_phrases":
            lines.append(f"  {field}: {len(value)} фраз")
        elif field == "input_device":
            lines.append(f"  {field}: {value or 'default'}")
        else:
            lines.append(f"  {field}: {value}")
    lines.append("")
    lines.append("Содержимое распознавания и история в диагностику не включаются.")
    return "\n".join(lines)


def write_diagnostics(text: str) -> Path:
    directory = paths.diagnostics_dir()
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"diagnostics_{datetime.now():%Y%m%d_%H%M%S}.txt"
    destination.write_text(text + "\n", encoding="utf-8")
    return destination


def open_folder(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    if sys.platform == "win32":
        os.startfile(str(path))  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        import subprocess

        subprocess.Popen(["open", str(path)])
    else:
        import subprocess

        subprocess.Popen(["xdg-open", str(path)])
