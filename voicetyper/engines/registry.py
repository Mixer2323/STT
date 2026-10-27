"""Engine factory and automatic selection."""

from __future__ import annotations

import importlib
import logging
import threading
from typing import Callable, Optional

from ..config import Config
from ..hardware import (
    EngineAvailability,
    HardwareInfo,
    choose_engine_name,
    engine_availability,
)
from .base import Engine

logger = logging.getLogger(__name__)

ENGINE_CLASSES = {
    "gigaam": "voicetyper.engines.gigaam:GigaAMEngine",
    "faster_whisper": "voicetyper.engines.whisper:WhisperEngine",
    "tone": "voicetyper.engines.tone:ToneEngine",
}


def describe_engines() -> dict[str, EngineAvailability]:
    return engine_availability()


def resolve_engine_name(cfg: Config, hw: HardwareInfo) -> str:
    return choose_engine_name(cfg, hw, engine_availability())


def create_engine(
    name: str,
    cfg: Config,
    hw: HardwareInfo,
    *,
    progress: Optional[Callable[[str, Optional[float]], None]] = None,
    cancel_event: Optional[threading.Event] = None,
) -> Engine:
    target = ENGINE_CLASSES.get(name)
    if not target:
        raise ValueError(f"Неизвестный движок: {name}")
    module_name, _, class_name = target.partition(":")
    module = importlib.import_module(module_name)
    engine_class = getattr(module, class_name)
    if name in ("faster_whisper", "tone"):
        return engine_class(cfg=cfg, hw=hw, progress=progress, cancel_event=cancel_event)
    return engine_class(progress=progress, cancel_event=cancel_event)
