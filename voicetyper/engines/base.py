"""Common engine interfaces."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Protocol, Sequence

import numpy as np


class EngineError(RuntimeError):
    """A user-facing engine failure (loading or transcription)."""


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    text: str


@dataclass
class TranscriptionResult:
    text: str
    segments: list[Segment] = field(default_factory=list)
    language: Optional[str] = None
    duration: float = 0.0
    engine: str = ""


ProgressReporter = Callable[[str, Optional[float]], None]


class StreamSession(Protocol):
    def feed_pcm16(self, pcm16: np.ndarray, sample_rate: int) -> str:
        """Consume int16 mono audio and return the current preview text."""

    def finish(self) -> str:
        """Finalize the stream and return the preview text."""

    def close(self) -> None:
        ...


class Engine:
    """Base class for speech engines loaded in a background thread."""

    name = "base"
    label = "Base"
    supports_streaming = False
    # ``None`` means the engine handles every language.
    languages: Optional[Sequence[str]] = None

    def __init__(
        self,
        *,
        progress: Optional[ProgressReporter] = None,
        cancel_event: Optional[threading.Event] = None,
    ):
        self._progress = progress
        self._cancel = cancel_event or threading.Event()

    def report(self, message: str, percent: Optional[float] = None) -> None:
        if self._progress is not None:
            try:
                self._progress(message, percent)
            except Exception:
                pass

    def check_cancelled(self) -> None:
        if self._cancel.is_set():
            raise EngineError("Загрузка отменена")

    def transcribe(self, path: Path, language: Optional[str] = None) -> TranscriptionResult:
        raise NotImplementedError

    def start_stream(self) -> Optional[StreamSession]:
        return None

    def release(self) -> None:
        """Drop references to native resources."""

    def __enter__(self) -> "Engine":
        return self

    def __exit__(self, *_exc) -> None:
        self.release()


def is_language_supported(engine: Engine, language: Optional[str]) -> bool:
    if not language or language == "auto" or engine.languages is None:
        return True
    return language in engine.languages
