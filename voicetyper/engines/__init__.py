"""Speech engine implementations and selection."""

from .base import (
    Engine,
    EngineError,
    Segment,
    StreamSession,
    TranscriptionResult,
    is_language_supported,
)
from .registry import (
    ENGINE_CLASSES,
    create_engine,
    describe_engines,
    resolve_engine_name,
)

__all__ = [
    "Engine",
    "EngineError",
    "Segment",
    "StreamSession",
    "TranscriptionResult",
    "is_language_supported",
    "ENGINE_CLASSES",
    "create_engine",
    "describe_engines",
    "resolve_engine_name",
]
