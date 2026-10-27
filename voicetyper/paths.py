"""Filesystem layout.

Small state (config, history, logs) lives in ``%LOCALAPPDATA%\\VoiceTyper`` by
default. A ``portable.txt`` marker or ``VOICETYPER_PORTABLE=1`` keeps all state
next to the executable. ``VOICETYPER_DATA_DIR`` and ``VOICETYPER_MODELS_DIR``
override individual locations (used by tests and advanced setups).
"""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


@lru_cache(maxsize=1)
def app_dir() -> Path:
    if is_frozen():
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


@lru_cache(maxsize=1)
def is_portable() -> bool:
    if os.environ.get("VOICETYPER_PORTABLE", "").strip() in ("1", "true", "yes"):
        return True
    return (app_dir() / "portable.txt").exists()


@lru_cache(maxsize=1)
def data_dir() -> Path:
    override = os.environ.get("VOICETYPER_DATA_DIR", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    if is_portable():
        return app_dir() / "data"

    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
        if base:
            return Path(base) / "VoiceTyper"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "VoiceTyper"
    base = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(base) / "VoiceTyper"


@lru_cache(maxsize=1)
def models_dir() -> Path:
    override = os.environ.get("VOICETYPER_MODELS_DIR", "").strip()
    if override:
        return Path(override).expanduser().resolve()

    legacy = app_dir() / "models"
    # Existing installations keep their multi-gigabyte downloads where they are.
    if legacy.is_dir():
        return legacy
    if is_portable():
        return data_dir() / "models"
    return data_dir() / "models"


@lru_cache(maxsize=1)
def logs_dir() -> Path:
    return data_dir() / "logs"


@lru_cache(maxsize=1)
def temp_dir() -> Path:
    return data_dir() / "tmp"


@lru_cache(maxsize=1)
def config_path() -> Path:
    return data_dir() / "config.json"


@lru_cache(maxsize=1)
def history_path() -> Path:
    return data_dir() / "history.db"


@lru_cache(maxsize=1)
def diagnostics_dir() -> Path:
    return data_dir() / "diagnostics"


def legacy_config_path() -> Path:
    return app_dir() / "config.json"


def resource_path(name: str) -> Path:
    """Return a bundled resource path (works in PyInstaller one-folder builds)."""
    base = getattr(sys, "_MEIPASS", None)
    if base:
        candidate = Path(base) / name
        if candidate.exists():
            return candidate
    return app_dir() / name


def ensure_runtime_dirs() -> None:
    for path in (data_dir(), logs_dir(), temp_dir()):
        path.mkdir(parents=True, exist_ok=True)


def ensure_models_dir() -> Path:
    path = models_dir()
    path.mkdir(parents=True, exist_ok=True)
    return path
