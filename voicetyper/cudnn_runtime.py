"""Locate and preload a complete cuDNN 9 runtime before CTranslate2 loads.

The ``ctranslate2`` wheel ships only the cuDNN 9 loader stub
(``cudnn64_9.dll``). The stub needs the split libraries
(``cudnn_ops64_9.dll``, ``cudnn_cnn64_9.dll``, ...) to be discoverable; when
they are missing, Windows terminates the process with a fail-fast error
(``0xc0000409``) inside the native transcription instead of raising an
exception Python could handle. Development environments often hide the problem
because torch ships its own cuDNN 9 copy, but the release build excludes torch.

The engine therefore preloads a complete system cuDNN 9 by absolute path
before ``faster_whisper`` is imported, and uses CPU when none is installed.
"""

from __future__ import annotations

import ctypes
import logging
import os
import re
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_LOADER = "cudnn64_9.dll"
_REQUIRED = (
    "cudnn_ops64_9.dll",
    "cudnn_cnn64_9.dll",
    "cudnn_graph64_9.dll",
    "cudnn_heuristic64_9.dll",
    "cudnn_engines_precompiled64_9.dll",
)

# Keep library and directory handles alive for the whole process. Dropping the
# add_dll_directory handle would remove the directory from the search path.
_KEEP: list[object] = []
_resolved = False
_directory: Optional[Path] = None


def _expand(value: str) -> list[Path]:
    path = Path(value)
    return [path, path / "bin"]


def _version_key(path: Path) -> tuple[int, ...]:
    return tuple(int(part) for part in re.findall(r"\d+", path.name))


def _candidate_dirs() -> list[Path]:
    candidates: list[Path] = []
    for variable in ("CUDNN_PATH", "CUDNN_HOME"):
        value = os.environ.get(variable)
        if value:
            candidates.extend(_expand(value))
    for variable in ("CUDA_PATH", "CUDA_HOME"):
        value = os.environ.get(variable)
        if value:
            candidates.append(Path(value) / "bin")
    for variable in ("ProgramW6432", "ProgramFiles"):
        base = os.environ.get(variable)
        if not base:
            continue
        root = Path(base) / "NVIDIA" / "CUDNN"
        try:
            versions = sorted(root.glob("v9*"), key=_version_key, reverse=True)
        except OSError:
            versions = []
        for version in versions:
            candidates.append(version / "bin")
            try:
                candidates.extend(
                    child for child in (version / "bin").iterdir() if child.is_dir()
                )
            except OSError:
                continue
    for entry in os.environ.get("PATH", "").split(os.pathsep):
        if entry:
            candidates.append(Path(entry))

    unique: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        try:
            key = str(path.resolve()).lower()
        except OSError:
            key = str(path).lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def _is_complete(directory: Path) -> bool:
    try:
        names = {item.name.lower() for item in directory.iterdir() if item.is_file()}
    except OSError:
        return False
    return _LOADER in names and all(name in names for name in _REQUIRED)


def find_cudnn_directory() -> Optional[Path]:
    """Return the first folder holding a complete cuDNN 9 install."""
    for directory in _candidate_dirs():
        if _is_complete(directory):
            return directory
    return None


def _preload(directory: Path) -> bool:
    try:
        win_dll = ctypes.WinDLL
    except AttributeError:
        return False

    try:
        _KEEP.append(os.add_dll_directory(str(directory)))
    except (AttributeError, OSError):
        pass

    try:
        libraries = sorted(
            (
                item
                for item in directory.iterdir()
                if item.is_file()
                and item.name.lower().startswith("cudnn")
                and item.name.lower().endswith("_9.dll")
            ),
            # Dependencies first, the loader stub last.
            key=lambda item: (item.name.lower() == _LOADER, item.name.lower()),
        )
    except OSError:
        return False

    loaded: set[str] = set()
    for library in libraries:
        try:
            _KEEP.append(win_dll(str(library)))
            loaded.add(library.name.lower())
        except OSError as exc:
            logger.debug("cuDNN: не удалось загрузить %s: %s", library, exc)

    missing = [name for name in (_LOADER, *_REQUIRED) if name not in loaded]
    if missing:
        logger.warning(
            "cuDNN: в %s не загрузились %s", directory, ", ".join(missing)
        )
        return False
    return True


def ensure_cudnn() -> Optional[Path]:
    """Preload system cuDNN 9 once and return its folder, or ``None``."""
    global _resolved, _directory
    if _resolved:
        return _directory
    _resolved = True

    if sys.platform != "win32":
        return None

    directory = find_cudnn_directory()
    if directory is None:
        logger.warning("cuDNN 9 не найден: CUDA для Whisper недоступна")
        return None
    if not _preload(directory):
        return None

    _directory = directory
    logger.info("cuDNN 9 предзагружен из %s", directory)
    return _directory
