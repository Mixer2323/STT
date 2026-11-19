"""Startup housekeeping for files that outlive a crashed run.

Recordings are deleted after transcription, but a crash or a forced power-off
leaves the WAV in ``tmp/`` forever — a privacy problem (raw voice on disk) and a
slow disk leak. Partially downloaded model files are handled the same way: old
``*.part`` files are dropped so a mirror switch or an aborted download cannot
fill the models folder.
"""

from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path

from . import paths
from .downloads import cleanup_partials, directory_size

logger = logging.getLogger(__name__)

# A recording left behind by a crashed run is useless after a few hours; a
# pending transcription is finished within minutes, and the app holds no file
# for longer than that in normal operation.
TEMP_MAX_AGE_SECONDS = 6 * 60 * 60
PART_MAX_AGE_SECONDS = 24 * 60 * 60


def _cleanup_temp(directory: Path, max_age_seconds: float, now: float) -> int:
    removed = 0
    try:
        candidates = list(directory.iterdir())
    except OSError:
        return 0
    for item in candidates:
        if not item.is_file() or item.suffix.lower() not in (".wav", ".tmp"):
            continue
        try:
            if now - item.stat().st_mtime <= max_age_seconds:
                continue
            item.unlink()
            removed += 1
        except OSError:
            logger.debug("Не удалось удалить временный файл %s", item, exc_info=True)
    return removed


def cleanup_stale_files(
    *,
    temp_max_age: float = TEMP_MAX_AGE_SECONDS,
    part_max_age: float = PART_MAX_AGE_SECONDS,
) -> tuple[int, int]:
    """Remove stale recordings and partial downloads. Returns ``(wav, part)``."""
    now = time.time()
    removed_temp = _cleanup_temp(paths.temp_dir(), temp_max_age, now)
    try:
        removed_parts = cleanup_partials(paths.models_dir(), part_max_age)
    except OSError:
        removed_parts = 0
    if removed_temp or removed_parts:
        logger.info(
            "Удалены устаревшие файлы: записей %d, частичных загрузок %d",
            removed_temp,
            removed_parts,
        )
    return removed_temp, removed_parts


def find_orphan_model_caches() -> list[Path]:
    """Hugging Face cache directories left by the pre-1.0.13 layout.

    Models now live in ``faster_whisper/<repo>``; older versions kept a full HF
    cache under ``models/models--*``. Nothing reads those directories any more,
    so they only consume disk space (several gigabytes for large-v3).
    """
    root = paths.models_dir()
    if not root.is_dir():
        return []
    try:
        return sorted(
            item for item in root.iterdir() if item.is_dir() and item.name.startswith("models--")
        )
    except OSError:
        logger.debug("Не удалось прочитать каталог моделей", exc_info=True)
        return []


def orphan_model_cache_size(caches: list[Path] | None = None) -> int:
    items = find_orphan_model_caches() if caches is None else caches
    return sum(directory_size(item) for item in items)


def cleanup_orphan_model_caches() -> tuple[int, int]:
    """Delete orphan HF caches. Returns ``(removed_count, freed_bytes)``."""
    caches = find_orphan_model_caches()
    removed = 0
    freed = 0
    for cache in caches:
        size = directory_size(cache)
        try:
            shutil.rmtree(cache)
        except OSError as exc:
            logger.warning("Не удалось удалить старый кэш %s: %s", cache, exc)
            continue
        removed += 1
        freed += size
    if removed:
        logger.info(
            "Удалены старые кэши моделей: %d, освобождено %d байт", removed, freed
        )
    return removed, freed
