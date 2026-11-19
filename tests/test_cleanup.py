"""Startup cleanup of files left behind by a crashed run."""

from __future__ import annotations

import os
import time
from pathlib import Path

from voicetyper import cleanup, paths


def _make_old(path: Path, age_seconds: float) -> None:
    stamp = time.time() - age_seconds
    os.utime(path, (stamp, stamp))


def test_stale_recordings_are_removed(data_dir: Path) -> None:
    paths.temp_dir().mkdir(parents=True, exist_ok=True)
    old = paths.temp_dir() / "rec_20200101_000000_1_1.wav"
    fresh = paths.temp_dir() / "rec_29990101_000000_1_1.wav"
    other = paths.temp_dir() / "notes.txt"
    for item in (old, fresh, other):
        item.write_bytes(b"x")
    _make_old(old, cleanup.TEMP_MAX_AGE_SECONDS + 60)
    _make_old(other, cleanup.TEMP_MAX_AGE_SECONDS + 60)

    removed_temp, _removed_parts = cleanup.cleanup_stale_files()

    assert removed_temp == 1
    assert not old.exists()
    assert fresh.exists()
    # Only recordings/temporaries are touched; unrelated files stay.
    assert other.exists()


def test_fresh_recordings_are_kept(data_dir: Path) -> None:
    paths.temp_dir().mkdir(parents=True, exist_ok=True)
    fresh = paths.temp_dir() / "rec_now.wav"
    fresh.write_bytes(b"x")

    removed_temp, _ = cleanup.cleanup_stale_files()

    assert removed_temp == 0
    assert fresh.exists()


def test_stale_partial_downloads_are_removed(data_dir: Path) -> None:
    models = paths.models_dir()
    models.mkdir(parents=True, exist_ok=True)
    part = models / "model.onnx.abc12345.part"
    part.write_bytes(b"x")
    _make_old(part, cleanup.PART_MAX_AGE_SECONDS + 60)

    removed_temp, removed_parts = cleanup.cleanup_stale_files()

    assert removed_temp == 0
    assert removed_parts == 1
    assert not part.exists()


def test_orphan_hf_caches_are_detected_and_removed(data_dir: Path) -> None:
    models = paths.models_dir()
    orphan = models / "models--Systran--faster-whisper-tiny"
    (orphan / "snapshots").mkdir(parents=True)
    (orphan / "snapshots" / "model.bin").write_bytes(b"x" * 100)
    current = models / "faster_whisper" / "Systran--faster-whisper-tiny"
    current.mkdir(parents=True)
    (current / "model.bin").write_bytes(b"current")

    assert cleanup.find_orphan_model_caches() == [orphan]
    assert cleanup.orphan_model_cache_size() >= 100

    removed, freed = cleanup.cleanup_orphan_model_caches()

    assert removed == 1
    assert freed >= 100
    assert not orphan.exists()
    # Current model layout is never touched.
    assert (current / "model.bin").read_bytes() == b"current"
