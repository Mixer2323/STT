"""Prebuilt model archives published in GitHub Releases.

Pulling one ZIP from GitHub is typically much faster than downloading dozens of
files from Hugging Face, so the engines try a pack first and keep the per-file
sources as a fallback.
"""

from __future__ import annotations

import logging
import os
import shutil
import threading
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from .downloads import DownloadCancelled, DownloadError, DownloadProgress, download_file, human_size

logger = logging.getLogger(__name__)

GITHUB_REPO = "mixer232323-web/STT"
MODELS_RELEASE_TAG = "models-v1"
MODELS_RELEASE_URL = f"https://github.com/{GITHUB_REPO}/releases/tag/{MODELS_RELEASE_TAG}"
MODELS_DOWNLOAD_BASE = f"https://github.com/{GITHUB_REPO}/releases/download/{MODELS_RELEASE_TAG}"

ReportCallback = Callable[[str, Optional[float]], None]


@dataclass(frozen=True)
class ModelPack:
    key: str
    title: str
    asset: str
    size: int
    sha256: str
    folder: tuple[str, ...]
    files: tuple[str, ...]

    @property
    def url(self) -> str:
        return f"{MODELS_DOWNLOAD_BASE}/{self.asset}"

    @property
    def folder_display(self) -> str:
        return "\\".join(self.folder)


PACKS: dict[str, ModelPack] = {
    pack.key: pack
    for pack in (
        ModelPack(
            key="gigastt",
            title="GigaAM v3",
            asset="VoiceTyper-models-gigastt-v3.zip",
            size=326_991_948,
            sha256="767ee52601ea00216b5b7cc347aed994052f1229842e97c147bca3a1c15219dc",
            folder=("gigastt",),
            files=(
                "v3_e2e_rnnt_encoder_int8.onnx",
                "v3_e2e_rnnt_decoder.onnx",
                "v3_e2e_rnnt_joint.onnx",
                "v3_e2e_rnnt_vocab.txt",
            ),
        ),
        ModelPack(
            key="tone",
            title="T-one",
            asset="VoiceTyper-models-tone-model.zip",
            size=144_193_489,
            sha256="28ca4d27680bf8785616f003b42a5afc60653a41ae17eb4d499629c53fdf05ef",
            folder=("tone",),
            files=("model.onnx",),
        ),
        ModelPack(
            key="whisper-large-v3-turbo",
            title="Faster-Whisper large-v3-turbo",
            asset="VoiceTyper-models-whisper-large-v3-turbo.zip",
            size=1_621_666_531,
            sha256="75a1e4bf40adf58dde7a219110670c887982aab84b5f61902deede1c7846c893",
            folder=("faster_whisper", "mobiuslabsgmbh--faster-whisper-large-v3-turbo"),
            files=(
                "config.json",
                "model.bin",
                "preprocessor_config.json",
                "tokenizer.json",
                "vocabulary.json",
            ),
        ),
    )
}

# Mirrors used when GitHub Releases is unreachable. Per-file downloads from
# these sources are handled by the engines themselves.
BACKUP_URLS: dict[str, tuple[str, ...]] = {
    "gigastt": (
        "https://huggingface.co/istupakov/gigaam-v3-onnx",
        "https://github.com/ekhodzitsky/gigastt/releases/tag/models-v3-2026-10-09",
    ),
    "tone": ("https://huggingface.co/t-tech/T-one",),
    "whisper-large-v3-turbo": (
        "https://huggingface.co/mobiuslabsgmbh/faster-whisper-large-v3-turbo",
    ),
}


def pack_ready(pack: ModelPack, target_dir: Path) -> bool:
    return all(
        (target_dir / name).is_file() and (target_dir / name).stat().st_size > 0
        for name in pack.files
    )


def _report(callback: Optional[ReportCallback], message: str, percent: Optional[float]) -> None:
    if callback is None:
        return
    try:
        callback(message, percent)
    except Exception:
        logger.debug("Ошибка колбэка прогресса пакета", exc_info=True)


def _extract(
    archive_path: Path,
    target_dir: Path,
    pack: ModelPack,
    cancel_event: Optional[threading.Event],
) -> None:
    expected = set(pack.files)
    with zipfile.ZipFile(archive_path) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            name = info.filename
            path = Path(name)
            if path.is_absolute() or len(path.parts) != 1 or ".." in path.parts:
                raise DownloadError(f"Недопустимый путь в архиве {pack.asset}: {name}")
            if name not in expected:
                continue
            if cancel_event is not None and cancel_event.is_set():
                raise DownloadCancelled("Загрузка отменена")
            destination = target_dir / name
            temp = destination.with_name(destination.name + ".tmp")
            try:
                with archive.open(info) as source, open(temp, "wb") as handle:
                    shutil.copyfileobj(source, handle, 8 * 1024 * 1024)
                os.replace(temp, destination)
            finally:
                temp.unlink(missing_ok=True)

    missing = [name for name in pack.files if not (target_dir / name).is_file()]
    if missing:
        raise DownloadError(f"В архиве {pack.asset} нет файлов: {', '.join(missing)}")


def download_pack(
    key: str,
    target_dir: Path | str,
    *,
    report: Optional[ReportCallback] = None,
    cancel_event: Optional[threading.Event] = None,
) -> Path:
    """Download and unpack ``key`` into ``target_dir``.

    Raises :class:`DownloadError` when the archive cannot be fetched or unpacked;
    callers are expected to fall back to the per-file sources.
    """
    pack = PACKS[key]
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    if pack_ready(pack, target_dir):
        _report(report, f"{pack.title}: файлы уже на месте", 100.0)
        return target_dir

    archive_path = target_dir / f".{pack.asset}.download"

    def on_progress(progress: DownloadProgress) -> None:
        percent = progress.percent
        speed = f" · {human_size(progress.speed_bps)}/с" if progress.speed_bps else ""
        if percent is None:
            _report(report, f"{pack.title}: {progress.message or 'скачивание'}{speed}", None)
        else:
            _report(
                report,
                f"{pack.title}: {progress.message or 'скачивание'} "
                f"{percent:.0f}%{speed} · {human_size(pack.size)}",
                percent,
            )

    try:
        download_file(
            pack.url,
            archive_path,
            expected_size=pack.size,
            expected_sha256=pack.sha256,
            progress=on_progress,
            cancel_event=cancel_event,
        )
        _report(report, f"{pack.title}: распаковка архива", None)
        _extract(archive_path, target_dir, pack, cancel_event)
    finally:
        archive_path.unlink(missing_ok=True)

    _report(report, f"{pack.title}: модель готова", 100.0)
    return target_dir
