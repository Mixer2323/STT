"""GigaAM v3 engine (Russian, ONNX via gigastt)."""

from __future__ import annotations

import gc
import logging
import os
import shutil
from pathlib import Path
from typing import Optional

from .. import paths
from ..constants import GIGASTT_ASSETS, GIGASTT_SOURCES
from ..downloads import DownloadCancelled, DownloadError, download_file, human_size
from ..model_packs import download_pack
from .base import Engine, EngineError, Segment, TranscriptionResult

logger = logging.getLogger(__name__)


def _sources() -> list[tuple[str, str, dict[str, tuple[str, Optional[int], Optional[str]]]]]:
    sources = list(GIGASTT_SOURCES)
    mirror = os.environ.get("VOICETYPER_GIGASTT_MIRROR", "").strip().rstrip("/")
    if mirror:
        # A custom mirror is expected to serve the files under their local names.
        sources.insert(
            0,
            ("зеркало", mirror, {name: (name, None, None) for name in GIGASTT_ASSETS}),
        )
    return sources


def _models_ready(model_dir: Path) -> bool:
    if not model_dir.is_dir():
        return False
    names = {item.name for item in model_dir.iterdir() if item.is_file()}
    has_onnx = any(name.endswith(".onnx") and (model_dir / name).stat().st_size > 0 for name in names)
    has_vocab = any(name.endswith(".txt") and (model_dir / name).stat().st_size > 0 for name in names)
    return has_onnx and has_vocab


def _word_segments(words) -> list[Segment]:
    segments: list[Segment] = []
    current_text: list[str] = []
    start: Optional[float] = None
    end: Optional[float] = None
    for word in words or []:
        text = str(getattr(word, "text", "") or "").strip()
        if not text:
            continue
        word_start = float(getattr(word, "start_s", 0.0) or 0.0)
        word_end = float(getattr(word, "end_s", word_start) or word_start)
        if start is None:
            start = word_start
        elif word_start - (end or word_start) > 0.8 and current_text:
            segments.append(Segment(start=start, end=end or word_start, text=" ".join(current_text)))
            current_text = []
            start = word_start
        current_text.append(text)
        end = word_end
    if current_text and start is not None:
        segments.append(Segment(start=start, end=end or start, text=" ".join(current_text)))
    return segments


class GigaAMEngine(Engine):
    name = "gigaam"
    label = "GigaAM v3"
    supports_streaming = False
    languages = ("ru",)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._engine = None
        self._model_dir = paths.models_dir() / "gigastt"
        self._load()

    def _download(self) -> None:
        self._model_dir.mkdir(parents=True, exist_ok=True)
        try:
            free = shutil.disk_usage(self._model_dir).free
            if free < 700 * 1024 * 1024:
                logger.warning("Мало места для GigaAM: свободно %s", human_size(free))
        except OSError:
            pass

        try:
            download_pack("gigastt", self._model_dir, report=self.report, cancel_event=self._cancel)
        except DownloadCancelled:
            raise
        except DownloadError as exc:
            logger.warning("GigaAM: пакет с GitHub недоступен (%s); пофайловая загрузка", exc)
        if _models_ready(self._model_dir):
            self.report("GigaAM: модель готова", 100.0)
            return

        total = len(GIGASTT_ASSETS)
        sources = _sources()
        for index, name in enumerate(GIGASTT_ASSETS):
            self.check_cancelled()
            destination = self._model_dir / name
            if destination.exists() and destination.stat().st_size > 0:
                self.report(f"GigaAM: {name} уже загружен", (index + 1) * 100 / total)
                continue

            failures: list[str] = []
            for source_name, base_url, assets in sources:
                remote_name, expected_size, expected_sha256 = assets.get(
                    name, (name, None, None)
                )
                url = f"{base_url}/{remote_name}"

                def on_progress(progress, _index=index, _name=name, _source=source_name):
                    percent = progress.percent
                    overall = ((_index + (percent or 0) / 100.0) / total) * 100.0
                    speed = f" · {human_size(progress.speed_bps)}/с" if progress.speed_bps else ""
                    if percent is None:
                        self.report(f"GigaAM: {_name} · {_source}", overall)
                    else:
                        self.report(
                            f"GigaAM: {_name} {percent:.0f}%{speed} · {_source}", overall
                        )

                try:
                    download_file(
                        url,
                        destination,
                        expected_size=expected_size,
                        expected_sha256=expected_sha256,
                        progress=on_progress,
                        cancel_event=self._cancel,
                    )
                except DownloadCancelled:
                    raise
                except DownloadError as exc:
                    failures.append(f"{source_name}: {exc}")
                    logger.warning(
                        "GigaAM: источник %s не подошёл для %s: %s", source_name, name, exc
                    )
                    continue
                self.report(f"GigaAM: {name} готов", (index + 1) * 100 / total)
                break
            else:
                raise EngineError("Не удалось скачать GigaAM: " + "; ".join(failures))

    def _load(self) -> None:
        try:
            import gigastt_uniffi as giga
        except Exception as exc:
            raise EngineError(
                "GigaAM не установлен. Установите пакет gigastt: pip install gigastt"
            ) from exc

        self.report("GigaAM: проверка моделей")
        if not _models_ready(self._model_dir):
            self._download()
        self.check_cancelled()
        if not _models_ready(self._model_dir):
            raise EngineError(f"Файлы GigaAM не найдены в {self._model_dir}")

        self.report("GigaAM: загрузка в память")
        try:
            self._engine = giga.Engine.new_with_pool_size(str(self._model_dir), 1)
        except Exception as exc:
            raise EngineError(f"Не удалось загрузить GigaAM: {exc}") from exc
        self.report("GigaAM готова", 100.0)
        logger.info("GigaAM v3 загружена из %s", self._model_dir)

    def transcribe(self, path: Path, language: Optional[str] = None) -> TranscriptionResult:
        if self._engine is None:
            raise EngineError("GigaAM не загружена")
        try:
            transcript = self._engine.transcribe_file(str(path))
        except Exception as exc:
            raise EngineError(f"GigaAM не смогла распознать аудио: {exc}") from exc
        text = str(getattr(transcript, "text", "") or "").strip()
        words = getattr(transcript, "words", []) or []
        duration = float(getattr(transcript, "duration_s", 0.0) or 0.0)
        return TranscriptionResult(
            text=text,
            segments=_word_segments(words),
            language="ru",
            duration=duration,
            engine=self.name,
        )

    def release(self) -> None:
        self._engine = None
        gc.collect()
