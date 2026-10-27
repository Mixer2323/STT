"""Faster-Whisper engine.

The model is downloaded once into ``models/faster_whisper/<repo>`` and loaded
from that directory, never through Hugging Face cache symlinks. This avoids the
``WinError 1314`` failure on Windows accounts without symlink privileges and
makes the model folder self-contained.
"""

from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Optional

from .. import paths
from ..config import Config
from ..constants import WHISPER_FILES, WHISPER_REPOS
from ..downloads import DownloadCancelled, DownloadError, download_hf_files, human_size
from ..hardware import HardwareInfo, resolve_whisper_runtime
from ..model_packs import download_pack
from .base import Engine, EngineError, Segment, TranscriptionResult

logger = logging.getLogger(__name__)

_CUDA_MARKERS = (
    "cuda",
    "cublas",
    "cudnn",
    "cudart",
    "nvidia",
    "gpu",
    "no kernel image",
    "device or backend",
)


def _looks_like_cuda_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(marker in message for marker in _CUDA_MARKERS)


class WhisperEngine(Engine):
    name = "faster_whisper"
    label = "Faster-Whisper"
    supports_streaming = False
    languages = None

    def __init__(self, cfg: Config, hw: HardwareInfo, **kwargs):
        super().__init__(**kwargs)
        self.cfg = cfg
        self.hw = hw
        self.model = None
        self.warnings: list[str] = []
        self.device = "cpu"
        self.compute_type = "int8"
        self.model_dir = self._local_dir()
        self._load()

    def _local_dir(self) -> Path:
        repo = WHISPER_REPOS.get(self.cfg.whisper_model, WHISPER_REPOS["large-v3-turbo"])
        safe = repo.replace("/", "--")
        return paths.models_dir() / "faster_whisper" / safe

    def _ensure_files(self) -> None:
        repo = WHISPER_REPOS.get(self.cfg.whisper_model, WHISPER_REPOS["large-v3-turbo"])
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.report(f"Whisper: проверка {repo}", 0.0)

        def progress(message: str, percent: Optional[float]) -> None:
            self.report(f"Whisper: {message}", percent)

        if (
            self.cfg.whisper_model == "large-v3-turbo"
            and not (self.model_dir / "model.bin").exists()
        ):
            try:
                download_pack(
                    "whisper-large-v3-turbo",
                    self.model_dir,
                    report=progress,
                    cancel_event=self._cancel,
                )
            except DownloadCancelled:
                raise
            except DownloadError as exc:
                logger.warning(
                    "Whisper: пакет с GitHub недоступен (%s); загрузка с Hugging Face", exc
                )

        try:
            download_hf_files(
                repo,
                self.model_dir,
                WHISPER_FILES,
                progress=progress,
                cancel_event=self._cancel,
                required="model.bin",
                ignore_missing=True,
            )
        except DownloadError as exc:
            if (self.model_dir / "model.bin").exists():
                self.warnings.append(f"Не удалось проверить файлы модели: {exc}")
                logger.warning("Whisper: использую локальные файлы (%s)", exc)
            else:
                raise EngineError(
                    f"Не удалось скачать модель Whisper {self.cfg.whisper_model}: {exc}"
                ) from exc
        self.check_cancelled()
        if not (self.model_dir / "model.bin").exists():
            raise EngineError(f"Модель Whisper не найдена в {self.model_dir}")

    def _load(self) -> None:
        # Resolve before importing faster_whisper: on Windows this preloads a
        # complete system cuDNN 9, without which CTranslate2 terminates the
        # process with 0xc0000409 instead of raising a Python exception.
        device, compute, warnings = resolve_whisper_runtime(self.cfg, self.hw)
        self.warnings.extend(warnings)
        for warning in warnings:
            logger.warning("Whisper: %s", warning)

        try:
            from faster_whisper import WhisperModel
        except Exception as exc:
            raise EngineError(
                "Faster-Whisper не установлен. Установите: pip install faster-whisper"
            ) from exc

        self._ensure_files()

        try:
            model_size = sum(item.stat().st_size for item in self.model_dir.glob("*"))
        except OSError:
            model_size = 0
        logger.info(
            "Загрузка Whisper %s из %s (%s), объём %s",
            self.cfg.whisper_model,
            self.model_dir,
            device,
            human_size(model_size),
        )

        attempts = [(device, compute)]
        if device == "cuda":
            if compute != "int8_float16":
                attempts.append(("cuda", "int8_float16"))
            attempts.append(("cpu", "int8"))

        last_error: Optional[Exception] = None
        for attempt_device, attempt_compute in attempts:
            self.check_cancelled()
            self.report(
                f"Whisper: загрузка на {attempt_device} ({attempt_compute})", 50.0
            )
            try:
                self.model = WhisperModel(
                    str(self.model_dir),
                    device=attempt_device,
                    compute_type=attempt_compute,
                    local_files_only=True,
                )
                self.device = attempt_device
                self.compute_type = attempt_compute
                break
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "Whisper: не удалось загрузить на %s/%s: %s",
                    attempt_device,
                    attempt_compute,
                    exc,
                )
                if attempt_device == "cpu" or not _looks_like_cuda_error(exc):
                    raise EngineError(
                        f"Не удалось загрузить Whisper {self.cfg.whisper_model}: {exc}"
                    ) from exc
                self.warnings.append(
                    f"GPU-загрузка Whisper не удалась ({exc}); выполнен переход на CPU "
                    "только для этого запуска"
                )

        if self.model is None:
            raise EngineError(
                f"Не удалось загрузить Whisper {self.cfg.whisper_model}: {last_error}"
            )

        if device == "cuda" and self.device == "cpu":
            logger.warning("Whisper работает на CPU после ошибки CUDA")
        self.report(f"Whisper готов ({self.device}/{self.compute_type})", 100.0)
        logger.info(
            "Faster-Whisper %s готова: %s / %s",
            self.cfg.whisper_model,
            self.device,
            self.compute_type,
        )

    def transcribe(self, path: Path, language: Optional[str] = None) -> TranscriptionResult:
        if self.model is None:
            raise EngineError("Whisper не загружен")
        lang = None if not language or language == "auto" else language
        try:
            raw_segments, info = self.model.transcribe(
                str(path),
                language=lang,
                vad_filter=bool(self.cfg.vad_filter),
                condition_on_previous_text=False,
                beam_size=int(self.cfg.beam_size),
            )
            segments: list[Segment] = []
            texts: list[str] = []
            for segment in raw_segments:
                text = str(getattr(segment, "text", "") or "").strip()
                if not text:
                    continue
                texts.append(text)
                segments.append(
                    Segment(
                        start=float(getattr(segment, "start", 0.0) or 0.0),
                        end=float(getattr(segment, "end", 0.0) or 0.0),
                        text=text,
                    )
                )
        except Exception as exc:
            raise EngineError(f"Whisper не смог распознать аудио: {exc}") from exc

        return TranscriptionResult(
            text=" ".join(texts).strip(),
            segments=segments,
            language=str(getattr(info, "language", lang or "") or lang or ""),
            duration=float(getattr(info, "duration", 0.0) or 0.0),
            engine=self.name,
        )

    def release(self) -> None:
        self.model = None
        gc.collect()

    def runtime_description(self) -> str:
        return f"{self.cfg.whisper_model} · {self.device}/{self.compute_type}"
