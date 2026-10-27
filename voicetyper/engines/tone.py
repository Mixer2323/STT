"""T-one streaming CTC engine (Russian, CPU, ONNX + optional KenLM)."""

from __future__ import annotations

import gc
import logging
import threading
from pathlib import Path
from typing import Optional

import numpy as np

from .. import paths
from ..audio import StreamingDownsampler, load_audio_int32_8k
from ..config import Config
from ..constants import TONE_REPO
from ..downloads import DownloadCancelled, DownloadError, download_hf_files, human_size
from ..hardware import HardwareInfo, choose_tone_decoder
from ..model_packs import download_pack
from .base import Engine, EngineError, Segment, StreamSession, TranscriptionResult

logger = logging.getLogger(__name__)

_TONE_CHUNK = 2400  # 300 ms at 8 kHz
_TONE_RATE = 8000


class ToneStreamSession(StreamSession):
    """Adapts 16 kHz int16 microphone blocks to T-one's 8 kHz int32 chunks."""

    def __init__(self, pipeline):
        self._pipeline = pipeline
        self._lock = threading.Lock()
        self._downsampler: Optional[StreamingDownsampler] = None
        self._buffer = np.zeros(0, dtype=np.int16)
        self._state = None
        self._texts: list[str] = []
        self._closed = False

    def _consume(self, chunk: np.ndarray, is_last: bool) -> None:
        phrases, self._state = self._pipeline.forward(
            chunk.astype(np.int32, copy=False), self._state, is_last=is_last
        )
        for phrase in phrases or []:
            text = str(getattr(phrase, "text", "") or "").strip()
            if text:
                self._texts.append(text)

    def feed_pcm16(self, pcm16: np.ndarray, sample_rate: int) -> str:
        if self._closed or pcm16.size == 0:
            return self.text()
        with self._lock:
            if self._downsampler is None:
                self._downsampler = StreamingDownsampler(int(sample_rate), _TONE_RATE)
            converted = self._downsampler.feed(np.asarray(pcm16, dtype=np.int16))
            if converted.size:
                self._buffer = np.concatenate((self._buffer, converted))
            while self._buffer.size >= _TONE_CHUNK:
                chunk = self._buffer[:_TONE_CHUNK]
                self._buffer = self._buffer[_TONE_CHUNK:]
                try:
                    self._consume(chunk, False)
                except Exception:
                    logger.debug("Ошибка стриминга T-one", exc_info=True)
                    self._closed = True
                    break
            return self.text()

    def finish(self) -> str:
        with self._lock:
            if self._closed:
                return self.text()
            self._closed = True
            try:
                if self._buffer.size:
                    padded = np.zeros(_TONE_CHUNK, dtype=np.int16)
                    padded[: min(self._buffer.size, _TONE_CHUNK)] = self._buffer[:_TONE_CHUNK]
                    phrases, self._state = self._pipeline.forward(
                        padded.astype(np.int32), self._state, is_last=False
                    )
                    self._append(phrases)
                    self._buffer = np.zeros(0, dtype=np.int16)
                phrases, self._state = self._pipeline.finalize(self._state)
                self._append(phrases)
            except Exception:
                logger.debug("Ошибка завершения стрима T-one", exc_info=True)
            return self.text()

    def _append(self, phrases) -> None:
        for phrase in phrases or []:
            text = str(getattr(phrase, "text", "") or "").strip()
            if text:
                self._texts.append(text)

    def text(self) -> str:
        return " ".join(self._texts).strip()

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._buffer = np.zeros(0, dtype=np.int16)
            self._state = None


class ToneEngine(Engine):
    name = "tone"
    label = "T-one"
    supports_streaming = True
    languages = ("ru",)

    def __init__(self, cfg: Config, hw: HardwareInfo, **kwargs):
        super().__init__(**kwargs)
        self.cfg = cfg
        self.hw = hw
        self.pipeline = None
        self.decoder_name = "greedy"
        self.warnings: list[str] = []
        self._model_dir = paths.models_dir() / "tone"
        self._load()

    def _progress_wrapper(self, label: str):
        def progress(item):
            percent = item.overall_percent
            suffix = f" {percent:.0f}%" if percent is not None else ""
            self.report(f"T-one: {label}{suffix}", percent)

        return progress

    def _ensure_files(self) -> None:
        self._model_dir.mkdir(parents=True, exist_ok=True)
        self.decoder_name = choose_tone_decoder(self.cfg, self.hw)
        files = ["model.onnx"]
        if self.decoder_name == "beam_search":
            files.append("kenlm.bin")
        if not (self._model_dir / "model.onnx").exists():
            try:
                download_pack(
                    "tone", self._model_dir, report=self.report, cancel_event=self._cancel
                )
            except DownloadCancelled:
                raise
            except DownloadError as exc:
                logger.warning(
                    "T-one: пакет с GitHub недоступен (%s); загрузка с Hugging Face", exc
                )
        try:
            download_hf_files(
                TONE_REPO,
                self._model_dir,
                files,
                progress=self._progress_wrapper("загрузка"),
                cancel_event=self._cancel,
                required="model.onnx",
                ignore_missing=True,
            )
        except DownloadError as exc:
            if not (self._model_dir / "model.onnx").exists():
                raise EngineError(f"Не удалось скачать T-one: {exc}") from exc
            if self.decoder_name == "beam_search":
                self.warnings.append(f"kenlm.bin недоступен ({exc}); используется greedy")
                logger.warning("T-one: kenlm недоступен, переход на greedy: %s", exc)
                self.decoder_name = "greedy"

        if not (self._model_dir / "model.onnx").exists():
            raise EngineError(f"Модель T-one не найдена в {self._model_dir}")
        if self.decoder_name == "beam_search" and not (self._model_dir / "kenlm.bin").exists():
            self.warnings.append("kenlm.bin не найден; используется greedy")
            self.decoder_name = "greedy"

    def _load(self) -> None:
        try:
            from tone import DecoderType, StreamingCTCPipeline
        except Exception as exc:
            raise EngineError(
                "T-one не установлен. Установите: "
                "pip install git+https://github.com/voicekit-team/T-one.git"
            ) from exc

        self._ensure_files()
        self.check_cancelled()
        decoder_type = (
            DecoderType.BEAM_SEARCH
            if self.decoder_name == "beam_search"
            else DecoderType.GREEDY
        )
        self.report("T-one: загрузка в память", 80.0)
        try:
            self.pipeline = StreamingCTCPipeline.from_local(
                str(self._model_dir), decoder_type=decoder_type
            )
        except Exception as exc:
            raise EngineError(f"Не удалось загрузить T-one: {exc}") from exc
        self.report("T-one готова", 100.0)
        logger.info(
            "T-one подключён (декодер: %s, файл: %s)",
            self.decoder_name,
            human_size((self._model_dir / "model.onnx").stat().st_size),
        )

    def start_stream(self) -> Optional[StreamSession]:
        if self.pipeline is None:
            return None
        return ToneStreamSession(self.pipeline)

    def transcribe(self, path: Path, language: Optional[str] = None) -> TranscriptionResult:
        if self.pipeline is None:
            raise EngineError("T-one не загружен")
        try:
            audio = load_audio_int32_8k(path)
            phrases = self.pipeline.forward_offline(audio)
        except Exception as exc:
            raise EngineError(f"T-one не смог распознать аудио: {exc}") from exc

        segments: list[Segment] = []
        texts: list[str] = []
        for phrase in phrases or []:
            text = str(getattr(phrase, "text", "") or "").strip()
            if not text:
                continue
            texts.append(text)
            segments.append(
                Segment(
                    start=float(getattr(phrase, "start_time", 0.0) or 0.0),
                    end=float(getattr(phrase, "end_time", 0.0) or 0.0),
                    text=text,
                )
            )
        return TranscriptionResult(
            text=" ".join(texts).strip(),
            segments=segments,
            language="ru",
            duration=audio.size / _TONE_RATE,
            engine=self.name,
        )

    def release(self) -> None:
        pipeline = self.pipeline
        self.pipeline = None
        if pipeline is None:
            gc.collect()
            return
        try:
            if getattr(pipeline, "model", None) is not None:
                pipeline.model._ort_sess = None
        except Exception:
            pass
        try:
            decoder = getattr(getattr(pipeline, "decoder", None), "_decoder", None)
            if decoder is not None and hasattr(decoder, "cleanup"):
                decoder.cleanup()
        except Exception:
            pass
        try:
            from pyctcdecode.decoder import BeamSearchDecoderCTC

            BeamSearchDecoderCTC.clear_class_models()
        except Exception:
            pass
        try:
            pipeline.model = None
            pipeline.logprob_splitter = None
            pipeline.decoder = None
        except Exception:
            pass
        gc.collect()

    def runtime_description(self) -> str:
        return f"T-one · {self.decoder_name}"
