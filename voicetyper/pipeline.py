"""Bounded transcription queue: engine lease -> cleanup -> insertion -> history."""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from .config import ConfigStore
from .constants import STATUS_IDLE, STATUS_TRANSCRIBING
from .history import HistoryStore
from .insertion import InsertionResult, WindowTarget, insert_text
from .logging_setup import log_transcript
from .models import EngineNotReady, ModelManager
from .postprocess import clean_transcript, matches_exclusion

logger = logging.getLogger(__name__)


@dataclass
class Job:
    id: int
    path: Path
    duration: float
    target: Optional[WindowTarget]
    preferred_engine: Optional[str]
    created_at: float = field(default_factory=time.monotonic)


@dataclass
class PipelineResult:
    job_id: int
    text: str = ""
    engine: str = ""
    language: str = ""
    duration: float = 0.0
    insertion: Optional[InsertionResult] = None
    history_id: Optional[int] = None
    excluded_by: Optional[str] = None
    error: Optional[str] = None
    elapsed: float = 0.0
    from_cache: bool = False

    @property
    def ok(self) -> bool:
        return not self.error and not self.excluded_by and bool(self.text)


StateCallback = Callable[[str, str], None]
ResultCallback = Callable[[PipelineResult], None]


class TranscriptionPipeline:
    def __init__(
        self,
        models: ModelManager,
        config_store: ConfigStore,
        history: HistoryStore,
        *,
        on_state: Optional[StateCallback] = None,
        on_result: Optional[ResultCallback] = None,
        max_queue: int = 8,
        acquire_timeout: float = 180.0,
    ):
        self._models = models
        self._store = config_store
        self._history = history
        self._on_state = on_state
        self._on_result = on_result
        self._queue: queue.Queue[Optional[Job]] = queue.Queue(maxsize=max_queue)
        self._worker = threading.Thread(
            target=self._run, name="transcription-worker", daemon=True
        )
        self._stopped = threading.Event()
        self._next_id = 1
        self._id_lock = threading.Lock()
        self._acquire_timeout = acquire_timeout
        self._worker.start()

    @property
    def pending(self) -> int:
        return self._queue.qsize()

    def submit(
        self,
        path: Path,
        duration: float,
        *,
        target: Optional[WindowTarget] = None,
        preferred_engine: Optional[str] = None,
    ) -> Optional[int]:
        path = Path(path)
        if self._stopped.is_set():
            self._safe_unlink(path)
            return None
        if not path.exists():
            logger.error("Файл для распознавания не найден: %s", path)
            self._emit_state("error", "Файл записи не найден")
            return None
        with self._id_lock:
            job_id = self._next_id
            self._next_id += 1
        job = Job(
            id=job_id,
            path=path,
            duration=duration,
            target=target,
            preferred_engine=preferred_engine,
        )
        try:
            self._queue.put_nowait(job)
        except queue.Full:
            self._safe_unlink(path)
            message = "Очередь распознавания переполнена — запись пропущена"
            logger.warning(message)
            self._emit_state("error", message)
            self._emit_result(PipelineResult(job_id=job_id, error=message))
            return None
        logger.info(
            "Запись #%d в очереди (%.2f с, %s)", job_id, duration, path.name
        )
        return job_id

    def _run(self) -> None:
        while True:
            job = self._queue.get()
            try:
                if job is None:
                    return
                if self._stopped.is_set():
                    self._safe_unlink(job.path)
                    continue
                self._process(job)
            finally:
                self._queue.task_done()

    def _process(self, job: Job) -> None:
        started = time.monotonic()
        self._emit_state(STATUS_TRANSCRIBING, "Распознавание…")
        cfg = self._store.get()
        lease = None
        try:
            try:
                lease = self._models.acquire(
                    timeout=self._acquire_timeout,
                    preferred_name=job.preferred_engine,
                )
            except EngineNotReady as exc:
                raise RuntimeError(str(exc)) from exc

            engine = lease.engine
            engine_name = lease.name
            language = None if cfg.language == "auto" else cfg.language
            outcome = engine.transcribe(job.path, language=language)
            text = clean_transcript(outcome.text, cfg)
            elapsed = time.monotonic() - started

            if not text:
                self._emit_result(
                    PipelineResult(
                        job_id=job.id,
                        engine=engine_name,
                        language=outcome.language or "",
                        duration=job.duration,
                        elapsed=elapsed,
                        error="Речь не распознана",
                    )
                )
                return

            log_transcript(
                text,
                store_text=cfg.log_transcripts,
                engine=engine_name,
            )
            excluded = matches_exclusion(text, cfg.exclude_phrases)
            if excluded:
                logger.info("Фраза «%s» — вставка отменена", excluded)
                self._emit_result(
                    PipelineResult(
                        job_id=job.id,
                        text=text,
                        engine=engine_name,
                        language=outcome.language or "",
                        duration=job.duration,
                        excluded_by=excluded,
                        elapsed=elapsed,
                    )
                )
                return

            try:
                insertion = insert_text(text, cfg, target=job.target)
            except Exception as exc:
                logger.exception("Неожиданная ошибка вставки")
                insertion = InsertionResult(
                    method=cfg.insert_mode,
                    error=f"Ошибка вставки: {exc}",
                )
            if insertion.error:
                logger.error("Вставка не удалась: %s", insertion.error)
            history_id = self._history.add(
                text,
                engine=engine_name,
                language=outcome.language or cfg.language,
                duration=job.duration,
                app=job.target.title if job.target else "",
                status=("insert_failed" if insertion.error else insertion.method),
            )
            self._emit_result(
                PipelineResult(
                    job_id=job.id,
                    text=text,
                    engine=engine_name,
                    language=outcome.language or "",
                    duration=job.duration,
                    insertion=insertion,
                    history_id=history_id,
                    elapsed=time.monotonic() - started,
                )
            )
        except Exception as exc:
            logger.exception("Ошибка распознавания записи #%d", job.id)
            self._emit_result(
                PipelineResult(
                    job_id=job.id,
                    duration=job.duration,
                    error=str(exc),
                    elapsed=time.monotonic() - started,
                )
            )
        finally:
            if lease is not None:
                lease.release()
            self._safe_unlink(job.path)
            self._emit_state(STATUS_IDLE, "")

    def _emit_state(self, state: str, detail: str) -> None:
        if self._on_state is None:
            return
        try:
            self._on_state(state, detail)
        except Exception:
            logger.exception("Ошибка обработчика состояния")

    def _emit_result(self, result: PipelineResult) -> None:
        if result.error:
            logger.warning(
                "Запись #%d: %s (%.2f с)", result.job_id, result.error, result.elapsed
            )
        elif result.excluded_by:
            logger.info("Запись #%d исключена фразой «%s»", result.job_id, result.excluded_by)
        else:
            logger.info(
                "Запись #%d готова: %d симв., %s, %.2f с",
                result.job_id,
                len(result.text),
                result.engine,
                result.elapsed,
            )
        if self._on_result is None:
            return
        try:
            self._on_result(result)
        except Exception:
            logger.exception("Ошибка обработчика результата")

    @staticmethod
    def _safe_unlink(path: Path) -> None:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            logger.debug("Не удалось удалить %s", path, exc_info=True)

    def stop(self, timeout: float = 8.0) -> None:
        if self._stopped.is_set():
            return
        self._stopped.set()
        while True:
            try:
                self._queue.put_nowait(None)
                break
            except queue.Full:
                try:
                    dropped = self._queue.get_nowait()
                except queue.Empty:
                    continue
                if dropped is not None:
                    self._safe_unlink(dropped.path)
                self._queue.task_done()
        if self._worker.is_alive():
            self._worker.join(timeout=timeout)
        while True:
            try:
                job = self._queue.get_nowait()
            except queue.Empty:
                break
            if job is not None:
                self._safe_unlink(job.path)
            self._queue.task_done()
        logger.info("Очередь распознавания остановлена")
