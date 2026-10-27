"""Background model loading with reference-counted leases and safe swapping."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from .config import Config, ConfigStore
from .constants import STATUS_ERROR, STATUS_IDLE, STATUS_LOADING, STATUS_READY
from .engines import Engine, create_engine, resolve_engine_name
from .engines.registry import language_warning
from .hardware import detect_hardware

logger = logging.getLogger(__name__)


class EngineNotReady(RuntimeError):
    pass


@dataclass(frozen=True)
class ModelStatus:
    state: str = STATUS_IDLE
    engine: Optional[str] = None
    message: str = ""
    progress: Optional[float] = None
    error: Optional[str] = None
    warnings: tuple[str, ...] = ()
    at: float = field(default_factory=time.time)


@dataclass
class _Loaded:
    engine: Engine
    name: str
    refs: int = 0
    retired: bool = False
    released: bool = False


class EngineLease:
    """Keeps an engine alive while a transcription is running."""

    def __init__(self, manager: "ModelManager", loaded: _Loaded):
        self._manager = manager
        self._loaded: Optional[_Loaded] = loaded
        self._released = False

    @property
    def engine(self) -> Engine:
        if self._loaded is None or self._released:
            raise EngineNotReady("Аренда движка уже освобождена")
        return self._loaded.engine

    @property
    def name(self) -> str:
        return self._loaded.name if self._loaded else ""

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        loaded = self._loaded
        self._loaded = None
        if loaded is not None:
            self._manager._release_lease(loaded)

    def __enter__(self) -> Engine:
        return self.engine

    def __exit__(self, *_exc) -> None:
        self.release()


class ModelManager:
    """Loads one engine at a time in the background.

    A new load is prepared while the current engine keeps serving requests.
    The old engine is released only after its last lease is returned, so a
    transcription can never observe a half-unloaded model.
    """

    def __init__(self, config_store: ConfigStore):
        self._store = config_store
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._current: Optional[_Loaded] = None
        self._loading = False
        self._target_name: Optional[str] = None
        self._pending_name: Optional[str] = None
        self._cancel = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._status = ModelStatus()
        self._listeners: list[Callable[[ModelStatus], None]] = []
        self._closed = False

    # -- status ----------------------------------------------------------
    @property
    def status(self) -> ModelStatus:
        with self._lock:
            return self._status

    def subscribe(self, listener: Callable[[ModelStatus], None]) -> None:
        with self._lock:
            if listener not in self._listeners:
                self._listeners.append(listener)

    def _publish(self, status: ModelStatus) -> None:
        with self._lock:
            self._status = status
            listeners = list(self._listeners)
            self._condition.notify_all()
        for listener in listeners:
            try:
                listener(status)
            except Exception:
                logger.exception("Ошибка подписчика статуса модели")

    # -- queries ---------------------------------------------------------
    @property
    def current_name(self) -> Optional[str]:
        with self._lock:
            return self._current.name if self._current else None

    @property
    def ready(self) -> bool:
        with self._lock:
            return self._current is not None

    @property
    def loading(self) -> bool:
        with self._lock:
            return self._loading

    def current_runtime(self) -> Optional[str]:
        with self._lock:
            engine = self._current.engine if self._current else None
        if engine is None:
            return None
        description = getattr(engine, "runtime_description", None)
        if callable(description):
            try:
                return str(description())
            except Exception:
                return None
        return None

    def wait_ready(self, timeout: float = 120.0) -> bool:
        deadline = time.monotonic() + timeout
        with self._condition:
            while True:
                if self._current is not None:
                    return True
                if not self._loading:
                    return False
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._condition.wait(remaining)

    def acquire(
        self,
        timeout: float = 120.0,
        preferred_name: Optional[str] = None,
    ) -> EngineLease:
        deadline = time.monotonic() + timeout
        with self._condition:
            while True:
                current = self._current
                if current is not None:
                    if preferred_name and current.name != preferred_name and self._loading:
                        # A queued job may wait briefly for the engine it expects.
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            break
                        self._condition.wait(min(remaining, 0.5))
                        continue
                    current.refs += 1
                    return EngineLease(self, current)
                if not self._loading:
                    error = self._status.error
                    raise EngineNotReady(error or "Модель ещё не загружена")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._condition.wait(remaining)
        raise EngineNotReady("Модель не успела загрузиться за отведённое время")

    # -- loading ---------------------------------------------------------
    def request(self, force: bool = False, cfg: Optional[Config] = None) -> None:
        cfg = cfg or self._store.get()
        with self._lock:
            if self._closed:
                return
        hw = detect_hardware()
        name = resolve_engine_name(cfg, hw)
        if name == "none":
            error = (
                "Не установлен ни один движок распознавания. "
                "Установите зависимости из requirements.txt."
            )
            self._publish(ModelStatus(state=STATUS_ERROR, message=error, error=error))
            logger.error(error)
            return

        with self._lock:
            same_ready = bool(
                self._current and self._current.name == name and not force
            )
            if same_ready:
                ready_status = ModelStatus(
                    state=STATUS_READY,
                    engine=name,
                    message="Модель уже загружена",
                )
                self._status = ready_status
            elif self._loading:
                if name != self._target_name:
                    self._pending_name = name
                    logger.info("Текущая загрузка идёт, отложен запрос: %s", name)
                return
            else:
                self._loading = True
                self._target_name = name
                self._pending_name = None
                self._cancel.clear()
                self._thread = threading.Thread(
                    target=self._worker,
                    args=(cfg, name, force),
                    name="model-loader",
                    daemon=True,
                )
                self._thread.start()
        if same_ready:
            self._publish(ready_status)
            return

    def _worker(self, cfg: Config, name: str, force: bool) -> None:
        while True:
            self._load_one(cfg, name, force)
            with self._lock:
                pending = self._pending_name
                self._pending_name = None
                if pending and pending != (self._current.name if self._current else None):
                    name = pending
                    force = False
                    self._target_name = name
                    cfg = self._store.get()
                    logger.info("Запуск отложенной загрузки: %s", name)
                else:
                    self._loading = False
                    self._target_name = None
                    self._condition.notify_all()
                    return

    def _load_one(self, cfg: Config, name: str, force: bool) -> None:
        logger.info("Фоновая загрузка модели: %s", name)
        self._publish(
            ModelStatus(
                state=STATUS_LOADING,
                engine=name,
                message=f"Загрузка {name}…",
                progress=0.0,
            )
        )
        hw = detect_hardware()

        warnings: list[str] = []
        warning = language_warning(name, cfg.language)
        if warning:
            warnings.append(warning)
            logger.warning(warning)

        last_emit = [0.0]

        def report(message: str, percent: Optional[float]) -> None:
            now = time.monotonic()
            if now - last_emit[0] < 0.15 and percent not in (0.0, 100.0):
                return
            last_emit[0] = now
            self._publish(
                ModelStatus(
                    state=STATUS_LOADING,
                    engine=name,
                    message=message,
                    progress=percent,
                    warnings=tuple(warnings),
                )
            )

        try:
            engine = create_engine(
                name,
                cfg,
                hw,
                progress=report,
                cancel_event=self._cancel,
            )
            extra = getattr(engine, "warnings", None)
            if extra:
                for item in extra:
                    if item not in warnings:
                        warnings.append(str(item))

            if self._cancel.is_set():
                engine.release()
                logger.info("Загрузка %s отменена", name)
                self._finish_cancelled()
                return

            old = self._install(engine, name)
            runtime = self.current_runtime()
            self._publish(
                ModelStatus(
                    state=STATUS_READY,
                    engine=name,
                    message=f"{name} готова" + (f" · {runtime}" if runtime else ""),
                    progress=100.0,
                    warnings=tuple(warnings),
                )
            )
            logger.info("Модель %s готова к работе", name)
            if old is not None:
                self._release_if_unused(old)
            for warning in warnings:
                logger.warning("Модель %s: %s", name, warning)
        except EngineNotReady:
            pass
        except Exception as exc:
            if self._cancel.is_set():
                logger.info("Загрузка %s прервана", name)
                self._finish_cancelled()
                return
            logger.exception("Ошибка загрузки модели %s", name)
            with self._lock:
                current = self._current
            message = f"Не удалось загрузить {name}: {exc}"
            if current is not None:
                message += f". Продолжает работать {current.name}"
            self._publish(
                ModelStatus(
                    state=STATUS_READY if current is not None else STATUS_ERROR,
                    engine=current.name if current else name,
                    message=message,
                    error=str(exc),
                    warnings=tuple(warnings),
                )
            )

    def _finish_cancelled(self) -> None:
        with self._lock:
            current = self._current
        if current is not None:
            self._publish(
                ModelStatus(
                    state=STATUS_READY,
                    engine=current.name,
                    message=f"Продолжает работать {current.name}",
                )
            )
        else:
            self._publish(
                ModelStatus(state=STATUS_IDLE, message="Загрузка отменена")
            )

    def _install(self, engine: Engine, name: str) -> Optional[_Loaded]:
        with self._lock:
            old = self._current
            self._current = _Loaded(engine=engine, name=name)
            if old is not None:
                old.retired = True
            self._condition.notify_all()
        if old is not None:
            logger.info("Новая модель %s готова, старая будет освобождена", name)
        return old

    def _release_lease(self, loaded: _Loaded) -> None:
        with self._lock:
            if loaded.refs > 0:
                loaded.refs -= 1
            should_release = loaded.retired and loaded.refs == 0
            if should_release:
                loaded.released = True
        if should_release:
            self._release_engine(loaded)

    @staticmethod
    def _release_engine(loaded: _Loaded) -> None:
        try:
            loaded.engine.release()
            logger.info("Движок %s освобождён", loaded.name)
        except Exception:
            logger.exception("Ошибка освобождения движка %s", loaded.name)

    def _release_if_unused(self, loaded: _Loaded) -> None:
        with self._lock:
            should_release = loaded.retired and loaded.refs == 0 and not loaded.released
            if should_release:
                loaded.released = True
        if should_release:
            self._release_engine(loaded)

    def shutdown(self, timeout: float = 5.0) -> None:
        with self._lock:
            self._closed = True
            self._cancel.set()
            thread = self._thread
            current = self._current
            self._current = None
            if current is not None:
                current.retired = True
                should_release = current.refs == 0
                if should_release:
                    current.released = True
            else:
                should_release = False
            self._condition.notify_all()
        if thread and thread.is_alive():
            thread.join(timeout=timeout)
        if should_release and current is not None:
            self._release_engine(current)
