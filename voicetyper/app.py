"""Application wiring: lifecycle, recording state machine and user actions."""

from __future__ import annotations

import argparse
import dataclasses
import logging
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Optional

from . import paths, single_instance
from .audio import AudioError, RecordingSession
from .autostart import is_enabled as autostart_is_enabled
from .autostart import is_supported as autostart_supported
from .autostart import set_enabled as set_autostart_enabled
from .config import Config, ConfigStore
from .constants import (
    APP_TITLE,
    APP_VERSION,
    ENGINE_LABELS,
    ENGINE_ORDER,
    MULTILINGUAL_ENGINES,
    STATUS_ERROR,
    STATUS_IDLE,
    STATUS_READY,
    STATUS_RECORDING,
    STATUS_TRANSCRIBING,
)
from .diagnostics import collect_diagnostics, open_folder, write_diagnostics
from .engines import resolve_engine_name
from .hardware import detect_hardware
from .history import HistoryStore
from .hotkeys import HotkeyError, HotkeyManager, normalize_hotkey
from .insertion import foreground_window
from .logging_setup import setup_logging
from .models import ModelManager, ModelStatus
from .notifications import Notifier
from .pipeline import PipelineResult, TranscriptionPipeline
from .postprocess import preview_text
from .settings_ui import SettingsManager
from .tray import TrayController

logger = logging.getLogger(__name__)


class _PreviewSession:
    """Feeds microphone blocks to a streaming engine without blocking PortAudio."""

    def __init__(self, app: "VoiceTyperApp", models: ModelManager, sample_rate: int):
        self._app = app
        self._models = models
        self._sample_rate = int(sample_rate)
        self._queue: queue.Queue = queue.Queue(maxsize=128)
        self._lock = threading.Lock()
        self._release_lock = threading.Lock()
        self._stream = None
        self._lease = None
        self._thread: Optional[threading.Thread] = None
        self._closed = False
        self._release_pending = False

        lease = None
        try:
            lease = models.acquire(timeout=0.2)
            stream = lease.engine.start_stream()
            if stream is None:
                lease.release()
                return
            self._lease = lease
            self._stream = stream
        except Exception as exc:
            logger.debug("Потоковый предпросмотр недоступен: %s", exc)
            if lease is not None:
                lease.release()

    @property
    def active(self) -> bool:
        return self._stream is not None

    def start(self) -> None:
        if not self.active:
            return
        self._thread = threading.Thread(
            target=self._run, name="stream-preview", daemon=True
        )
        self._thread.start()

    def feed(self, chunk, sample_rate: int) -> None:
        if self._closed or self._stream is None:
            return
        item = (chunk, int(sample_rate))
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            # Keep the preview live: drop the oldest block instead of letting
            # the backlog grow without bound on a slow machine.
            try:
                self._queue.get_nowait()
                self._queue.task_done()
                self._queue.put_nowait(item)
            except (queue.Empty, queue.Full):
                logger.debug("Очередь предпросмотра переполнена")

    def _run(self) -> None:
        last = ""
        try:
            while True:
                item = self._queue.get()
                try:
                    if item is None:
                        return
                    chunk, sample_rate = item
                    try:
                        text = self._stream.feed_pcm16(chunk, sample_rate)
                    except Exception:
                        logger.debug("Ошибка потокового распознавания", exc_info=True)
                        return
                    if text and text != last:
                        last = text
                        self._app._on_preview_text(text)
                finally:
                    self._queue.task_done()
        finally:
            if self._release_pending:
                self._release()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        while True:
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except queue.Empty:
                break
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=4.0)
        if thread is not None and thread.is_alive():
            self._release_pending = True
        else:
            self._release()

    def _release(self) -> None:
        with self._release_lock:
            stream = self._stream
            self._stream = None
            lease = self._lease
            self._lease = None
        if stream is not None:
            try:
                stream.close()
            except Exception:
                logger.debug("Не удалось закрыть потоковый сеанс", exc_info=True)
        if lease is not None:
            lease.release()


class VoiceTyperApp:
    def __init__(
        self,
        config_store: ConfigStore,
        history: HistoryStore,
        instance: Optional[single_instance.InstanceLock],
        *,
        first_run: bool = False,
        open_settings: bool = False,
    ):
        self.config_store = config_store
        self.history = history
        self._instance = instance
        self._first_run = first_run
        self._open_settings_on_start = open_settings

        self.models = ModelManager(config_store)
        self.notifier = Notifier(config_store.get().notifications)
        self.tray = TrayController(self, config_store)
        self.settings = SettingsManager(self, config_store)
        self.hotkeys = HotkeyManager(
            config_store,
            on_press=self._on_hotkey_press,
            on_release=self._on_hotkey_release,
            on_error=self._on_hotkey_error,
        )

        self._recording_lock = threading.RLock()
        self._recording: Optional[RecordingSession] = None
        self._target = None
        self._preview: Optional[_PreviewSession] = None

        self._pipeline: Optional[TranscriptionPipeline] = None
        self._transcribing = False
        self._model_status = ModelStatus(state=STATUS_IDLE, message="Ожидание")
        self._model_notified = False
        self._last_model_error = ""
        self._error_generation = 0

        self._shutdown_event = threading.Event()
        self._shutdown_lock = threading.Lock()
        self._shutdown_started = False
        self._cleanup_lock = threading.Lock()
        self._cleaned = False
        self._started = False

        self.models.subscribe(self._on_model_status)

    # -- lifecycle -------------------------------------------------------
    @property
    def is_recording(self) -> bool:
        with self._recording_lock:
            session = self._recording
            return bool(session is not None and session.is_recording)

    def run(self) -> None:
        if self._instance is not None:
            self._instance.start_activation_listener(self.open_settings)
        try:
            if self.tray.available:
                self.tray.run(setup=self._on_tray_ready)
            else:
                logger.warning("pystray недоступен — работа без значка в трее")
                self._on_tray_ready(None)
                while (
                    not self._shutdown_event.is_set()
                    and self.settings.is_open
                ):
                    time.sleep(0.25)
                if not self._shutdown_event.is_set():
                    self.shutdown()
        except Exception:
            logger.exception("Критическая ошибка приложения")
            self.shutdown()
        finally:
            self._cleanup()

    def _on_tray_ready(self, icon) -> None:
        if self._started:
            return
        self._started = True
        if icon is not None:
            self.notifier.set_icon(icon)

        cfg = self.config_store.get()
        self._pipeline = TranscriptionPipeline(
            self.models,
            self.config_store,
            self.history,
            on_state=self._on_pipeline_state,
            on_result=self._on_pipeline_result,
        )
        self.models.request(cfg=cfg)

        if self.hotkeys.available:
            try:
                self.hotkeys.start()
            except HotkeyError as exc:
                logger.error("Горячая клавиша не запущена: %s", exc)
                self.notifier.notify(
                    "Горячая клавиша",
                    f"Не удалось зарегистрировать {cfg.hotkey}: {exc}",
                    force=True,
                )
            else:
                bound = self.hotkeys.bound_hotkey
                if bound and bound != cfg.hotkey:
                    try:
                        self.config_store.replace(dataclasses.replace(cfg, hotkey=bound))
                        logger.info(
                            "Горячая клавиша нормализована: %s → %s", cfg.hotkey, bound
                        )
                    except Exception:
                        logger.debug("Не удалось сохранить горячую клавишу", exc_info=True)
        else:
            logger.warning("Библиотека keyboard недоступна")
            self.notifier.notify(
                "Горячая клавиша",
                "Библиотека keyboard не установлена — используйте меню трея.",
                force=True,
            )

        self._sync_autostart(cfg)
        self.tray.set_state(
            self._model_status.state,
            self._model_status.message,
            engine=self._model_status.engine,
            progress=self._model_status.progress,
        )
        self.notifier.notify(
            "Voice Typer запущен",
            "Приложение работает в фоне. Значок — в области уведомлений; "
            "если его не видно, раскройте скрытые значки (^).",
            key="startup",
            cooldown=3600.0,
        )

        if self._open_settings_on_start or self._first_run:
            self.settings.open("main")

    def _cleanup(self) -> None:
        with self._cleanup_lock:
            if self._cleaned:
                return
            self._cleaned = True
        self._abort_recording()
        try:
            self.hotkeys.close()
        except Exception:
            logger.debug("Ошибка остановки горячей клавиши", exc_info=True)
        try:
            self.settings.close()
        except Exception:
            logger.debug("Ошибка остановки окна настроек", exc_info=True)
        if self._pipeline is not None:
            try:
                self._pipeline.stop(timeout=5.0)
            except Exception:
                logger.debug("Ошибка остановки очереди", exc_info=True)
        try:
            self.tray.dispose()
        except Exception:
            logger.debug("Ошибка остановки трея", exc_info=True)
        try:
            self.models.shutdown()
        except Exception:
            logger.debug("Ошибка остановки моделей", exc_info=True)
        if self._instance is not None:
            try:
                self._instance.release()
            except Exception:
                logger.debug("Ошибка освобождения блокировки экземпляра", exc_info=True)
        logger.info("Приложение остановлено")

    @property
    def is_shutting_down(self) -> bool:
        return self._shutdown_started

    def shutdown(self) -> None:
        with self._shutdown_lock:
            if self._shutdown_started:
                return
            self._shutdown_started = True
        logger.info("Завершение работы…")
        self._shutdown_event.set()
        self._abort_recording()
        try:
            self.hotkeys.close()
        except Exception:
            logger.debug("Ошибка остановки горячей клавиши", exc_info=True)
        try:
            self.settings.close()
        except Exception:
            logger.debug("Ошибка остановки окна настроек", exc_info=True)
        if self._pipeline is not None:
            try:
                self._pipeline.stop(timeout=3.0)
            except Exception:
                logger.debug("Ошибка остановки очереди", exc_info=True)
        self.tray.dispose()

    # -- recordings ------------------------------------------------------
    def toggle_recording(self) -> None:
        if self.is_recording:
            self.stop_recording()
        else:
            self.start_recording()

    def start_recording(self) -> None:
        if self._shutdown_event.is_set():
            return
        with self._recording_lock:
            if self._recording is not None:
                return
            cfg = self.config_store.get()
            target = foreground_window()
            session = RecordingSession(
                cfg,
                on_auto_stop=self._on_auto_stop,
                on_chunk=self._on_audio_chunk,
            )
            try:
                session.start()
            except AudioError as exc:
                logger.error("Не удалось начать запись: %s", exc)
                self._flash_error(str(exc), "Запись")
                return
            except Exception as exc:
                logger.exception("Неожиданная ошибка начала записи")
                self._flash_error(f"Не удалось начать запись: {exc}", "Запись")
                return
            self._recording = session
            self._target = target

        self._start_preview(session)
        self.tray.set_state(STATUS_RECORDING, "")

    def stop_recording(self) -> None:
        with self._recording_lock:
            session = self._recording
            target = self._target
            self._recording = None
            self._target = None
        if session is None:
            return

        result = session.stop()
        self._stop_preview()

        if result.error:
            logger.info("Запись не отправлена: %s", result.error)
            if result.silent:
                self.notifier.notify("Запись", result.error, key="silent", cooldown=3.0)
            else:
                self._flash_error(result.error, "Запись")
            return
        if result.path is None:
            return
        if self._pipeline is None:
            try:
                result.path.unlink(missing_ok=True)
            except OSError:
                pass
            self._restore_tray_state()
            return

        self._transcribing = True
        job_id = self._pipeline.submit(
            result.path,
            result.duration,
            target=target,
        )
        if job_id is None:
            self._transcribing = False
            self._restore_tray_state()
            return
        self.tray.set_state(STATUS_TRANSCRIBING, "Распознавание…")

    def _on_auto_stop(self, session: RecordingSession) -> None:
        with self._recording_lock:
            if self._recording is not session:
                return
        self.stop_recording()

    def _abort_recording(self) -> None:
        with self._recording_lock:
            session = self._recording
            self._recording = None
            self._target = None
        if session is not None:
            try:
                session.abort()
            except Exception:
                logger.debug("Ошибка прерывания записи", exc_info=True)
        self._stop_preview()

    def _on_audio_chunk(self, chunk, sample_rate: int) -> None:
        preview = self._preview
        if preview is not None:
            preview.feed(chunk, sample_rate)

    def _start_preview(self, session: RecordingSession) -> None:
        cfg = self.config_store.get()
        if not cfg.streaming_preview:
            return
        preview = _PreviewSession(self, self.models, session.capture_rate)
        if not preview.active:
            return
        self._preview = preview
        preview.start()

    def _stop_preview(self) -> None:
        preview = self._preview
        self._preview = None
        if preview is not None:
            preview.close()

    def _on_preview_text(self, text: str) -> None:
        if not self.is_recording:
            return
        tail = text[-64:].strip()
        if tail:
            self.tray.set_state(STATUS_RECORDING, tail)

    def _on_hotkey_press(self) -> None:
        cfg = self.config_store.get()
        if cfg.hotkey_mode == "toggle":
            self.toggle_recording()
        elif not self.is_recording:
            self.start_recording()

    def _on_hotkey_release(self) -> None:
        cfg = self.config_store.get()
        if cfg.hotkey_mode == "hold" and self.is_recording:
            self.stop_recording()

    def _on_hotkey_error(self, message: str) -> None:
        self._flash_error(message, "Горячая клавиша")

    # -- model / pipeline status ----------------------------------------
    def _on_model_status(self, status: ModelStatus) -> None:
        self._model_status = status
        logger.debug(
            "Статус модели: %s %s %s", status.state, status.engine, status.message
        )

        if status.error:
            if status.error != self._last_model_error:
                self.notifier.notify("Модель", status.message, key="model-error", cooldown=8.0)
            self._last_model_error = status.error
        else:
            self._last_model_error = ""

        if self.is_recording or self._transcribing:
            return

        self.tray.set_state(
            status.state,
            status.message,
            engine=status.engine,
            progress=status.progress,
        )
        if (
            status.state == STATUS_READY
            and not status.error
            and not self._model_notified
        ):
            self._model_notified = True
            self.notifier.notify("Модель готова", status.message)

    def _restore_tray_state(self) -> None:
        if self.is_recording:
            return
        status = self._model_status
        self.tray.set_state(
            status.state,
            status.message,
            engine=status.engine,
            progress=status.progress,
        )

    def _on_pipeline_state(self, state: str, detail: str) -> None:
        if state == STATUS_TRANSCRIBING:
            self._transcribing = True
            if not self.is_recording:
                self.tray.set_state(STATUS_TRANSCRIBING, detail or "Распознавание…")
            return

        self._transcribing = False
        if self.is_recording:
            return
        if self._pipeline is not None and self._pipeline.pending:
            return
        self._restore_tray_state()

    def _on_pipeline_result(self, result: PipelineResult) -> None:
        if result.error:
            self.notifier.notify(
                "Распознавание",
                result.error,
                key="pipeline-error",
                cooldown=3.0,
            )
            return
        if result.excluded_by:
            return
        insertion = result.insertion
        if insertion is None:
            return
        if insertion.error:
            message = insertion.error
            if result.text:
                message += "\n" + preview_text(result.text, 180)
            self.notifier.notify("Текст не вставлен", message, force=True)
            return
        if insertion.method == "clipboard" and not insertion.inserted:
            self.notifier.notify(
                "Скопировано",
                "Текст распознан и скопирован в буфер обмена.",
                key="clipboard",
                cooldown=2.0,
            )

    # -- user actions (tray / hotkeys) ----------------------------------
    def open_settings(self, tab: Optional[str] = None) -> None:
        self.settings.open(tab)

    def open_history(self) -> None:
        self.settings.open("history")

    def open_data_dir(self) -> None:
        self._open_path(paths.data_dir(), "Папка данных")

    def open_logs(self) -> None:
        self._open_path(paths.logs_dir(), "Логи")

    def open_models_dir(self) -> None:
        self._open_path(paths.models_dir(), "Модели")

    def _open_path(self, path: Path, title: str) -> None:
        try:
            open_folder(path)
        except Exception as exc:
            logger.exception("Не удалось открыть %s", path)
            self.notifier.notify(title, f"Не удалось открыть папку: {exc}", force=True)

    def run_diagnostics(self) -> None:
        try:
            text = self.collect_diagnostics_text()
            destination = write_diagnostics(text)
            open_folder(destination.parent)
            self.notifier.notify(
                "Диагностика",
                f"Сохранено: {destination.name}",
                force=True,
            )
        except Exception as exc:
            logger.exception("Не удалось собрать диагностику")
            self.notifier.notify("Диагностика", f"Ошибка: {exc}", force=True)

    def collect_diagnostics_text(self) -> str:
        return collect_diagnostics(self.config_store, self.models, self.history)

    def reload_model(self) -> None:
        self.models.request(force=True)

    def set_engine(self, name: str) -> None:
        if name not in ENGINE_ORDER:
            return
        cfg = self._save(model_type=name)
        logger.info("Выбран движок: %s", ENGINE_LABELS.get(name, name))
        self.models.request(force=True, cfg=cfg)

    def set_language(self, code: str) -> None:
        if self.config_store.get().model_type not in MULTILINGUAL_ENGINES:
            return
        cfg = self._save(language=code)
        logger.info("Выбран язык: %s", code)
        self.models.request(force=True, cfg=cfg)

    def set_whisper_model(self, name: str) -> None:
        cfg = self._save(whisper_model=name)
        if self._resolved_engine(cfg) == "faster_whisper":
            self.models.request(force=True, cfg=cfg)

    def set_tone_decoder(self, name: str) -> None:
        cfg = self._save(tone_decoder=name)
        if self._resolved_engine(cfg) == "tone":
            self.models.request(force=True, cfg=cfg)

    def set_hotkey_mode(self, mode: str) -> None:
        if mode not in ("hold", "toggle"):
            return
        self._save(hotkey_mode=mode)

    def set_autostart(self, enabled: bool) -> None:
        if not autostart_supported():
            self.notifier.notify(
                "Автозапуск",
                "Автозапуск поддерживается только в Windows.",
                force=True,
            )
            return
        ok, message = set_autostart_enabled(bool(enabled))
        if not ok:
            self.notifier.notify("Автозапуск", message, force=True)
            return
        self._save(autostart=bool(enabled))
        self.notifier.notify("Автозапуск", message, key="autostart", cooldown=2.0)

    def _save(self, **changes) -> Config:
        cfg = self.config_store.update(**changes)
        self.tray.refresh_menu()
        return cfg

    def _resolved_engine(self, cfg: Config) -> str:
        return resolve_engine_name(cfg, detect_hardware())

    def _sync_autostart(self, cfg: Config) -> None:
        if not autostart_supported():
            return
        try:
            actual = autostart_is_enabled()
            if actual != bool(cfg.autostart):
                ok, message = set_autostart_enabled(bool(cfg.autostart))
                if not ok:
                    logger.warning("Не удалось синхронизировать автозапуск: %s", message)
        except Exception:
            logger.debug("Ошибка синхронизации автозапуска", exc_info=True)

    def apply_config(self, cfg: Config) -> tuple[bool, str]:
        old = self.config_store.get()
        normalized = cfg.hotkey.strip() if cfg.hotkey else ""
        if not normalized:
            return False, "Горячая клавиша не задана"
        if self.hotkeys.available:
            try:
                normalized, _keys = normalize_hotkey(normalized)
            except HotkeyError as exc:
                return False, str(exc)

        autostart_changed = bool(cfg.autostart) != bool(old.autostart)
        if autostart_changed:
            ok, message = set_autostart_enabled(bool(cfg.autostart))
            if not ok:
                return False, message

        hotkey_changed = normalized != old.hotkey
        if hotkey_changed and self.hotkeys.available and self._started:
            try:
                self.hotkeys.rebind(normalized)
            except HotkeyError as exc:
                if autostart_changed:
                    set_autostart_enabled(bool(old.autostart))
                return False, str(exc)

        applied = self.config_store.replace(dataclasses.replace(cfg, hotkey=normalized))
        self.notifier.enabled = applied.notifications
        self.history.update_settings(
            enabled=applied.history_enabled,
            limit=applied.history_limit,
        )
        self.tray.refresh_menu()
        if not applied.streaming_preview:
            self._stop_preview()

        hw = detect_hardware()
        old_engine = resolve_engine_name(old, hw)
        new_engine = resolve_engine_name(applied, hw)
        reload = (
            old_engine != new_engine
            or old.model_type != applied.model_type
            or old.language != applied.language
        )
        if new_engine == "faster_whisper":
            reload = reload or any(
                getattr(old, field) != getattr(applied, field)
                for field in ("whisper_model", "device", "compute_type", "beam_size", "vad_filter")
            )
        if new_engine == "tone":
            reload = reload or old.tone_decoder != applied.tone_decoder
        if reload:
            self.models.request(force=True, cfg=applied)

        notes = ["Настройки применены"]
        if hotkey_changed and not self.hotkeys.available:
            notes.append("горячая клавиша будет доступна после установки keyboard")
        if applied.single_instance != old.single_instance:
            notes.append("режим единственного экземпляра изменится после перезапуска")
        return True, ". ".join(notes)

    # -- errors ----------------------------------------------------------
    def _flash_error(self, message: str, title: str) -> None:
        self._error_generation += 1
        generation = self._error_generation
        self.tray.set_state(STATUS_ERROR, message)
        self.notifier.notify(title, message, force=True)

        def restore() -> None:
            time.sleep(2.5)
            if generation != self._error_generation:
                return
            if self._shutdown_event.is_set() or self.is_recording or self._transcribing:
                return
            status = self._model_status
            self.tray.set_state(
                status.state,
                status.message,
                engine=status.engine,
                progress=status.progress,
            )

        threading.Thread(target=restore, name="tray-restore", daemon=True).start()


def _parse_args(argv: Optional[list[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="VoiceTyper",
        description="Локальное распознавание речи с вставкой текста в активное окно.",
    )
    parser.add_argument("--settings", action="store_true", help="открыть настройки при запуске")
    parser.add_argument(
        "--minimized",
        action="store_true",
        help="не открывать окно настроек при первом запуске",
    )
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="напечатать диагностику и сохранить её в файл, не запуская приложение",
    )
    parser.add_argument("--verbose", action="store_true", help="подробное логирование")
    parser.add_argument("--version", action="store_true", help="показать версию")
    return parser.parse_args(argv)


def _fatal(message: str) -> None:
    logger.critical("%s", message)
    if sys.platform == "win32":
        try:
            import ctypes

            ctypes.windll.user32.MessageBoxW(None, message, APP_TITLE, 0x10)
        except Exception:
            pass


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    if args.version:
        print(f"{APP_TITLE} {APP_VERSION}")
        return 0

    paths.ensure_runtime_dirs()
    setup_logging(verbose=args.verbose)
    logger.info("%s %s запускается", APP_TITLE, APP_VERSION)

    first_run = not paths.config_path().exists() and not paths.legacy_config_path().exists()
    store = ConfigStore(paths.config_path(), legacy_path=paths.legacy_config_path())
    for warning in store.load():
        logger.warning("Конфигурация: %s", warning)
    cfg = store.get()

    if args.diagnostics:
        history = HistoryStore(
            paths.history_path(),
            limit=cfg.history_limit,
            enabled=cfg.history_enabled,
        )
        models = ModelManager(store)
        try:
            text = collect_diagnostics(store, models, history)
            print(text)
            destination = write_diagnostics(text)
            print(f"\nСохранено: {destination}")
        except Exception as exc:
            logger.exception("Ошибка диагностики")
            print(f"Не удалось собрать диагностику: {exc}", file=sys.stderr)
            return 1
        return 0

    instance = None
    if cfg.single_instance:
        instance = single_instance.acquire()
        if instance.already_running:
            if instance.notify_existing():
                logger.info("Приложение уже запущено — активировано окно настроек")
            else:
                logger.info("Приложение уже запущено")
            instance.release()
            return 0

    history = HistoryStore(
        paths.history_path(),
        limit=cfg.history_limit,
        enabled=cfg.history_enabled,
    )
    app = VoiceTyperApp(
        store,
        history,
        instance,
        first_run=first_run and not args.minimized,
        open_settings=args.settings,
    )
    try:
        app.run()
    except KeyboardInterrupt:
        app.shutdown()
    except Exception as exc:
        logger.exception("Не удалось запустить приложение")
        _fatal(f"Не удалось запустить {APP_TITLE}:\n{exc}")
        app.shutdown()
        return 1
    finally:
        if instance is not None:
            instance.release()
    return 0
