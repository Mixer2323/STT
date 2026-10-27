"""Single-instance guard and activation signal.

The second launch signals a named auto-reset event, so ``VoiceTyper.exe``
started again from the Start menu opens the settings window of the running
instance instead of silently doing nothing.
"""

from __future__ import annotations

import logging
import sys
import threading
from typing import Callable, Optional

from .constants import ACTIVATION_EVENT_NAME, MUTEX_NAME

logger = logging.getLogger(__name__)

_ERROR_ALREADY_EXISTS = 183
_WAIT_OBJECT_0 = 0
_WAIT_TIMEOUT = 258
_EVENT_MODIFY_STATE = 0x0002


class InstanceLock:
    def __init__(self, already_running: bool):
        self.already_running = already_running
        self._mutex = None
        self._event = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._kernel32 = None
        self._posix_handle = None

    # -- Windows ---------------------------------------------------------
    def start_activation_listener(self, callback: Callable[[], None]) -> None:
        if self.already_running or self._kernel32 is None or self._event is None:
            return
        if self._thread and self._thread.is_alive():
            return

        def listen() -> None:
            assert self._kernel32 is not None
            while not self._stop.is_set():
                result = self._kernel32.WaitForSingleObject(self._event, 300)
                if result == _WAIT_OBJECT_0:
                    try:
                        callback()
                    except Exception:
                        logger.exception("Ошибка обработчика повторного запуска")
                elif result not in (_WAIT_TIMEOUT,):
                    logger.debug("WaitForSingleObject вернул %s", result)
                    break

        self._thread = threading.Thread(target=listen, name="activation-listener", daemon=True)
        self._thread.start()

    def notify_existing(self) -> bool:
        """Signal the already running instance. Returns True on success."""
        if sys.platform != "win32":
            return False
        try:
            import ctypes
            from ctypes import wintypes

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.OpenEventW.restype = wintypes.HANDLE
            kernel32.OpenEventW.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.LPCWSTR]
            event = kernel32.OpenEventW(_EVENT_MODIFY_STATE, False, ACTIVATION_EVENT_NAME)
            if not event:
                return False
            try:
                return bool(kernel32.SetEvent(event))
            finally:
                kernel32.CloseHandle(event)
        except Exception:
            logger.debug("Не удалось сигнализировать работающий экземпляр", exc_info=True)
            return False

    def release(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)
        if sys.platform == "win32" and self._kernel32 is not None:
            for handle in (self._event, self._mutex):
                if handle:
                    try:
                        self._kernel32.CloseHandle(handle)
                    except Exception:
                        pass
        if self._posix_handle is not None:
            try:
                self._posix_handle.close()
            except Exception:
                pass
        self._event = None
        self._mutex = None
        self._posix_handle = None


def _acquire_windows() -> InstanceLock:
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateMutexW.restype = wintypes.HANDLE
    kernel32.CreateMutexW.argtypes = [wintypes.LPVOID, wintypes.BOOL, wintypes.LPCWSTR]
    kernel32.CreateEventW.restype = wintypes.HANDLE
    kernel32.CreateEventW.argtypes = [
        wintypes.LPVOID,
        wintypes.BOOL,
        wintypes.BOOL,
        wintypes.LPCWSTR,
    ]
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]

    mutex = kernel32.CreateMutexW(None, True, MUTEX_NAME)
    last_error = ctypes.get_last_error()
    if not mutex:
        logger.warning("CreateMutexW failed: %s", ctypes.WinError(last_error))
        return InstanceLock(False)
    if last_error == _ERROR_ALREADY_EXISTS:
        kernel32.CloseHandle(mutex)
        return InstanceLock(True)

    event = kernel32.CreateEventW(None, False, False, ACTIVATION_EVENT_NAME)
    if not event:
        logger.warning("CreateEventW failed: %s", ctypes.WinError(ctypes.get_last_error()))

    lock = InstanceLock(False)
    lock._kernel32 = kernel32
    lock._mutex = mutex
    lock._event = event
    return lock


def _acquire_posix() -> InstanceLock:
    """Best-effort guard for development on Unix (activation is unsupported)."""
    import fcntl

    from . import paths

    path = paths.data_dir() / "voicetyper.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(path, "a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        return InstanceLock(True)
    lock = InstanceLock(False)
    lock._posix_handle = handle
    return lock


def acquire() -> InstanceLock:
    if sys.platform == "win32":
        try:
            return _acquire_windows()
        except Exception:
            logger.warning("Не удалось проверить единственный экземпляр", exc_info=True)
            return InstanceLock(False)
    try:
        return _acquire_posix()
    except Exception:
        return InstanceLock(False)
