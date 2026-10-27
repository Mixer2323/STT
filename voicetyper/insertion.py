"""Text insertion into the active application (clipboard, paste or typing)."""

from __future__ import annotations

import ctypes
import logging
import sys
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

from .config import Config

logger = logging.getLogger(__name__)

try:
    import pyperclip

    HAS_PYPERCLIP = True
except Exception:
    pyperclip = None
    HAS_PYPERCLIP = False

try:
    import keyboard

    HAS_KEYBOARD = True
except Exception:
    keyboard = None
    HAS_KEYBOARD = False


class InsertionError(RuntimeError):
    pass


@dataclass(frozen=True)
class WindowTarget:
    hwnd: int
    title: str = ""


@dataclass
class InsertionResult:
    method: str
    inserted: bool = False
    clipboard_restored: bool = False
    target_focused: Optional[bool] = None
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Win32 helpers
# ---------------------------------------------------------------------------
_CF_UNICODETEXT = 13
_VK_CONTROL = 0x11
_VK_V = 0x56
_INPUT_KEYBOARD = 1
_KEYEVENTF_KEYUP = 0x0002
_SW_RESTORE = 9


@lru_cache(maxsize=1)
def _user32():
    if sys.platform != "win32":
        return None
    try:
        from ctypes import wintypes

        user32 = ctypes.WinDLL("user32", use_last_error=True)
        user32.GetForegroundWindow.restype = wintypes.HWND
        user32.GetForegroundWindow.argtypes = []
        user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
        user32.GetWindowTextLengthW.restype = ctypes.c_int
        user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
        user32.GetWindowTextW.restype = ctypes.c_int
        user32.IsWindow.argtypes = [wintypes.HWND]
        user32.IsWindow.restype = wintypes.BOOL
        user32.IsIconic.argtypes = [wintypes.HWND]
        user32.IsIconic.restype = wintypes.BOOL
        user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
        user32.ShowWindow.restype = wintypes.BOOL
        user32.SetForegroundWindow.argtypes = [wintypes.HWND]
        user32.SetForegroundWindow.restype = wintypes.BOOL
        user32.BringWindowToTop.argtypes = [wintypes.HWND]
        user32.BringWindowToTop.restype = wintypes.BOOL
        user32.AttachThreadInput.argtypes = [wintypes.DWORD, wintypes.DWORD, wintypes.BOOL]
        user32.AttachThreadInput.restype = wintypes.BOOL
        user32.GetWindowThreadProcessId.argtypes = [
            wintypes.HWND,
            ctypes.POINTER(wintypes.DWORD),
        ]
        user32.GetWindowThreadProcessId.restype = wintypes.DWORD
        user32.IsClipboardFormatAvailable.argtypes = [wintypes.UINT]
        user32.IsClipboardFormatAvailable.restype = wintypes.BOOL
        user32.GetClipboardSequenceNumber.restype = wintypes.DWORD
        user32.GetClipboardSequenceNumber.argtypes = []
        return user32
    except Exception:
        logger.debug("user32 недоступен", exc_info=True)
        return None


def foreground_window() -> Optional[WindowTarget]:
    user32 = _user32()
    if user32 is None:
        return None
    try:
        hwnd = int(user32.GetForegroundWindow() or 0)
        if not hwnd:
            return None
        length = int(user32.GetWindowTextLengthW(hwnd))
        buffer = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buffer, length + 1)
        return WindowTarget(hwnd=hwnd, title=buffer.value)
    except Exception:
        logger.debug("Не удалось получить активное окно", exc_info=True)
        return None


def is_window(hwnd: int) -> bool:
    user32 = _user32()
    if user32 is None:
        return False
    try:
        return bool(user32.IsWindow(int(hwnd)))
    except Exception:
        return False


def focus_window(hwnd: int) -> bool:
    user32 = _user32()
    if user32 is None:
        return False
    try:
        target = int(hwnd)
        if not user32.IsWindow(target):
            return False
        if user32.IsIconic(target):
            user32.ShowWindow(target, _SW_RESTORE)

        current = int(user32.GetForegroundWindow() or 0)
        if current == target:
            return True

        current_thread = int(ctypes.windll.kernel32.GetCurrentThreadId())
        target_thread = int(user32.GetWindowThreadProcessId(current, None)) if current else 0
        attached = False
        if target_thread and target_thread != current_thread:
            attached = bool(user32.AttachThreadInput(current_thread, target_thread, True))
        try:
            user32.SetForegroundWindow(target)
            user32.BringWindowToTop(target)
        finally:
            if attached:
                user32.AttachThreadInput(current_thread, target_thread, False)
        time.sleep(0.05)
        return int(user32.GetForegroundWindow() or 0) == target
    except Exception:
        logger.debug("Не удалось переключить окно", exc_info=True)
        return False


def _clipboard_has_text() -> Optional[bool]:
    user32 = _user32()
    if user32 is None:
        return None
    try:
        return bool(user32.IsClipboardFormatAvailable(_CF_UNICODETEXT))
    except Exception:
        return None


def _clipboard_sequence() -> Optional[int]:
    user32 = _user32()
    if user32 is None:
        return None
    try:
        return int(user32.GetClipboardSequenceNumber())
    except Exception:
        return None


class _KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", ctypes.c_ushort),
        ("wScan", ctypes.c_ushort),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", ctypes.c_size_t),
    ]


class _MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", ctypes.c_long),
        ("dy", ctypes.c_long),
        ("mouseData", ctypes.c_ulong),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", ctypes.c_size_t),
    ]


class _HARDWAREINPUT(ctypes.Structure):
    _fields_ = [
        ("uMsg", ctypes.c_ulong),
        ("wParamL", ctypes.c_ushort),
        ("wParamH", ctypes.c_ushort),
    ]


class _INPUTUNION(ctypes.Union):
    _fields_ = [
        ("ki", _KEYBDINPUT),
        ("mi", _MOUSEINPUT),
        ("hi", _HARDWAREINPUT),
    ]


class _INPUT(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong), ("union", _INPUTUNION)]


def _send_ctrl_v_win32() -> bool:
    user32 = _user32()
    if user32 is None:
        return False
    try:
        events = (_INPUT * 4)()
        definitions = (
            (_VK_CONTROL, 0),
            (_VK_V, 0),
            (_VK_V, _KEYEVENTF_KEYUP),
            (_VK_CONTROL, _KEYEVENTF_KEYUP),
        )
        for event, (vk, flags) in zip(events, definitions, strict=True):
            event.type = _INPUT_KEYBOARD
            event.union.ki.wVk = vk
            event.union.ki.wScan = 0
            event.union.ki.dwFlags = flags
        sent = user32.SendInput(4, events, ctypes.sizeof(_INPUT))
        return sent == 4
    except Exception:
        logger.debug("SendInput не сработал", exc_info=True)
        return False


def _send_paste() -> None:
    if not _send_ctrl_v_win32():
        if not HAS_KEYBOARD:
            raise InsertionError("Не удалось отправить Ctrl+V: keyboard не установлен")
        keyboard.press_and_release("ctrl+v")


def _copy(text: str) -> None:
    if not HAS_PYPERCLIP:
        raise InsertionError("pyperclip не установлен — вставка невозможна")
    last_error: Optional[Exception] = None
    for attempt in range(3):
        try:
            pyperclip.copy(text)
            return
        except Exception as exc:
            last_error = exc
            time.sleep(0.05 * (attempt + 1))
    raise InsertionError(f"Не удалось скопировать текст в буфер обмена: {last_error}")


def _read_clipboard_text() -> Optional[str]:
    if not HAS_PYPERCLIP:
        return None
    has_text = _clipboard_has_text()
    if has_text is False:
        # Do not overwrite images/files with an empty string later.
        return None
    try:
        return pyperclip.paste()
    except Exception:
        return None


def insert_text(
    text: str,
    cfg: Config,
    *,
    target: Optional[WindowTarget] = None,
) -> InsertionResult:
    """Insert ``text`` according to ``cfg.insert_mode``."""
    if not text.strip():
        return InsertionResult(method=cfg.insert_mode, error="Пустой текст")

    focused: Optional[bool] = None
    if target is not None and cfg.restore_target_window:
        if is_window(target.hwnd):
            focused = focus_window(target.hwnd)
            if not focused:
                logger.warning("Не удалось вернуть фокус в «%s»", target.title)

    if cfg.insert_mode == "clipboard":
        try:
            _copy(text)
            return InsertionResult(
                method="clipboard",
                inserted=False,
                target_focused=focused,
            )
        except InsertionError as exc:
            return InsertionResult(method="clipboard", error=str(exc))

    if cfg.insert_mode == "type":
        if not HAS_KEYBOARD:
            return InsertionResult(method="type", target_focused=focused, error="keyboard не установлен")
        try:
            keyboard.write(text, delay=0.002)
            return InsertionResult(method="type", inserted=True, target_focused=focused)
        except Exception as exc:
            return InsertionResult(method="type", target_focused=focused, error=f"Ошибка ввода: {exc}")

    old_text = _read_clipboard_text()
    try:
        _copy(text)
    except InsertionError as exc:
        return InsertionResult(method="paste", target_focused=focused, error=str(exc))

    ours_sequence = _clipboard_sequence()
    time.sleep(max(0, cfg.paste_delay_ms) / 1000.0)
    try:
        _send_paste()
    except InsertionError as exc:
        return InsertionResult(method="paste", target_focused=focused, error=str(exc))
    except Exception as exc:
        return InsertionResult(method="paste", target_focused=focused, error=f"Ошибка вставки: {exc}")

    restored = False
    if cfg.restore_clipboard and old_text is not None and old_text != text:
        time.sleep(0.2)
        current_sequence = _clipboard_sequence()
        if ours_sequence is None or current_sequence == ours_sequence:
            try:
                _copy(old_text)
                restored = True
            except InsertionError:
                logger.debug("Не удалось восстановить буфер обмена", exc_info=True)

    return InsertionResult(
        method="paste",
        inserted=True,
        clipboard_restored=restored,
        target_focused=focused,
    )
