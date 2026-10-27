"""Global hotkey handling.

Callbacks never run inside the Windows low-level keyboard hook: the hook only
updates a latch and enqueues an event, so slow audio/UI work cannot delay the
system-wide hook and trigger a Windows hook timeout.
"""

from __future__ import annotations

import ctypes
import logging
import queue
import threading
import time
from typing import Callable, Iterable, Optional

from .config import ConfigStore

logger = logging.getLogger(__name__)

try:
    import keyboard

    HAS_KEYBOARD = True
except Exception:
    keyboard = None
    HAS_KEYBOARD = False

try:
    _user32 = ctypes.WinDLL("user32", use_last_error=True)
except Exception:
    _user32 = None

_MAPVK_VK_TO_CHAR = 2
_MAPVK_VSC_TO_VK_EX = 3


class HotkeyError(RuntimeError):
    pass


_MODIFIERS = {"ctrl", "alt", "shift", "windows"}
_MODIFIER_ORDER = ("ctrl", "alt", "shift", "windows")
_ALIASES = {
    "control": "ctrl",
    "win": "windows",
    "return": "enter",
    "escape": "esc",
    "del": "delete",
    "ins": "insert",
    "pgup": "page up",
    "pgdn": "page down",
}
# Windows reports left/right variants while hotkey strings use neutral names.
_EVENT_ALIASES = {
    "left ctrl": "ctrl",
    "right ctrl": "ctrl",
    "left alt": "alt",
    "right alt": "alt",
    "alt gr": "alt",
    "left shift": "shift",
    "right shift": "shift",
    "left windows": "windows",
    "right windows": "windows",
    "left meta": "windows",
    "right meta": "windows",
}

# Physical keys of the Russian layout mapped to their Latin/ASCII names.
# Hotkeys must stay layout independent: otherwise a combination captured as
# "alt+ч" cannot be registered after switching to the English layout.
_CYRILLIC_TO_KEY = {
    "й": "q", "ц": "w", "у": "e", "к": "r", "е": "t", "н": "y",
    "г": "u", "ш": "i", "щ": "o", "з": "p",
    "ф": "a", "ы": "s", "в": "d", "а": "f", "п": "g", "р": "h",
    "о": "j", "л": "k", "д": "l",
    "я": "z", "ч": "x", "с": "c", "м": "v", "и": "b", "т": "n",
    "ь": "m",
}


def key_name_from_scan_code(scan_code: Optional[int]) -> str:
    """Return a layout-independent name for a physical letter/digit key."""
    if _user32 is None or scan_code is None:
        return ""
    try:
        vk = _user32.MapVirtualKeyW(int(scan_code), _MAPVK_VSC_TO_VK_EX)
        if not vk:
            return ""
        char = _user32.MapVirtualKeyW(vk, _MAPVK_VK_TO_CHAR) & 0xFFFF
    except Exception:
        return ""
    if not char:
        return ""
    name = chr(char).lower()
    return name if name.isascii() and name.isalnum() else ""


def _canonical_key_name(name: str) -> str:
    lowered = _ALIASES.get(name.lower(), name.lower())
    if len(lowered) != 1 or lowered.isascii():
        return lowered
    mapped = _CYRILLIC_TO_KEY.get(lowered)
    if mapped:
        return mapped
    if HAS_KEYBOARD:
        try:
            codes = keyboard.key_to_scan_codes(lowered)
        except Exception:
            codes = ()
        for code in codes:
            candidate = key_name_from_scan_code(code)
            if candidate:
                return candidate
    return lowered


def _is_single_key_name(text: str) -> bool:
    try:
        parsed = keyboard.parse_hotkey(text)
    except Exception:
        return False
    return len(parsed) == 1 and len(parsed[0]) == 1


def _split_hotkey_parts(hotkey: str) -> list[str]:
    if "+" in hotkey:
        raw_parts = [part.strip() for part in hotkey.split("+")]
    else:
        raw_parts = [hotkey.strip()]
    parts: list[str] = []
    for part in raw_parts:
        if not part:
            continue
        if " " in part and not _is_single_key_name(part):
            parts.extend(token for token in part.split() if token)
        else:
            parts.append(part)
    return parts


def normalize_hotkey(hotkey: str) -> tuple[str, list[str]]:
    if not hotkey or not hotkey.strip():
        raise HotkeyError("Горячая клавиша не задана")
    parts = _split_hotkey_parts(hotkey.strip())
    if not parts:
        raise HotkeyError("Горячая клавиша не задана")
    if len(parts) > 5:
        raise HotkeyError("Слишком много клавиш в комбинации")
    names: list[str] = []
    for part in parts:
        name = _canonical_key_name(part)
        if HAS_KEYBOARD:
            if not _is_single_key_name(name):
                raise HotkeyError(f"Неизвестная клавиша: {part}")
            name = keyboard.normalize_name(name)
        if name in names:
            raise HotkeyError(f"Клавиша {part} указана дважды")
        names.append(name)
    if not HAS_KEYBOARD:
        raise HotkeyError("Библиотека keyboard не установлена")
    try:
        keyboard.parse_hotkey("+".join(names))
    except Exception as exc:
        raise HotkeyError(f"Некорректная комбинация: {hotkey}") from exc
    if all(name in _MODIFIERS for name in names):
        raise HotkeyError(
            "Одних модификаторов (Ctrl/Alt/Shift/Win) недостаточно — добавьте обычную клавишу"
        )
    return "+".join(names), names


def _is_function_key(name: str) -> bool:
    return name.startswith("f") and name[1:].isdigit() and 1 <= int(name[1:]) <= 24


def hotkey_name_from_event(name: str) -> str:
    """Map a keyboard event name to the token used in hotkey strings."""
    lowered = (name or "").strip().lower()
    return _EVENT_ALIASES.get(lowered, lowered)


def hotkey_from_capture(
    modifiers: Iterable[str],
    key: str,
    scan_code: Optional[int] = None,
) -> str:
    """Build a normalized hotkey from held modifiers and the last normal key."""
    main = key_name_from_scan_code(scan_code) or hotkey_name_from_event(key)
    if not main or main in _MODIFIERS:
        raise HotkeyError(
            "Одних модификаторов (Ctrl/Alt/Shift/Win) недостаточно — добавьте обычную клавишу"
        )
    held = {hotkey_name_from_event(name) for name in modifiers}
    if not held and not _is_function_key(main):
        raise HotkeyError(
            "Добавьте модификатор (Ctrl/Alt/Shift/Win): одна обычная клавиша "
            "будет перехвачена во всех приложениях"
        )
    names = [name for name in _MODIFIER_ORDER if name in held]
    names.append(main)
    return normalize_hotkey("+".join(names))[0]


def capture_keyboard(callback: Callable[[object], bool]) -> Callable[[], None]:
    """Install a suppressing global hook while the user records a hotkey.

    The callback receives keyboard events on the hook thread and must return
    ``False`` to swallow the key so it never reaches other applications.
    """
    if not HAS_KEYBOARD:
        raise HotkeyError("Библиотека keyboard не установлена")
    return keyboard.hook(callback, suppress=True)


class HotkeyManager:
    def __init__(
        self,
        config_store: ConfigStore,
        *,
        on_press: Callable[[], None],
        on_release: Callable[[], None],
        on_error: Optional[Callable[[str], None]] = None,
    ):
        self._store = config_store
        self._on_press = on_press
        self._on_release = on_release
        self._on_error = on_error
        self._lock = threading.RLock()
        self._queue: queue.Queue[tuple[str, float]] = queue.Queue(maxsize=32)
        self._worker: Optional[threading.Thread] = None
        self._running = False
        self._handler = None
        self._hook = None
        self._keys: list[str] = []
        self._latch = False
        self._last_press = 0.0
        self._bound = ""

    @property
    def available(self) -> bool:
        return HAS_KEYBOARD

    @property
    def running(self) -> bool:
        return self._running

    @property
    def bound_hotkey(self) -> str:
        return self._bound

    def start(self) -> None:
        if not HAS_KEYBOARD:
            logger.warning("keyboard не установлен — глобальная горячая клавиша недоступна")
            return
        with self._lock:
            if self._running:
                return
            self._running = True
            cfg = self._store.get()
            normalized, keys = normalize_hotkey(cfg.hotkey)
            self._keys = keys
            try:
                self._handler = keyboard.add_hotkey(
                    normalized,
                    self._pressed,
                    suppress=True,
                    trigger_on_release=False,
                )
                self._hook = keyboard.hook(self._key_event, suppress=False)
            except Exception as exc:
                self._running = False
                self._keys = []
                self._handler = None
                self._hook = None
                raise HotkeyError(f"Не удалось зарегистрировать {normalized}: {exc}") from exc
            self._bound = normalized
            self._worker = threading.Thread(
                target=self._event_worker, name="hotkey-events", daemon=True
            )
            self._worker.start()
        logger.info("Горячая клавиша активна: %s", normalized)

    def rebind(self, hotkey: str) -> str:
        if not HAS_KEYBOARD:
            raise HotkeyError("Библиотека keyboard не установлена")
        normalized, keys = normalize_hotkey(hotkey)
        with self._lock:
            if not self._running:
                self._bound = normalized
                self._keys = keys
                return normalized
            new_handler = None
            try:
                new_handler = keyboard.add_hotkey(
                    normalized,
                    self._pressed,
                    suppress=True,
                    trigger_on_release=False,
                )
            except Exception as exc:
                raise HotkeyError(f"Не удалось зарегистрировать {normalized}: {exc}") from exc
            old_handler = self._handler
            self._handler = new_handler
            self._keys = keys
            self._bound = normalized
            self._latch = False
            if old_handler is not None:
                try:
                    keyboard.remove_hotkey(old_handler)
                except Exception:
                    logger.debug("Не удалось снять старую горячую клавишу", exc_info=True)
        logger.info("Горячая клавиша изменена: %s", normalized)
        return normalized

    def _pressed(self) -> None:
        now = time.monotonic()
        with self._lock:
            if not self._running or self._latch:
                return
            if now - self._last_press < 0.15:
                return
            self._last_press = now
            self._latch = True
        self._emit("press")

    def _key_event(self, event) -> None:
        if getattr(event, "event_type", "") != "up":
            return
        with self._lock:
            if not self._latch or not self._running:
                return
            keys = list(self._keys)
        try:
            still_pressed = any(keyboard.is_pressed(key) for key in keys)
        except Exception:
            still_pressed = False
        if still_pressed:
            return
        with self._lock:
            if not self._latch:
                return
            self._latch = False
        self._emit("release")

    def reset_latch(self) -> None:
        with self._lock:
            self._latch = False

    def _emit(self, event: str) -> None:
        try:
            self._queue.put_nowait((event, time.monotonic()))
        except queue.Full:
            logger.warning("Очередь горячих клавиш переполнена")

    def _event_worker(self) -> None:
        while True:
            event, _timestamp = self._queue.get()
            if event == "__stop__":
                return
            try:
                if event == "press":
                    self._on_press()
                else:
                    self._on_release()
            except Exception as exc:
                logger.exception("Ошибка обработчика горячей клавиши")
                if self._on_error is not None:
                    try:
                        self._on_error(f"Ошибка горячей клавиши: {exc}")
                    except Exception:
                        pass
            finally:
                self._queue.task_done()

    def close(self) -> None:
        with self._lock:
            if not self._running:
                return
            self._running = False
            handler = self._handler
            hook = self._hook
            self._handler = None
            self._hook = None
            self._latch = False
        try:
            self._queue.put_nowait(("__stop__", time.monotonic()))
        except queue.Full:
            pass
        if handler is not None:
            try:
                keyboard.remove_hotkey(handler)
            except Exception:
                pass
        if hook is not None:
            try:
                keyboard.unhook(hook)
            except Exception:
                pass
        if self._worker and self._worker.is_alive():
            self._worker.join(timeout=2)
        logger.info("Горячая клавиша отключена")
