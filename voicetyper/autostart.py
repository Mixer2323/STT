"""Windows autostart via HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Run."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

_RUN_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"
_VALUE_NAME = "VoiceTyper"


def _entry_point() -> str:
    if getattr(sys, "frozen", False):
        return f'"{Path(sys.executable)}" --minimized'
    python = Path(sys.executable)
    pythonw = python.with_name("pythonw.exe")
    if pythonw.exists():
        python = pythonw
    script = Path(__file__).resolve().parent.parent / "voice_typer.py"
    return f'"{python}" "{script}" --minimized'


def is_supported() -> bool:
    return sys.platform == "win32"


def is_enabled() -> bool:
    if not is_supported():
        return False
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, _RUN_KEY) as key:
            value, _kind = winreg.QueryValueEx(key, _VALUE_NAME)
            return bool(value)
    except FileNotFoundError:
        return False
    except OSError:
        logger.debug("Не удалось прочитать автозапуск", exc_info=True)
        return False


def set_enabled(enabled: bool) -> tuple[bool, str]:
    if not is_supported():
        return False, "Автозапуск поддерживается только в Windows"
    try:
        import winreg

        with winreg.OpenKey(
            winreg.HKEY_CURRENT_USER, _RUN_KEY, 0, winreg.KEY_SET_VALUE
        ) as key:
            if enabled:
                winreg.SetValueEx(key, _VALUE_NAME, 0, winreg.REG_SZ, _entry_point())
                return True, "Автозапуск включён"
            try:
                winreg.DeleteValue(key, _VALUE_NAME)
            except FileNotFoundError:
                pass
            return True, "Автозапуск выключен"
    except OSError as exc:
        logger.warning("Не удалось изменить автозапуск: %s", exc)
        return False, f"Не удалось изменить автозапуск: {exc}"


def actual_state() -> bool:
    return is_enabled()
