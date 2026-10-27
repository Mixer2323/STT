"""Tray balloon notifications with rate limiting."""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)

# Windows tray balloons reject strings longer than 256 characters.
_MAX_TITLE = 100
_MAX_MESSAGE = 240


def _clip(text: str, limit: int) -> str:
    text = str(text).strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


class Notifier:
    def __init__(self, enabled: bool = True):
        self._icon = None
        self._lock = threading.Lock()
        self._last: dict[str, float] = {}
        self.enabled = enabled

    def set_icon(self, icon) -> None:
        self._icon = icon

    def notify(
        self,
        title: str,
        message: str,
        *,
        key: Optional[str] = None,
        cooldown: float = 5.0,
        force: bool = False,
    ) -> bool:
        if not self.enabled:
            return False
        now = time.monotonic()
        with self._lock:
            if not force and key:
                last = self._last.get(key, 0.0)
                if now - last < cooldown:
                    return False
                self._last[key] = now
            icon = self._icon
        if icon is None:
            logger.info("%s: %s", title, message)
            return False
        try:
            icon.notify(_clip(message, _MAX_MESSAGE), _clip(title, _MAX_TITLE))
            return True
        except Exception:
            logger.warning("Не удалось показать уведомление «%s»", title, exc_info=True)
            return False
