"""System tray icon, menu and status indication."""

from __future__ import annotations

import logging
import threading
from typing import Any, Optional

from .config import ConfigStore
from .constants import (
    APP_NAME,
    APP_TITLE,
    APP_VERSION,
    ENGINE_LABELS,
    ENGINE_ORDER,
    LANGUAGES,
    STATUS_ERROR,
    STATUS_IDLE,
    STATUS_LOADING,
    STATUS_READY,
    STATUS_RECORDING,
    STATUS_TRANSCRIBING,
    WHISPER_MODELS,
)
from .engines import describe_engines

logger = logging.getLogger(__name__)

try:
    import pystray
    from PIL import Image, ImageDraw, ImageFont

    HAS_TRAY = True
except Exception:
    pystray = None
    Image = None
    ImageDraw = None
    ImageFont = None
    HAS_TRAY = False

_LANGUAGE_LABELS = {
    "auto": "Авто",
    "ru": "Русский",
    "en": "English",
    "de": "Deutsch",
    "fr": "Français",
    "es": "Español",
    "it": "Italiano",
    "pt": "Português",
    "zh": "中文",
    "ja": "日本語",
}
_LANGUAGE_CODES = {"auto": "A", "ru": "RU", "en": "EN", "de": "DE", "fr": "FR", "es": "ES", "it": "IT", "pt": "PT", "zh": "中", "ja": "日"}

_STATE_COLORS = {
    STATUS_IDLE: (0, 122, 66),
    STATUS_LOADING: (190, 110, 0),
    STATUS_RECORDING: (190, 25, 25),
    STATUS_TRANSCRIBING: (30, 80, 180),
    STATUS_ERROR: (140, 0, 0),
}
_STATE_GLYPHS = {
    STATUS_IDLE: None,
    STATUS_LOADING: "…",
    STATUS_RECORDING: "REC",
    STATUS_TRANSCRIBING: "…",
    STATUS_ERROR: "!",
}

_font_cache: dict[tuple[str, int], Any] = {}
_icon_cache: dict[tuple[str, str], Any] = {}


def _font(size: int):
    key = ("arialbd.ttf", size)
    if key in _font_cache:
        return _font_cache[key]
    try:
        font = ImageFont.truetype("arialbd.ttf", size)
    except Exception:
        try:
            font = ImageFont.truetype("arial.ttf", size)
        except Exception:
            font = ImageFont.load_default()
    _font_cache[key] = font
    return font


def create_image(text: str, state: str = STATUS_IDLE, size: int = 64):
    if not HAS_TRAY:
        return None
    cache_key = (text, state)
    cached = _icon_cache.get(cache_key)
    if cached is not None:
        return cached
    color = _STATE_COLORS.get(state, _STATE_COLORS[STATUS_IDLE])
    image = Image.new("RGB", (size, size), color)
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 0, size - 1, size - 1], outline=tuple(max(0, c - 35) for c in color))
    length = max(1, len(text))
    font_size = 22 if length <= 2 else 16 if length == 3 else 12
    font = _font(font_size)
    try:
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        width, height = right - left, bottom - top
        draw.text(
            ((size - width) // 2 - left, (size - height) // 2 - top),
            text,
            fill=(255, 255, 255),
            font=font,
        )
    except Exception:
        draw.text((8, 20), text, fill=(255, 255, 255), font=font)
    _icon_cache[cache_key] = image
    return image


class TrayController:
    def __init__(self, app, config_store: ConfigStore):
        self._app = app
        self._store = config_store
        self._icon = None
        self._state = STATUS_IDLE
        self._message = "Готово"
        self._progress: Optional[float] = None
        self._engine: Optional[str] = None
        self._last_key = None
        self._refresh_lock = threading.Lock()
        self._available = HAS_TRAY

    @property
    def available(self) -> bool:
        return self._available

    @property
    def icon(self):
        return self._icon

    def run(self, setup=None) -> None:
        if not HAS_TRAY:
            raise RuntimeError("pystray не установлен")
        cfg = self._store.get()
        self._icon = pystray.Icon(
            APP_NAME,
            self._image_for(cfg),
            self._title(cfg),
            self.build_menu(),
        )

        def ready(icon):
            # pystray only shows the icon automatically when no custom setup
            # callback is supplied, so make it visible explicitly.
            icon.visible = True
            logger.info("Трей запущен")
            if setup is not None:
                try:
                    setup(icon)
                except Exception:
                    logger.exception("Ошибка инициализации трея")

        self._icon.run(setup=ready)

    def dispose(self) -> None:
        icon = self._icon
        self._icon = None
        if icon is not None:
            try:
                icon.visible = False
            except Exception:
                pass
            try:
                icon.stop()
            except Exception:
                logger.debug("Не удалось остановить трей", exc_info=True)

    def notify(self, title: str, message: str) -> bool:
        if self._icon is None:
            return False
        try:
            self._icon.notify(message, title)
            return True
        except Exception:
            logger.debug("Уведомление трея не показано", exc_info=True)
            return False

    # -- state -----------------------------------------------------------
    def set_state(
        self,
        state: str,
        message: str = "",
        *,
        engine: Optional[str] = None,
        progress: Optional[float] = None,
    ) -> None:
        if engine is not None:
            self._engine = engine
        key = (state, message, round(progress) if progress is not None else None, self._engine)
        if key == self._last_key:
            return
        state_changed = state != self._state
        self._last_key = key
        self._state = state
        self._message = message
        self._progress = progress
        cfg = self._store.get()
        icon = self._icon
        if icon is None:
            return
        try:
            icon.icon = self._image_for(cfg)
            icon.title = self._title(cfg)
        except Exception:
            logger.debug("Не удалось обновить иконку трея", exc_info=True)
        if state_changed:
            self.refresh_menu()

    def _image_for(self, cfg):
        glyph = _STATE_GLYPHS.get(self._state)
        if glyph is None:
            glyph = _LANGUAGE_CODES.get(cfg.language, "?")
        return create_image(glyph, self._state)

    def _title(self, cfg) -> str:
        base = APP_TITLE
        if self._state == STATUS_LOADING:
            suffix = f" · {self._message}"
            if self._progress is not None:
                suffix += f" {self._progress:.0f}%"
        elif self._state == STATUS_RECORDING:
            suffix = " · запись"
            if self._message:
                suffix += f": {self._message[:48]}"
        elif self._state == STATUS_TRANSCRIBING:
            suffix = " · распознавание"
        elif self._state == STATUS_ERROR:
            suffix = f" · ошибка: {self._message}"
        else:
            suffix = f" · {self._engine}" if self._engine else ""
        return (base + suffix)[:120]

    def refresh_menu(self) -> None:
        icon = self._icon
        if icon is None:
            return
        if not self._refresh_lock.acquire(blocking=False):
            return
        try:
            icon.menu = self.build_menu()
            icon.update_menu()
        except Exception:
            logger.debug("Не удалось обновить меню трея", exc_info=True)
        finally:
            self._refresh_lock.release()

    # -- menu ------------------------------------------------------------
    def build_menu(self):
        if not HAS_TRAY:
            return None
        cfg = self._store.get()
        try:
            availability = describe_engines()
        except Exception:
            availability = {}
        try:
            recording = bool(self._app.is_recording)
        except Exception:
            recording = False

        status = self._status_text(cfg)
        engine_items = []
        for name in ENGINE_ORDER:
            label = ENGINE_LABELS.get(name, name)
            installed = True
            if name != "auto":
                info = availability.get(name)
                installed = bool(info and info.installed)
                if not installed:
                    label += " (не установлен)"
            engine_items.append(
                pystray.MenuItem(
                    label,
                    self._action(self._app.set_engine, name),
                    checked=self._checked("model_type", name),
                    enabled=installed,
                )
            )

        language_items = [
            pystray.MenuItem(
                _LANGUAGE_LABELS.get(code, code),
                self._action(self._app.set_language, code),
                checked=self._checked("language", code),
            )
            for code in LANGUAGES
        ]

        whisper_items = [
            pystray.MenuItem(
                name,
                self._action(self._app.set_whisper_model, name),
                checked=self._checked("whisper_model", name),
            )
            for name in WHISPER_MODELS
        ]

        tone_items = [
            pystray.MenuItem(
                label,
                self._action(self._app.set_tone_decoder, value),
                checked=self._checked("tone_decoder", value),
            )
            for value, label in (
                ("auto", "Авто (по железу)"),
                ("beam_search", "Beam search (точнее)"),
                ("greedy", "Greedy (легче)"),
            )
        ]

        mode_items = [
            pystray.MenuItem(
                label,
                self._action(self._app.set_hotkey_mode, value),
                checked=self._checked("hotkey_mode", value),
            )
            for value, label in (("hold", "Удержание"), ("toggle", "Переключение"))
        ]

        autostart_enabled = bool(cfg.autostart)
        record_label = "Остановить запись" if recording else "Начать запись"
        hotkey_label = f"Клавиша: {cfg.hotkey} · {'удержание' if cfg.hotkey_mode == 'hold' else 'переключение'}"

        return pystray.Menu(
            pystray.MenuItem(f"{APP_TITLE} {APP_VERSION}", None, enabled=False),
            pystray.MenuItem(status, None, enabled=False),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem(
                record_label,
                self._action(self._app.toggle_recording),
                default=recording,
            ),
            pystray.MenuItem(
                "Настройки…",
                self._action(self._app.open_settings),
                default=not recording,
            ),
            pystray.MenuItem("История…", self._action(self._app.open_history)),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Движок", pystray.Menu(*engine_items)),
            pystray.MenuItem("Модель Whisper", pystray.Menu(*whisper_items)),
            pystray.MenuItem("Язык", pystray.Menu(*language_items)),
            pystray.MenuItem("Декодер T-one", pystray.Menu(*tone_items)),
            pystray.MenuItem("Режим клавиши", pystray.Menu(*mode_items)),
            pystray.MenuItem(hotkey_label, None, enabled=False),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem(
                "Запускать при входе",
                self._action(self._app.set_autostart, not autostart_enabled),
                checked=lambda _item: bool(self._store.get().autostart),
            ),
            pystray.MenuItem("Открыть папку данных", self._action(self._app.open_data_dir)),
            pystray.MenuItem("Открыть логи", self._action(self._app.open_logs)),
            pystray.MenuItem("Перезагрузить модель", self._action(self._app.reload_model)),
            pystray.MenuItem("Диагностика", self._action(self._app.run_diagnostics)),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Выход", self._action(self._app.shutdown)),
        )

    def _status_text(self, cfg) -> str:
        state = self._state
        if state == STATUS_LOADING:
            suffix = f" {self._progress:.0f}%" if self._progress is not None else ""
            text = f"Загрузка: {self._message}{suffix}"
        elif state == STATUS_RECORDING:
            text = "● Идёт запись…"
            if self._message:
                text += f" {self._message}"
        elif state == STATUS_TRANSCRIBING:
            text = "Распознавание…"
        elif state == STATUS_ERROR:
            text = f"Ошибка: {self._message}"
        elif state == STATUS_READY:
            text = f"Готово · {self._engine or ENGINE_LABELS.get(cfg.model_type, cfg.model_type)}"
        else:
            text = "Готово"
        return text[:80]

    def _checked(self, field: str, value):
        return lambda _item: getattr(self._store.get(), field, None) == value

    def _action(self, function, *args):
        def action(_icon, _item):
            try:
                function(*args)
            except Exception:
                logger.exception("Ошибка действия меню")

        return action
