"""User-facing labels shared by the tray menu and the settings window.

Keeping them in one module prevents the two UI surfaces from drifting apart
(a language added to the settings window but not to the tray menu, a different
wording for the same option, and so on). They are still Russian literals: a
real translation layer is a separate task (see ``CODE_REVIEW_2.md``, F7).
"""

from __future__ import annotations

LANGUAGE_LABELS = {
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

# Short glyphs drawn on the tray icon while idle.
LANGUAGE_CODES = {
    "auto": "A",
    "ru": "RU",
    "en": "EN",
    "de": "DE",
    "fr": "FR",
    "es": "ES",
    "it": "IT",
    "pt": "PT",
    "zh": "中",
    "ja": "日",
}

TONE_DECODER_LABELS = {
    "auto": "Авто (по железу)",
    "beam_search": "Beam search (точнее)",
    "greedy": "Greedy (легче)",
}

DEVICE_LABELS = {"auto": "Авто", "cpu": "CPU", "cuda": "CUDA"}

COMPUTE_TYPE_LABELS = {
    "auto": "Авто",
    "int8": "int8",
    "int8_float16": "int8_float16",
    "float16": "float16",
    "float32": "float32",
}

HOTKEY_MODE_LABELS = {"hold": "Удержание", "toggle": "Переключение"}
