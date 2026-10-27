from __future__ import annotations

import pytest

from voicetyper.hotkeys import (
    HotkeyError,
    hotkey_from_capture,
    hotkey_name_from_event,
    normalize_hotkey,
)


def test_normalize_common_hotkeys() -> None:
    assert normalize_hotkey("Ctrl+Alt+Space")[0] == "ctrl+alt+space"
    assert normalize_hotkey("ctrl+shift+f9")[0] == "ctrl+shift+f9"
    assert normalize_hotkey("page up")[0] == "page up"
    assert normalize_hotkey("ctrl+alt+page up")[0] == "ctrl+alt+page up"


def test_invalid_hotkeys_are_rejected() -> None:
    with pytest.raises(HotkeyError):
        normalize_hotkey("")
    with pytest.raises(HotkeyError):
        normalize_hotkey("ctrl+alt+notakey")
    with pytest.raises(HotkeyError):
        normalize_hotkey("ctrl+alt+")
    with pytest.raises(HotkeyError):
        normalize_hotkey("ctrl+alt+ctrl")
    with pytest.raises(HotkeyError):
        normalize_hotkey("ctrl+alt+shift")


def test_capture_maps_sided_modifiers() -> None:
    assert hotkey_name_from_event("left ctrl") == "ctrl"
    assert hotkey_name_from_event("right alt") == "alt"
    assert hotkey_name_from_event("Alt Gr") == "alt"
    assert hotkey_name_from_event("left shift") == "shift"
    assert hotkey_name_from_event("right windows") == "windows"
    assert hotkey_name_from_event("Space") == "space"


def test_hotkey_from_capture_builds_combination() -> None:
    assert hotkey_from_capture(["left ctrl", "right alt"], "space") == "ctrl+alt+space"
    assert hotkey_from_capture(["shift", "left ctrl"], "f9") == "ctrl+shift+f9"
    assert hotkey_from_capture([], "F9") == "f9"
    assert hotkey_from_capture(["ctrl"], "page up") == "ctrl+page up"


def test_hotkey_from_capture_uses_physical_key_not_layout_symbol() -> None:
    assert hotkey_from_capture(["alt"], "ч", 45) == "alt+x"
    assert hotkey_from_capture(["ctrl", "left alt"], "Ч", 45) == "ctrl+alt+x"
    assert hotkey_from_capture(["alt"], "x", 45) == "alt+x"


def test_cyrillic_hotkey_is_normalized_to_latin() -> None:
    assert normalize_hotkey("alt+ч")[0] == "alt+x"
    assert normalize_hotkey("Alt+Ч")[0] == "alt+x"


def test_hotkey_from_capture_requires_normal_key() -> None:
    for modifiers in (["left ctrl", "right alt"], ["ctrl", "shift"], []):
        with pytest.raises(HotkeyError):
            hotkey_from_capture(modifiers, "left shift")
    with pytest.raises(HotkeyError):
        hotkey_from_capture(["ctrl"], "notakey")


def test_hotkey_from_capture_rejects_bare_regular_key() -> None:
    with pytest.raises(HotkeyError):
        hotkey_from_capture([], "space")
    with pytest.raises(HotkeyError):
        hotkey_from_capture([], "a")
    assert hotkey_from_capture([], "f10") == "f10"
