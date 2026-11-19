from __future__ import annotations

import pytest

from voicetyper import insertion
from voicetyper.config import Config

_RU = 0x04190419
# VkKeyScanEx results: low byte = virtual key, high byte = required shift state.
_KEYS = {
    "ф": 0x0041,  # A key, no shift
    "Ф": 0x0141,  # A key with shift
    "и": 0x0042,  # B key
    " ": 0x0020,
    "^": 0x025E,  # requires Ctrl — a shortcut, not text
    "😀": -1,  # no key on this layout
}
_SCANS = {0x10: 42, 0x41: 30, 0x42: 48, 0x20: 57, 0x5E: 41}


class FakeUser32:
    def __init__(self, *, accepted: bool = True) -> None:
        self.batches: list[list[tuple[int, int, int]]] = []
        self.accepted = accepted

    def GetForegroundWindow(self) -> int:
        return 0

    def GetKeyboardLayout(self, _thread_id: int) -> int:
        return _RU

    def GetWindowThreadProcessId(self, _hwnd: int, _lp) -> int:
        return 0

    def VkKeyScanExW(self, char: str, _hkl) -> int:
        return _KEYS.get(char, -1)

    def MapVirtualKeyExW(self, vk: int, _map_type: int, _hkl) -> int:
        return _SCANS.get(vk, 0)

    def SendInput(self, count: int, events, _size: int) -> int:
        parsed = [
            (event.union.ki.wVk, event.union.ki.wScan, event.union.ki.dwFlags)
            for event in events
        ]
        if not self.accepted:
            return count - 1 if count else 0
        self.batches.append(parsed)
        return count


@pytest.fixture(autouse=True)
def _fresh_key_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(insertion, "_KEY_CACHE", {})


def test_lowercase_uses_the_physical_scan_code(monkeypatch: pytest.MonkeyPatch) -> None:
    user32 = FakeUser32()
    monkeypatch.setattr(insertion, "_user32", lambda: user32)

    assert insertion.type_scancodes("и", delay_ms=0)

    scan = insertion._KEYEVENTF_SCANCODE
    assert user32.batches == [[(0, 48, scan), (0, 48, scan | insertion._KEYEVENTF_KEYUP)]]


def test_capital_letters_press_shift_with_a_scan_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user32 = FakeUser32()
    monkeypatch.setattr(insertion, "_user32", lambda: user32)

    insertion.type_scancodes("Ф", delay_ms=0)

    up = insertion._KEYEVENTF_KEYUP
    scan = insertion._KEYEVENTF_SCANCODE
    assert user32.batches == [
        [(0, 42, scan), (0, 30, scan), (0, 30, scan | up), (0, 42, scan | up)]
    ]


def test_ctrl_v_uses_scan_codes(monkeypatch: pytest.MonkeyPatch) -> None:
    # RDP forwards physical keys, so the paste shortcut must not rely on wVk.
    user32 = FakeUser32()
    monkeypatch.setattr(insertion, "_user32", lambda: user32)

    assert insertion._send_ctrl_v_win32()

    up = insertion._KEYEVENTF_KEYUP
    scan = insertion._KEYEVENTF_SCANCODE
    assert user32.batches[-1] == [
        (0, 0x1D, scan),
        (0, 0x2F, scan),
        (0, 0x2F, scan | up),
        (0, 0x1D, scan | up),
    ]


def test_keys_that_would_be_shortcuts_are_injected_as_unicode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user32 = FakeUser32()
    monkeypatch.setattr(insertion, "_user32", lambda: user32)

    insertion.type_scancodes("^😀", delay_ms=0)

    unicode_flags = insertion._KEYEVENTF_UNICODE
    assert user32.batches[0] == [
        (0, 0x5E, unicode_flags),
        (0, 0x5E, unicode_flags | insertion._KEYEVENTF_KEYUP),
    ]
    # Surrogate pairs are sent as UTF-16 units, each pressed and released.
    assert [event[1] for event in user32.batches[1]] == [0xD83D, 0xD83D, 0xDE00, 0xDE00]


def test_spacing_between_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    user32 = FakeUser32()
    sleeps: list[float] = []
    monkeypatch.setattr(insertion, "_user32", lambda: user32)
    monkeypatch.setattr(insertion.time, "sleep", lambda value: sleeps.append(value))

    insertion.type_scancodes("иф", delay_ms=25)

    assert sleeps == [0.025, 0.025]


def test_reports_failure_when_sendinput_is_blocked(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(insertion, "_user32", lambda: FakeUser32(accepted=False))

    assert not insertion.type_scancodes("и", delay_ms=0)


def test_unicode_fallback_characters_are_counted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(insertion, "_user32", lambda: FakeUser32())

    result = insertion.insert_text(
        "^😀", Config(insert_mode="type_scancode", type_delay_ms=0)
    )

    assert result.inserted
    # "^" is a Ctrl shortcut and "😀" has no key: both go through Unicode.
    assert result.unicode_fallback == 2


def test_off_windows_there_is_no_sendinput(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(insertion, "_user32", lambda: None)

    assert not insertion.type_scancodes("и")


def test_scancode_mode_never_touches_the_clipboard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user32 = FakeUser32()
    monkeypatch.setattr(insertion, "_user32", lambda: user32)

    def forbidden(_text: str) -> None:
        raise AssertionError("Буфер обмена использовать нельзя")

    monkeypatch.setattr(insertion, "_copy", forbidden)

    result = insertion.insert_text("иф", Config(insert_mode="type_scancode", type_delay_ms=0))

    assert result.inserted
    assert result.method == "type_scancode"
    assert result.error is None
    assert len(user32.batches) == 2
    scan = insertion._KEYEVENTF_SCANCODE
    assert [
        event[1] for batch in user32.batches for event in batch if event[2] & scan
    ] == [48, 48, 30, 30]


def test_scancode_mode_reports_an_error_when_input_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(insertion, "_user32", lambda: FakeUser32(accepted=False))

    result = insertion.insert_text("привет", Config(insert_mode="type_scancode"))

    assert not result.inserted
    assert result.error


def _stub_paste(monkeypatch: pytest.MonkeyPatch, *, snapshot, sequences):
    monkeypatch.setattr(insertion, "_snapshot_clipboard", lambda: snapshot)
    monkeypatch.setattr(insertion, "_copy", lambda _text: None)
    monkeypatch.setattr(insertion, "_send_paste", lambda: None)
    monkeypatch.setattr(insertion.time, "sleep", lambda _value: None)
    remaining = iter(sequences)
    monkeypatch.setattr(insertion, "_clipboard_sequence", lambda: next(remaining))
    restored: list = []
    monkeypatch.setattr(
        insertion,
        "_restore_clipboard",
        lambda snap: restored.append(snap) or True,
    )
    return restored


def test_paste_restores_images_and_files(monkeypatch: pytest.MonkeyPatch) -> None:
    snapshot = [(8, b"DIB-BYTES"), (15, b"HDROP-FILES")]
    restored = _stub_paste(monkeypatch, snapshot=snapshot, sequences=[5, 5])

    result = insertion.insert_text(
        "привет", Config(insert_mode="paste", paste_delay_ms=0)
    )

    assert result.inserted
    assert result.clipboard_restored
    assert restored == [snapshot]


def test_paste_keeps_the_new_clipboard_when_the_target_changed_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    restored = _stub_paste(monkeypatch, snapshot=[(8, b"DIB")], sequences=[5, 9])

    result = insertion.insert_text(
        "привет", Config(insert_mode="paste", paste_delay_ms=0)
    )

    assert result.inserted
    assert not result.clipboard_restored
    assert restored == []


def test_paste_waits_for_the_configured_restore_delay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A slow target reads the clipboard after Ctrl+V: restoring it too early
    # makes that application paste the old contents instead of the dictation.
    sleeps: list[float] = []
    monkeypatch.setattr(insertion, "_snapshot_clipboard", lambda: [(8, b"DIB")])
    monkeypatch.setattr(insertion, "_copy", lambda _text: None)
    monkeypatch.setattr(insertion, "_send_paste", lambda: None)
    monkeypatch.setattr(insertion.time, "sleep", lambda value: sleeps.append(value))
    monkeypatch.setattr(insertion, "_clipboard_sequence", lambda: 5)
    monkeypatch.setattr(insertion, "_restore_clipboard", lambda _snap: True)

    result = insertion.insert_text(
        "привет",
        Config(insert_mode="paste", paste_delay_ms=70, restore_delay_ms=450),
    )

    assert result.clipboard_restored
    assert sleeps == [0.07, 0.45]


def test_closed_target_window_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(insertion, "is_window", lambda _hwnd: False)
    monkeypatch.setattr(insertion, "_copy", lambda _text: None)
    monkeypatch.setattr(insertion, "_send_paste", lambda: None)
    monkeypatch.setattr(insertion.time, "sleep", lambda _value: None)
    monkeypatch.setattr(insertion, "_clipboard_sequence", lambda: 5)

    result = insertion.insert_text(
        "привет",
        Config(insert_mode="paste", restore_clipboard=False),
        target=insertion.WindowTarget(hwnd=42, title="Блокнот"),
    )

    assert result.inserted
    assert result.target_missing


def test_existing_target_window_is_not_marked_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(insertion, "is_window", lambda _hwnd: True)
    monkeypatch.setattr(insertion, "focus_window", lambda _hwnd: True)
    monkeypatch.setattr(insertion, "_copy", lambda _text: None)
    monkeypatch.setattr(insertion, "_send_paste", lambda: None)
    monkeypatch.setattr(insertion.time, "sleep", lambda _value: None)
    monkeypatch.setattr(insertion, "_clipboard_sequence", lambda: 5)

    result = insertion.insert_text(
        "привет",
        Config(insert_mode="paste", restore_clipboard=False),
        target=insertion.WindowTarget(hwnd=42, title="Блокнот"),
    )

    assert result.inserted
    assert not result.target_missing
    assert result.target_focused is True


def test_paste_does_not_snapshot_when_restore_is_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshots = []
    monkeypatch.setattr(insertion, "_snapshot_clipboard", lambda: snapshots.append(True))
    monkeypatch.setattr(insertion, "_copy", lambda _text: None)
    monkeypatch.setattr(insertion, "_send_paste", lambda: None)
    monkeypatch.setattr(insertion.time, "sleep", lambda _value: None)
    monkeypatch.setattr(insertion, "_clipboard_sequence", lambda: 5)
    restored = []
    monkeypatch.setattr(
        insertion,
        "_restore_clipboard",
        lambda snap: restored.append(snap) or True,
    )

    result = insertion.insert_text(
        "привет",
        Config(insert_mode="paste", restore_clipboard=False, paste_delay_ms=0),
    )

    assert result.inserted
    assert not result.clipboard_restored
    assert snapshots == []
    assert restored == []


def test_clipboard_snapshot_requires_win32(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(insertion, "_user32", lambda: None)

    assert insertion._snapshot_clipboard() is None
    assert insertion._restore_clipboard([(8, b"DIB")]) is False


class _ClipboardUser32:
    def __init__(self, *, failures: int = 0) -> None:
        self.failures = failures
        self.opened = 0
        self.emptied = False
        self.events: list[str] = []

    def OpenClipboard(self, _hwnd):
        self.opened += 1
        if self.opened <= self.failures:
            raise OSError("clipboard busy")
        return True

    def EmptyClipboard(self):
        self.emptied = True
        self.events.append("empty")
        return True

    def SetClipboardData(self, _fmt, _handle):
        self.events.append("set")
        return True

    def CloseClipboard(self):
        self.events.append("close")
        return True


def test_empty_clipboard_is_restored_by_clearing_it(monkeypatch: pytest.MonkeyPatch) -> None:
    # The user's clipboard was empty before dictation: "restore" means the
    # dictated text must not stay there.
    user32 = _ClipboardUser32()
    monkeypatch.setattr(insertion, "_user32", lambda: user32)
    monkeypatch.setattr(insertion, "_kernel32", lambda: object())

    assert insertion._restore_clipboard([]) is True
    assert user32.emptied is True


def test_clipboard_open_is_retried_after_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    user32 = _ClipboardUser32(failures=2)
    monkeypatch.setattr(insertion, "_user32", lambda: user32)
    monkeypatch.setattr(insertion, "_kernel32", lambda: object())
    monkeypatch.setattr(insertion.time, "sleep", lambda _value: None)

    assert insertion._restore_clipboard([]) is True
    assert user32.opened == 3
