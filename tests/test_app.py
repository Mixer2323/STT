"""Application-level recording state transitions."""

from __future__ import annotations

from pathlib import Path

import pytest

from voicetyper.audio import RecordingResult
from voicetyper.config import ConfigStore
from voicetyper.history import HistoryStore


@pytest.fixture()
def app(tmp_path: Path):
    from voicetyper.app import VoiceTyperApp

    store = ConfigStore(tmp_path / "config.json")
    store.load()
    history = HistoryStore(tmp_path / "history.db")
    instance = VoiceTyperApp(store, history, None)
    yield instance
    instance._cleanup()


def _captured_states(app, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    states: list[str] = []
    monkeypatch.setattr(
        app.tray,
        "set_state",
        lambda state, message="", **_kwargs: states.append(state),
    )
    return states


class _Session:
    def __init__(self, result: RecordingResult) -> None:
        self._result = result

    def stop(self) -> RecordingResult:
        return self._result


def test_silent_recording_clears_the_recording_indicator(app, monkeypatch) -> None:
    states = _captured_states(app, monkeypatch)
    app._recording = _Session(
        RecordingResult(error="Тишина — распознавание пропущено", silent=True)
    )

    app.stop_recording()

    assert app._recording is None
    assert states[-1] == app._model_status.state
    assert states[-1] != "recording"


def test_recording_without_a_file_clears_the_recording_indicator(app, monkeypatch) -> None:
    states = _captured_states(app, monkeypatch)
    app._recording = _Session(RecordingResult())

    app.stop_recording()

    assert app._recording is None
    assert states[-1] == app._model_status.state


def test_replacement_microphone_is_reported(app, monkeypatch) -> None:
    messages: list[str] = []
    monkeypatch.setattr(
        app.notifier,
        "notify",
        lambda title, message, **_kwargs: messages.append(message) or True,
    )

    class _Replaced:
        def capture_mismatch(self):
            return "Головной телефон (1MORE SonoFlow)"

    app._warn_about_capture_device(_Replaced())

    assert messages
    assert "Головной телефон (1MORE SonoFlow)" in messages[0]


def test_matching_microphone_is_not_reported(app, monkeypatch) -> None:
    messages: list[str] = []
    monkeypatch.setattr(
        app.notifier,
        "notify",
        lambda title, message, **_kwargs: messages.append(message) or True,
    )

    class _Matching:
        def capture_mismatch(self):
            return None

        server_side_fallback = False

    app._warn_about_capture_device(_Matching())

    assert messages == []


def test_server_side_microphone_is_reported(app, monkeypatch) -> None:
    messages: list[tuple[object, str]] = []
    monkeypatch.setattr(
        app.notifier,
        "notify",
        lambda title, message, **kwargs: messages.append(
            (kwargs.get("key"), message)
        )
        or True,
    )

    class _ServerSide:
        server_side_fallback = True
        device_name = "FrontMic (Realtek HD Audio)"

        def capture_mismatch(self):
            return None

    app._warn_about_capture_device(_ServerSide())

    assert messages
    assert messages[0][0] == "mic-server"
    assert "Перенаправленный микрофон RDP" in messages[0][1]


def test_warm_up_does_not_change_the_recording_state(app, monkeypatch) -> None:
    from voicetyper import app as app_module

    states = _captured_states(app, monkeypatch)
    warmed: list[object] = []
    monkeypatch.setattr(
        app_module, "warm_up_microphone", lambda cfg: warmed.append(cfg) or True
    )

    app._warm_up_microphone()

    assert warmed
    # The tray must stay idle: the warm-up is not a recording.
    assert states == []


def test_warm_up_is_skipped_while_recording(app, monkeypatch) -> None:
    from voicetyper import app as app_module

    warmed: list[object] = []
    monkeypatch.setattr(app_module, "warm_up_microphone", lambda cfg: warmed.append(cfg))
    app._recording = _Session(RecordingResult())

    app._warm_up_microphone()

    assert warmed == []
