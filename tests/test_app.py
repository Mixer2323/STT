"""Application-level recording state transitions."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from voicetyper.audio import RecordingResult
from voicetyper.config import ConfigStore
from voicetyper.history import HistoryStore
from voicetyper.insertion import InsertionResult
from voicetyper.pipeline import PipelineResult


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


def test_audio_overflow_is_reported(app, monkeypatch) -> None:
    messages: list[str] = []
    monkeypatch.setattr(
        app.notifier,
        "notify",
        lambda title, message, **kwargs: messages.append(message) or True,
    )
    app._recording = _Session(RecordingResult(overflowed=True))

    app.stop_recording()

    assert any("переполняло буфер" in message for message in messages)


def test_capture_failure_is_reported(app, monkeypatch, tmp_path) -> None:
    messages: list[str] = []
    monkeypatch.setattr(
        app.notifier,
        "notify",
        lambda title, message, **kwargs: messages.append(message) or True,
    )
    path = tmp_path / "rec.wav"
    path.write_bytes(b"audio")
    app._recording = _Session(
        RecordingResult(
            path=path, duration=1.5, capture_error="устройство отключено"
        )
    )

    app.stop_recording()

    assert any("Захват звука прервался" in message for message in messages)
    assert any("устройство отключено" in message for message in messages)


def test_apply_config_reports_a_save_failure(app, monkeypatch) -> None:
    from voicetyper import config as config_module

    def boom(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(config_module, "_atomic_json_write", boom)
    ok, message = app.apply_config(app.config_store.get())

    assert ok is True
    assert "не удалось сохранить" in message.lower()


def test_focus_restore_failure_is_reported(app, monkeypatch) -> None:
    messages: list[str] = []
    monkeypatch.setattr(
        app.notifier,
        "notify",
        lambda title, message, **kwargs: messages.append(message) or True,
    )
    app._on_pipeline_result(
        PipelineResult(
            job_id=1,
            text="привет",
            insertion=InsertionResult(
                method="paste", inserted=True, target_focused=False
            ),
        )
    )

    assert any("фокус" in message for message in messages)


def test_unicode_fallback_is_reported_for_scancode_typing(app, monkeypatch) -> None:
    messages: list[str] = []
    monkeypatch.setattr(
        app.notifier,
        "notify",
        lambda title, message, **kwargs: messages.append(message) or True,
    )
    app._on_pipeline_result(
        PipelineResult(
            job_id=1,
            text="привет 😀",
            insertion=InsertionResult(
                method="type_scancode", inserted=True, unicode_fallback=2
            ),
        )
    )

    assert any("Unicode" in message for message in messages)


def test_closed_target_window_is_reported(app, monkeypatch) -> None:
    messages: list[str] = []
    monkeypatch.setattr(
        app.notifier,
        "notify",
        lambda title, message, **kwargs: messages.append(message) or True,
    )

    app._on_pipeline_result(
        PipelineResult(
            job_id=1,
            text="привет",
            insertion=InsertionResult(
                method="paste", inserted=True, target_missing=True
            ),
        )
    )

    assert any("закрылось" in message for message in messages)


def test_reset_hotkey_clears_a_stuck_latch(app, monkeypatch) -> None:
    monkeypatch.setattr(app.notifier, "notify", lambda *args, **kwargs: True)
    app.hotkeys._latch = True

    app.reset_hotkey()

    assert app.hotkeys._latch is False


def test_failed_startup_shuts_the_app_down(app, monkeypatch) -> None:
    import voicetyper.app as app_module

    fatal: list[str] = []
    monkeypatch.setattr(app_module, "_fatal", lambda message: fatal.append(message))
    monkeypatch.setattr(
        app_module,
        "TranscriptionPipeline",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    monkeypatch.setattr(app.notifier, "notify", lambda *args, **kwargs: True)

    app._on_tray_ready(None)

    assert app.is_shutting_down
    assert fatal and "boom" in fatal[0]


class _FakeEngine:
    def transcribe(self, path, language=None):
        from voicetyper.engines.base import TranscriptionResult

        return TranscriptionResult(text="привет   мир", language=language or "ru")


class _FakeLease:
    def __init__(self) -> None:
        self.name = "fake"
        self.engine = _FakeEngine()
        self.released = False

    def release(self) -> None:
        self.released = True


class _FakeModels:
    def __init__(self, _store):
        self.shutdown_called = False
        self.requested = 0

    def request(self, **_kwargs) -> None:
        self.requested += 1

    def acquire(self, timeout: float = 0.0) -> _FakeLease:
        return _FakeLease()

    def shutdown(self, timeout: float = 0.0) -> None:
        self.shutdown_called = True


def test_transcribe_cli_writes_text(tmp_path, monkeypatch, capsys, data_dir) -> None:
    import logging

    import voicetyper.app as app_module

    monkeypatch.setattr(app_module, "setup_logging", lambda **_kwargs: logging.getLogger("x"))
    monkeypatch.setattr(app_module, "ModelManager", _FakeModels)
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"fake")

    code = app_module.main(
        ["--transcribe", str(audio), "--out-dir", str(tmp_path / "out")]
    )

    assert code == 0
    assert (tmp_path / "out" / "audio.txt").read_text(encoding="utf-8").strip() == "Привет мир"
    assert "Привет мир" in capsys.readouterr().out


def test_transcribe_cli_reports_a_missing_file(tmp_path, monkeypatch, capsys, data_dir) -> None:
    import logging

    import voicetyper.app as app_module

    monkeypatch.setattr(app_module, "setup_logging", lambda **_kwargs: logging.getLogger("x"))

    code = app_module.main(["--transcribe", str(tmp_path / "nope.wav")])

    assert code == 2
    assert "не найден" in capsys.readouterr().err


def test_headless_run_quits_when_a_later_window_closes(app, monkeypatch) -> None:
    # Without a tray, a settings window opened through the second-instance event
    # must remain a way out of the app.
    app.tray._available = False
    monkeypatch.setattr(app, "_on_tray_ready", lambda icon: None)

    class _FakeSettings:
        is_open = False

        def close(self) -> None:
            pass

    fake = _FakeSettings()
    app.settings = fake
    thread = threading.Thread(target=app.run, daemon=True)
    thread.start()
    try:
        time.sleep(0.35)
        fake.is_open = True
        time.sleep(0.4)
        fake.is_open = False
        thread.join(5)
    finally:
        app.shutdown()
        thread.join(5)

    assert not thread.is_alive()


def test_model_is_loaded_on_demand_after_idle_unload(app, monkeypatch) -> None:
    requested: list[bool] = []
    fake = SimpleNamespace(
        ready=False,
        loading=False,
        request=lambda *_args, **_kwargs: requested.append(True),
    )
    monkeypatch.setattr(app, "models", fake)

    app._ensure_model_loaded()
    assert requested

    fake.ready = True
    app._ensure_model_loaded()
    assert len(requested) == 1


def test_old_model_cache_cleanup_requires_confirmation(app, monkeypatch) -> None:
    import voicetyper.app as app_module

    monkeypatch.setattr(app_module, "find_orphan_model_caches", lambda: [Path("old")])
    monkeypatch.setattr(app_module, "orphan_model_cache_size", lambda _caches: 1024)
    removed: list[bool] = []
    monkeypatch.setattr(
        app_module,
        "cleanup_orphan_model_caches",
        lambda: removed.append(True) or (1, 1024),
    )
    messages: list[str] = []
    monkeypatch.setattr(
        app.notifier,
        "notify",
        lambda title, message, **kwargs: messages.append(message) or True,
    )

    monkeypatch.setattr(app, "_confirm", lambda _message, _title: False)
    app.cleanup_old_model_caches()
    assert not removed
    assert messages == []

    monkeypatch.setattr(app, "_confirm", lambda _message, _title: True)
    app.cleanup_old_model_caches()
    assert removed
    assert any("освобождено" in message for message in messages)


def test_headless_run_keeps_running_without_a_tray(app, monkeypatch) -> None:
    # Without pystray and without an open settings window the old code called
    # shutdown() immediately, so the hotkey never became available.
    app.tray._available = False
    started: list[object] = []
    monkeypatch.setattr(app, "_on_tray_ready", lambda icon: started.append(icon))

    thread = threading.Thread(target=app.run, daemon=True)
    thread.start()
    try:
        time.sleep(0.4)
        assert started == [None]
        assert thread.is_alive()
    finally:
        app.shutdown()
        thread.join(5)
    assert not thread.is_alive()


def test_hotkey_is_registered_before_the_hardware_probe(app, monkeypatch) -> None:
    # Probing the GPU can take seconds on a cold start (nvidia-smi, ctranslate2);
    # the dictation hotkey must work before that.
    order: list[str] = []

    class FakeHotkeys:
        available = True
        bound_hotkey = ""

        def start(self) -> None:
            order.append("hotkey")

        def close(self) -> None:
            pass

    class FakeModels:
        def request(self, **_kwargs) -> None:
            order.append("model")

    monkeypatch.setattr(app, "hotkeys", FakeHotkeys())
    monkeypatch.setattr(app, "models", FakeModels())
    monkeypatch.setattr(app.tray, "set_state", lambda *a, **k: None)
    monkeypatch.setattr(app.notifier, "notify", lambda *a, **k: None)
    monkeypatch.setattr(app, "_sync_autostart", lambda cfg: None)
    monkeypatch.setattr(app, "_warm_up_microphone", lambda: None)
    app.config_store.update(warmup_microphone=False)

    app._on_tray_ready(None)

    assert order == ["hotkey", "model"]
