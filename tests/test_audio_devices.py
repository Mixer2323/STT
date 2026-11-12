"""Capture tests: one microphone — the selected one or the Windows default."""

from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np
import pytest

from voicetyper import audio
from voicetyper.config import Config


@pytest.fixture(autouse=True)
def _isolate_audio(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the tests away from real hardware and reset the MF failure cache."""
    monkeypatch.setattr(audio, "HAS_SOUNDCARD", False)
    monkeypatch.setattr(audio, "HAS_MEDIA_FOUNDATION", False)
    audio._reset_capture_cache()


class FakeStream:
    def __init__(self, device, samplerate, channels):
        self.device = device
        self.samplerate = samplerate
        self.channels = channels
        self.started = False
        self.closed = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.started = False

    def close(self) -> None:
        self.closed = True


class FakeSoundDevice:
    """Minimal sounddevice stand-in: one default endpoint and a device list."""

    def __init__(
        self,
        devices,
        default_index,
        *,
        wasapi_default=-1,
        failing=None,
        failing_devices=None,
    ):
        self._devices = devices
        self.default = SimpleNamespace(device=[default_index, -1])
        self.wasapi_default = wasapi_default
        self.failing = failing or set()
        self.failing_devices = failing_devices or set()
        self.opened: list[tuple[object, int, int]] = []
        self.refreshed = 0

    def query_hostapis(self):
        return [
            {"name": "MME", "default_input_device": self.default.device[0]},
            {"name": "Windows DirectSound", "default_input_device": -1},
            {"name": "Windows WASAPI", "default_input_device": self.wasapi_default},
            {"name": "Windows WDM-KS", "default_input_device": -1},
        ]

    def query_devices(self, device=None, kind=None):
        if device is None and kind is None:
            size = max((item["index"] for item in self._devices), default=-1) + 1
            result = []
            for index in range(size):
                match = next(
                    (item for item in self._devices if item["index"] == index), None
                )
                result.append(
                    match
                    if match is not None
                    else {
                        "name": f"unused-{index}",
                        "max_input_channels": 0,
                        "default_samplerate": 44100.0,
                        "hostapi": 0,
                    }
                )
            return result
        index = self.default.device[0] if device is None else int(device)
        for item in self._devices:
            if item["index"] == index:
                return item
        raise ValueError(f"Invalid device: {device}")

    def InputStream(
        self,
        *,
        device,
        samplerate,
        channels,
        dtype,
        blocksize,
        callback,
        extra_settings=None,
    ):
        self.opened.append((device, samplerate, channels))
        if device in self.failing_devices:
            raise RuntimeError("Error opening InputStream: MME error 11")
        if samplerate in self.failing:
            raise RuntimeError("Error opening InputStream: Invalid sample rate")
        return FakeStream(device, samplerate, channels)

    def _terminate(self):
        self.refreshed += 1

    def _initialize(self):
        pass


def _device(
    index: int,
    name: str,
    *,
    channels: int = 1,
    rate: float = 44100.0,
    hostapi: int = 0,
) -> dict:
    return {
        "index": index,
        "name": name,
        "max_input_channels": channels,
        "default_samplerate": rate,
        "hostapi": hostapi,
    }


def _use(monkeypatch: pytest.MonkeyPatch, fake: FakeSoundDevice) -> FakeSoundDevice:
    monkeypatch.setattr(audio, "sd", fake)
    monkeypatch.setattr(audio, "HAS_SOUNDDEVICE", True)
    return fake


def _machine() -> list[dict]:
    return [
        _device(1, "Микрофон (USB PnP)", hostapi=0, channels=1, rate=44100.0),
        _device(2, "Микрофон (USB PnP)", hostapi=2, channels=2, rate=48000.0),
        _device(3, "Гарнитура (1MORE)", hostapi=2, channels=1, rate=16000.0),
        _device(4, "Стерео микшер", hostapi=0, channels=2, rate=48000.0),
        _device(9, "Микрофон (Realtek)", hostapi=3, channels=2, rate=44100.0),
    ]


def test_windows_default_endpoint_is_used(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _use(monkeypatch, FakeSoundDevice(_machine(), 1, wasapi_default=2))

    session = audio.RecordingSession(Config())
    session.start()
    try:
        assert session._device_index == 2
        assert fake.opened == [(2, 48000, 2)]
    finally:
        session.stop()


def test_manual_device_is_used(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _use(monkeypatch, FakeSoundDevice(_machine(), 1, wasapi_default=2))

    session = audio.RecordingSession(Config(input_device=3))
    session.start()
    try:
        assert session._device_index == 3
        assert fake.opened[0][0] == 3
    finally:
        session.stop()


def test_missing_manual_device_falls_back_to_windows_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _use(monkeypatch, FakeSoundDevice(_machine(), 1, wasapi_default=2))

    session = audio.RecordingSession(Config(input_device=99))
    session.start()
    try:
        assert session._device_index == 2
        assert fake.opened[0][0] == 2
    finally:
        session.stop()


def test_device_switched_in_windows_is_followed(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _use(monkeypatch, FakeSoundDevice(_machine(), 1, wasapi_default=2))

    first = audio.RecordingSession(Config())
    first.start()
    first.stop()
    assert first._device_index == 2

    # The user plugs a headset and Windows makes it the default endpoint.
    fake.wasapi_default = 3
    second = audio.RecordingSession(Config())
    second.start()
    try:
        assert second._device_index == 3
    finally:
        second.stop()


def test_device_list_is_refreshed_before_every_recording(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _use(monkeypatch, FakeSoundDevice(_machine(), 1, wasapi_default=2))

    for _attempt in range(2):
        session = audio.RecordingSession(Config())
        session.start()
        session.stop()

    assert fake.refreshed == 2


def test_only_the_target_device_is_tried(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _use(
        monkeypatch,
        FakeSoundDevice(
            _machine(), 1, wasapi_default=2, failing_devices={2, 1, 3, 4, 9}
        ),
    )

    with pytest.raises(audio.AudioError) as failure:
        audio.RecordingSession(Config()).start()

    assert {attempt[0] for attempt in fake.opened} == {2}
    assert "Микрофон «Микрофон (USB PnP)»" in str(failure.value)
    assert "занято другим приложением" in str(failure.value)


def test_format_ladder_prefers_the_device_rate(monkeypatch: pytest.MonkeyPatch) -> None:
    _use(monkeypatch, FakeSoundDevice(_machine(), 1, wasapi_default=2))

    mme = audio._device_formats(1, 16000)
    assert mme[0] == (44100, 1)
    assert (16000, 1) in mme
    wasapi = audio._device_formats(2, 16000)
    assert wasapi[0] == (48000, 2)
    assert (48000, 1) in wasapi
    mono = audio._device_formats(3, 16000)
    assert all(channels == 1 for _rate, channels in mono)


def test_visible_devices_collapse_the_host_api_duplicates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _use(monkeypatch, FakeSoundDevice(_machine(), 1, wasapi_default=2))

    assert [device.index for device in audio.visible_input_devices()] == [2, 3]
    assert [
        device.index for device in audio.visible_input_devices(include_kernel=True)
    ] == [2, 3, 9]


def test_visible_devices_hide_mixers_and_the_sound_mapper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    devices_data = [
        _device(0, "Microsoft Sound Mapper - Input", hostapi=0),
        _device(1, "Переназначение звуковых устр-", hostapi=0),
        _device(2, "Микрофон (USB Headset)", hostapi=0),
        _device(3, "Стерео микшер", hostapi=0),
        _device(4, "Микрофон (USB Headset)", hostapi=1),
        _device(8, "Первичный драйвер записи звука", hostapi=2),
    ]
    _use(monkeypatch, FakeSoundDevice(devices_data, default_index=0, wasapi_default=4))

    assert [device.index for device in audio.visible_input_devices()] == [4]


def test_device_names_stay_on_one_line(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeSoundDevice(
        [_device(1, "Вход (bthhfenum.sys,#4;%1 Hands-Free%0\r\n;(Phone))", hostapi=1)],
        default_index=1,
        wasapi_default=1,
    )
    _use(monkeypatch, fake)

    assert audio.visible_input_devices()[0].name == (
        "Вход (bthhfenum.sys,#4;%1 Hands-Free%0 ;(Phone))"
    )


def test_capture_mismatch_reports_a_replacement_microphone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The chosen headset cannot be opened, so the Windows default is used.
    _use(
        monkeypatch,
        FakeSoundDevice(_machine(), 1, wasapi_default=2, failing_devices={3}),
    )

    session = audio.RecordingSession(Config(input_device=3))
    session.start()
    try:
        assert session._device_index == 2
        assert session.capture_mismatch() == "Микрофон (USB PnP)"
    finally:
        session.stop()


def test_capture_mismatch_is_none_for_the_default_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _use(monkeypatch, FakeSoundDevice(_machine(), 1, wasapi_default=2))

    session = audio.RecordingSession(Config())
    session.start()
    try:
        assert session.capture_mismatch() is None
    finally:
        session.stop()


def test_server_side_fallback_flags_a_local_microphone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _use(monkeypatch, FakeSoundDevice(_machine(), 1, wasapi_default=2))
    monkeypatch.setattr(audio, "is_remote_session", lambda: True)

    local = audio.RecordingSession(Config(input_device=3))
    local.start()
    try:
        assert local.server_side_fallback
    finally:
        local.stop()


def test_redirected_endpoint_is_not_a_server_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    devices = [_device(2, "Удаленное аудио", hostapi=1, channels=1, rate=44100.0)]
    fake = FakeSoundDevice(devices, default_index=2, wasapi_default=2)
    _use(monkeypatch, fake)
    monkeypatch.setattr(audio, "is_remote_session", lambda: True)

    session = audio.RecordingSession(Config())
    session.start()
    try:
        assert not session.server_side_fallback
    finally:
        session.stop()


def test_warm_up_opens_and_discards_the_default_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _use(monkeypatch, FakeSoundDevice(_machine(), 1, wasapi_default=2))
    monkeypatch.setattr(
        audio, "save_wav", lambda *_args, **_kwargs: pytest.fail("warm-up wrote a file")
    )

    assert audio.warm_up_microphone(Config(), settle=0)
    assert [attempt[0] for attempt in fake.opened] == [2]


def test_warm_up_reports_an_unavailable_microphone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _use(
        monkeypatch,
        FakeSoundDevice(_machine(), 1, wasapi_default=2, failing_devices={2}),
    )

    assert audio.warm_up_microphone(Config(), settle=0) is False


class FakeRecorder:
    samplerate = 16000
    sample_format = "pcm16"
    opened_with = "формат устройства"
    attempts: list[BaseException] = []
    _mix_channels = 1

    def __enter__(self) -> "FakeRecorder":
        return self

    def __exit__(self, *_args) -> bool:
        return False

    def record(self, numframes: int) -> np.ndarray:
        time.sleep(0.001)
        return np.zeros((numframes, 1), dtype=np.float32)


class FakeMicrophone:
    name = "Удалённое аудио"


def test_media_foundation_is_not_used_outside_rdp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _use(
        monkeypatch,
        FakeSoundDevice(_machine(), 1, wasapi_default=2, failing_devices={2}),
    )
    monkeypatch.setattr(audio, "HAS_SOUNDCARD", True)
    monkeypatch.setattr(audio, "HAS_MEDIA_FOUNDATION", True)
    monkeypatch.setattr(audio, "is_remote_session", lambda: False)
    probed = []
    monkeypatch.setattr(
        audio, "_find_soundcard_microphone", lambda names: probed.append(names)
    )

    with pytest.raises(audio.AudioError):
        audio.RecordingSession(Config()).start()

    assert probed == []


def test_media_foundation_captures_the_same_endpoint_in_rdp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    devices = [_device(2, "Удалённое аудио", hostapi=1, channels=1, rate=44100.0)]
    fake = FakeSoundDevice(devices, default_index=2, wasapi_default=2, failing_devices={2})
    _use(monkeypatch, fake)
    monkeypatch.setattr(audio, "HAS_SOUNDCARD", True)
    monkeypatch.setattr(audio, "HAS_MEDIA_FOUNDATION", True)
    monkeypatch.setattr(audio, "is_remote_session", lambda: True)
    monkeypatch.setattr(
        audio, "_find_soundcard_microphone", lambda names: FakeMicrophone()
    )
    monkeypatch.setattr(
        audio,
        "open_recorder",
        lambda microphone, seconds=0.1, preferred_rate=16000: FakeRecorder(),
    )

    session = audio.RecordingSession(Config())
    session.start()
    try:
        assert session.is_recording
        assert session._native_mic_name == "Удалённое аудио"
    finally:
        session.stop()


def test_media_foundation_failure_is_remembered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    devices = [_device(2, "Удалённое аудио", hostapi=1, channels=1, rate=44100.0)]
    fake = FakeSoundDevice(devices, default_index=2, wasapi_default=2, failing_devices={2})
    _use(monkeypatch, fake)
    monkeypatch.setattr(audio, "HAS_SOUNDCARD", True)
    monkeypatch.setattr(audio, "HAS_MEDIA_FOUNDATION", True)
    monkeypatch.setattr(audio, "is_remote_session", lambda: True)
    calls = []
    monkeypatch.setattr(
        audio, "_find_soundcard_microphone", lambda names: calls.append(names)
    )

    with pytest.raises(audio.AudioError):
        audio.RecordingSession(Config()).start()
    assert audio._mf_failed_until > time.monotonic()

    with pytest.raises(audio.AudioError):
        audio.RecordingSession(Config()).start()
    assert len(calls) == 1  # the second attempt is skipped by the negative cache


def test_probe_input_devices_reports_every_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _use(monkeypatch, FakeSoundDevice(_machine(), 1, wasapi_default=2))
    monkeypatch.setattr(audio, "HAS_SOUNDCARD", False)
    monkeypatch.setattr(audio, "HAS_MEDIA_FOUNDATION", False)
    monkeypatch.setattr(
        audio, "_capture_env", lambda: SimpleNamespace(blockers=[], report_lines=lambda: [])
    )

    report = audio.probe_input_devices(Config())

    assert "Микрофон (USB PnP)" in report
    assert "Будет использован:" in report
    assert "== Вывод ==" in report


def test_verdict_blames_the_client_when_the_endpoint_exists_but_refuses() -> None:
    lines = audio._verdict_lines(
        is_rdp=True,
        opened=["[14] Микрофон (Logi USB Headset H340)"],
        failed_redirect=["[9] Удаленное аудио"],
        local_opened=["[14] Микрофон (Logi USB Headset H340)"],
        redirect_seen=True,
        mf_text="Initialize [формат устройства]: HRESULT 0x80070057 (E_INVALIDARG)",
        blockers=[],
    )

    report = "\n".join(lines)
    assert "IAudioClient::Initialize возвращает E_INVALIDARG" in report
    assert "Проверять нужно клиент, а не сервер" in report


def test_verdict_warns_when_the_redirect_endpoint_is_absent() -> None:
    lines = audio._verdict_lines(
        is_rdp=True,
        opened=["[14] Микрофон (Logi USB Headset H340)"],
        failed_redirect=[],
        local_opened=["[14] Микрофон (Logi USB Headset H340)"],
        redirect_seen=False,
        mf_text="",
        blockers=[],
    )

    report = "\n".join(lines)
    assert "Перенаправленного микрофона RDP в сеансе нет" in report
    assert "не подставляет автоматически" in report
