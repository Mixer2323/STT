from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np
import pytest

from voicetyper import audio
from voicetyper.config import Config


@pytest.fixture(autouse=True)
def _disable_soundcard(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the PortAudio tests away from real Media Foundation hardware."""
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
    """Minimal sounddevice stand-in for the device/open retry logic."""

    def __init__(
        self,
        devices,
        default_index,
        *,
        wasapi_default=None,
        failing=None,
        failing_devices=None,
        shared_fail_devices=None,
        fail_until_refresh=None,
    ):
        self._devices = devices
        self.default = SimpleNamespace(device=[default_index, -1])
        self.wasapi_default = wasapi_default
        self.failing = failing or set()
        self.failing_devices = failing_devices or set()
        self.shared_fail_devices = shared_fail_devices or set()
        self.fail_until_refresh = fail_until_refresh
        self.opened: list[tuple[object, int, int]] = []
        self.exclusive_opened: list[tuple[object, int, int]] = []
        self.refreshed = 0

    def query_hostapis(self):
        return [
            {"name": "MME", "default_input_device": self.default.device[0]},
            {
                "name": "Windows WASAPI",
                "default_input_device": (
                    self.wasapi_default if self.wasapi_default is not None else -1
                ),
            },
            {"name": "Windows DirectSound", "default_input_device": -1},
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
        exclusive = extra_settings is not None
        if exclusive:
            self.exclusive_opened.append((device, samplerate, channels))
        else:
            self.opened.append((device, samplerate, channels))
        if self.fail_until_refresh is not None and self.refreshed == 0:
            raise RuntimeError("Error opening InputStream: MME error 11")
        if not self._devices:
            raise RuntimeError("Error opening InputStream: no input device")
        if device in self.failing_devices:
            raise RuntimeError("Error opening InputStream: MME error 11")
        if exclusive and device in self.shared_fail_devices:
            return FakeStream(device, samplerate, channels)
        if not exclusive and device in self.shared_fail_devices:
            raise RuntimeError("Error opening InputStream: shared mode refused")
        if samplerate in self.failing:
            raise RuntimeError("Error opening InputStream: MME error 11")
        return FakeStream(device, samplerate, channels)

    def _terminate(self):
        self.refreshed += 1

    def _initialize(self):
        pass

    @staticmethod
    def WasapiSettings(exclusive=False):
        return SimpleNamespace(exclusive=exclusive)


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


@pytest.fixture()
def fake_sd(monkeypatch: pytest.MonkeyPatch) -> FakeSoundDevice:
    fake = FakeSoundDevice([_device(1, "Микрофон (2- USB PnP Audio Device)")], 1)
    monkeypatch.setattr(audio, "sd", fake)
    monkeypatch.setattr(audio, "HAS_SOUNDDEVICE", True)
    return fake


def test_candidate_formats_prefer_native_rate(fake_sd: FakeSoundDevice) -> None:
    fake_sd._devices = [_device(1, "Микрофон", channels=2, rate=48000.0)]

    formats = audio.RecordingSession(Config())._candidate_formats(1)

    assert formats[0] == (48000, 1)
    assert formats[1] == (48000, 2)
    assert (16000, 1) in formats


def _field_machine() -> list[dict]:
    """Devices of the reported RDP machine: the session host APIs only carry the
    redirected endpoint, the physical microphones show up through WDM-KS."""
    return [
        _device(1, "Удаленное аудио", channels=2, hostapi=0),
        _device(5, "Удаленное аудио", channels=2, hostapi=2),
        _device(9, "Удаленное аудио", channels=2, hostapi=1),
        _device(10, "FrontMic (Realtek HD Audio)", channels=2, hostapi=3),
        _device(11, "Лин. вход (Realtek HD Audio Line input)", channels=2, hostapi=3),
        _device(12, "Стерео микшер", channels=2, hostapi=3),
        _device(14, "Микрофон (Logi USB Headset H340)", channels=2, hostapi=3),
    ]


def test_kernel_devices_are_left_out_of_the_queue_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeSoundDevice(_field_machine(), 1, wasapi_default=9)
    monkeypatch.setattr(audio, "sd", fake)
    monkeypatch.setattr(audio, "HAS_SOUNDDEVICE", True)

    candidates = audio.RecordingSession(Config())._candidate_devices(1)

    assert candidates == [9, 1, 5]


def test_kernel_devices_join_the_queue_only_when_the_headset_is_local(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeSoundDevice(_field_machine(), 1, wasapi_default=9)
    monkeypatch.setattr(audio, "sd", fake)
    monkeypatch.setattr(audio, "HAS_SOUNDDEVICE", True)

    candidates = audio.RecordingSession(
        Config(server_microphone=True)
    )._candidate_devices(1)

    assert candidates[:3] == [9, 1, 5]
    assert candidates.index(10) < candidates.index(14) < candidates.index(11)
    assert 12 not in candidates


def test_start_refuses_the_server_mic_in_an_rdp_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeSoundDevice(_field_machine(), 1, wasapi_default=9)
    fake.failing_devices = {1, 5, 9}
    monkeypatch.setattr(audio, "sd", fake)
    monkeypatch.setattr(audio, "HAS_SOUNDDEVICE", True)
    monkeypatch.setattr(audio, "is_remote_session", lambda: True)

    with pytest.raises(audio.AudioError):
        audio.RecordingSession(Config()).start()


def test_start_records_from_the_server_mic_when_the_option_is_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeSoundDevice(_field_machine(), 1, wasapi_default=9)
    fake.failing_devices = {1, 5, 9}
    monkeypatch.setattr(audio, "sd", fake)
    monkeypatch.setattr(audio, "HAS_SOUNDDEVICE", True)
    monkeypatch.setattr(audio, "is_remote_session", lambda: True)

    session = audio.RecordingSession(Config(server_microphone=True))
    session.start()

    assert session.is_recording
    assert fake.opened[-1][0] in (10, 14)
    session.stop()


def test_visible_devices_collapse_the_host_api_duplicates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeSoundDevice(_field_machine(), 1, wasapi_default=9)
    monkeypatch.setattr(audio, "sd", fake)
    monkeypatch.setattr(audio, "HAS_SOUNDDEVICE", True)

    devices = audio.visible_input_devices()

    assert [device.index for device in devices] == [9]
    assert [device.index for device in audio.visible_input_devices(include_kernel=True)] == [
        9,
        10,
        11,
        14,
    ]


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
    fake = FakeSoundDevice(devices_data, default_index=0, wasapi_default=4)
    monkeypatch.setattr(audio, "sd", fake)
    monkeypatch.setattr(audio, "HAS_SOUNDDEVICE", True)

    assert [device.index for device in audio.visible_input_devices()] == [4]


def test_device_names_stay_on_one_line(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeSoundDevice(
        [_device(1, "Вход (bthhfenum.sys,#4;%1 Hands-Free%0\r\n;(Phone))", hostapi=1)],
        default_index=1,
        wasapi_default=1,
    )
    monkeypatch.setattr(audio, "sd", fake)
    monkeypatch.setattr(audio, "HAS_SOUNDDEVICE", True)

    assert audio.visible_input_devices()[0].name == (
        "Вход (bthhfenum.sys,#4;%1 Hands-Free%0 ;(Phone))"
    )


def test_server_side_device_is_not_reported_as_fallback_locally(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeSoundDevice(_field_machine(), 1, wasapi_default=9)
    monkeypatch.setattr(audio, "sd", fake)
    monkeypatch.setattr(audio, "HAS_SOUNDDEVICE", True)
    monkeypatch.setattr(audio, "is_remote_session", lambda: True)

    assert audio._is_server_side_fallback(14)
    assert not audio._is_server_side_fallback(9)
    monkeypatch.setattr(audio, "is_remote_session", lambda: False)
    assert not audio._is_server_side_fallback(14)


def test_verdict_blames_the_client_when_the_endpoint_exists_but_refuses() -> None:
    lines = audio._verdict_lines(
        is_rdp=True,
        opened=["[14] Микрофон (Logi USB Headset H340)"],
        failed_redirect=["[9] Удаленное аудио"],
        local_opened=["[14] Микрофон (Logi USB Headset H340)"],
        redirect_seen=True,
        mf_text="Initialize [формат устройства]: HRESULT 0x80070057 (E_INVALIDARG)",
        blockers=[],
        server_microphone=False,
    )

    report = "\n".join(lines)
    assert "IAudioClient::Initialize возвращает E_INVALIDARG" in report
    assert "Проверять нужно клиент, а не сервер" in report
    assert "приложение их не открывает" in report


def test_verdict_warns_when_the_local_only_devices_are_enabled() -> None:
    lines = audio._verdict_lines(
        is_rdp=True,
        opened=["[14] Микрофон (Logi USB Headset H340)"],
        failed_redirect=["[9] Удаленное аудио"],
        local_opened=["[14] Микрофон (Logi USB Headset H340)"],
        redirect_seen=True,
        mf_text="",
        blockers=[],
        server_microphone=True,
    )

    assert "запись будет молчать" in "\n".join(lines)


def test_verdict_names_absent_redirect_as_client_request() -> None:
    lines = audio._verdict_lines(
        is_rdp=True,
        opened=["[10] FrontMic (Realtek HD Audio)"],
        failed_redirect=[],
        local_opened=["[10] FrontMic (Realtek HD Audio)"],
        redirect_seen=False,
        mf_text="",
        blockers=[],
        server_microphone=False,
    )

    report = "\n".join(lines)
    assert "Перенаправленного микрофона RDP в сеансе нет" in report
    assert "audiocapturemode:i:1" in report


def test_start_falls_back_when_mme_rejects_rate(fake_sd: FakeSoundDevice) -> None:
    fake_sd.failing = {16000, 48000, 44100}
    session = audio.RecordingSession(Config())

    session.start()

    assert session.capture_rate == 8000
    assert fake_sd.opened[-1][1] == 8000
    session.stop()


def test_start_refreshes_devices_before_giving_up(fake_sd: FakeSoundDevice) -> None:
    fake_sd.fail_until_refresh = True
    session = audio.RecordingSession(Config())

    session.start()

    assert fake_sd.refreshed == 1
    assert session.is_recording
    session.stop()


def test_missing_index_falls_back_to_default(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeSoundDevice([_device(2, "Перенаправление звука в RDP")], 2)
    monkeypatch.setattr(audio, "sd", fake)
    monkeypatch.setattr(audio, "HAS_SOUNDDEVICE", True)

    assert audio.resolve_input_device(99) == 2


def test_no_devices_reports_rdp_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeSoundDevice([], -1)
    monkeypatch.setattr(audio, "sd", fake)
    monkeypatch.setattr(audio, "HAS_SOUNDDEVICE", True)
    monkeypatch.setattr(audio, "is_remote_session", lambda: True)

    with pytest.raises(audio.AudioError) as excinfo:
        audio.RecordingSession(Config()).start()

    message = str(excinfo.value)
    assert "Микрофоны не найдены" in message
    assert "RDP" in message


def test_open_failure_prefers_the_windows_blocker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        audio,
        "_capture_blockers",
        lambda: ["доступ к микрофону запрещён (HKCU microphone\\NonPackaged = Deny)."],
    )

    message = audio._open_microphone_message(RuntimeError("MME error 11"))

    assert "Не удалось открыть микрофон" in message
    assert "доступ к микрофону запрещён" in message


def test_open_failure_falls_back_to_the_rdp_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(audio, "_capture_blockers", lambda: [])
    monkeypatch.setattr(audio, "is_remote_session", lambda: True)

    message = audio._open_microphone_message(RuntimeError("MME error 11"))

    assert "Запись с этого компьютера" in message


def test_wasapi_is_preferred_for_default_device(monkeypatch: pytest.MonkeyPatch) -> None:
    devices = [
        _device(1, "Удалённое аудио", hostapi=0),
        _device(2, "Удалённое аудио", hostapi=1, channels=1, rate=48000.0),
    ]
    fake = FakeSoundDevice(
        devices, default_index=1, wasapi_default=2, failing_devices={1}
    )
    monkeypatch.setattr(audio, "sd", fake)
    monkeypatch.setattr(audio, "HAS_SOUNDDEVICE", True)

    session = audio.RecordingSession(Config())
    session.start()

    assert fake.opened[0] == (2, 48000, 2)
    assert session.capture_rate == 48000
    assert session._capture_channels == 2
    assert session._stream.device == 2
    session.stop()


def test_explicit_device_falls_back_to_wasapi_equivalent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    devices = [
        _device(1, "Микрофон (USB PnP Audio Device)", hostapi=0),
        _device(2, "Микрофон (USB PnP Audio Device)", hostapi=1, rate=48000.0),
    ]
    fake = FakeSoundDevice(
        devices, default_index=1, wasapi_default=2, failing_devices={1}
    )
    monkeypatch.setattr(audio, "sd", fake)
    monkeypatch.setattr(audio, "HAS_SOUNDDEVICE", True)

    session = audio.RecordingSession(Config(input_device=1))
    session.start()

    assert fake.opened[0][0] == 1
    assert fake.opened[-1][0] == 2
    assert session._stream.device == 2
    session.stop()


def test_wasapi_tries_stereo_before_mono(monkeypatch: pytest.MonkeyPatch) -> None:
    devices = [_device(2, "Удалённое аудио", hostapi=1, channels=1, rate=48000.0)]
    fake = FakeSoundDevice(devices, default_index=2, wasapi_default=2)
    monkeypatch.setattr(audio, "sd", fake)
    monkeypatch.setattr(audio, "HAS_SOUNDDEVICE", True)

    session = audio.RecordingSession(Config())
    session.start()

    assert fake.opened[0] == (2, 48000, 2)
    session.stop()


def test_wasapi_exclusive_mode_is_the_last_resort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    devices = [_device(2, "Удалённое аудио", hostapi=1, channels=1, rate=48000.0)]
    fake = FakeSoundDevice(
        devices,
        default_index=2,
        wasapi_default=2,
        shared_fail_devices={2},
    )
    monkeypatch.setattr(audio, "sd", fake)
    monkeypatch.setattr(audio, "HAS_SOUNDDEVICE", True)

    session = audio.RecordingSession(Config())
    session.start()

    assert fake.exclusive_opened
    assert fake.exclusive_opened[0][0] == 2
    assert session._stream.device == 2
    session.stop()


def test_all_wasapi_devices_are_tried(monkeypatch: pytest.MonkeyPatch) -> None:
    devices = [
        _device(1, "Микрофон (USB)", hostapi=0),
        _device(2, "Микрофон (USB)", hostapi=1),
        _device(3, "Удалённое аудио", hostapi=1),
    ]
    fake = FakeSoundDevice(
        devices, default_index=1, wasapi_default=2, failing_devices={1, 2}
    )
    monkeypatch.setattr(audio, "sd", fake)
    monkeypatch.setattr(audio, "HAS_SOUNDDEVICE", True)

    session = audio.RecordingSession(Config())
    session.start()

    assert session._stream.device == 3
    session.stop()


class _StubEnv:
    """Stand-in for capture_env.CaptureEnvironment in report tests."""

    def __init__(self, blockers=()) -> None:
        self.blockers = list(blockers)

    def report_lines(self) -> list[str]:
        return ["== Ограничения записи Windows =="]


def test_probe_input_devices_reports_success(monkeypatch: pytest.MonkeyPatch) -> None:
    devices = [_device(2, "Удалённое аудио", hostapi=1, channels=1, rate=48000.0)]
    fake = FakeSoundDevice(devices, default_index=2, wasapi_default=2)
    monkeypatch.setattr(audio, "sd", fake)
    monkeypatch.setattr(audio, "HAS_SOUNDDEVICE", True)
    monkeypatch.setattr(audio, "_soundcard_microphones", lambda: [])
    monkeypatch.setattr(audio, "_capture_env", lambda: _StubEnv())

    report = audio.probe_input_devices(Config())

    assert "Voice Typer" in report
    assert "[2] Удалённое аудио" in report
    assert "shared: OK" in report
    assert "== Ограничения записи Windows ==" in report
    assert "== Вывод ==" in report


def test_probe_input_devices_lists_every_refused_format(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    devices = [_device(9, "Удалённое аудио", hostapi=1, rate=44100.0)]
    fake = FakeSoundDevice(devices, default_index=9, wasapi_default=9)
    monkeypatch.setattr(audio, "sd", fake)
    monkeypatch.setattr(audio, "HAS_SOUNDDEVICE", True)
    monkeypatch.setattr(audio, "_soundcard_microphones", lambda: [])
    monkeypatch.setattr(audio, "_capture_env", lambda: _StubEnv())

    def refusing_open(self, device, rate, channels, *, exclusive=False):
        if rate in (44100, 48000):
            raise RuntimeError("Unanticipated host error [MME error 11]")
        raise RuntimeError("Invalid sample rate [PaErrorCode -9997]")

    monkeypatch.setattr(audio.RecordingSession, "_open_stream", refusing_open)

    report = audio.probe_input_devices(Config())

    assert "shared: не удалось — 8 форматов:" in report
    assert "44100 Гц/2 кан., 44100 Гц/1 кан." in report
    assert "Unanticipated host error [MME error 11]" in report
    assert "Invalid sample rate" in report
    assert "ИТОГ: устройство не открывается" in report


def test_probe_explains_a_blocked_rdp_redirect(monkeypatch: pytest.MonkeyPatch) -> None:
    devices = [
        _device(9, "Удаленное аудио", hostapi=1, rate=44100.0),
        _device(1, "Микрофон (Logi USB Headset)", hostapi=0, rate=44100.0),
    ]
    fake = FakeSoundDevice(devices, default_index=1, wasapi_default=9, failing_devices={9})
    monkeypatch.setattr(audio, "sd", fake)
    monkeypatch.setattr(audio, "HAS_SOUNDDEVICE", True)
    monkeypatch.setattr(audio, "is_remote_session", lambda: True)
    monkeypatch.setattr(audio, "_capture_env", lambda: _StubEnv())
    monkeypatch.setattr(audio, "HAS_MEDIA_FOUNDATION", True)
    monkeypatch.setattr(audio, "_soundcard_microphones", lambda: [FakeMicrophone()])
    monkeypatch.setattr(
        audio,
        "probe_microphone",
        lambda microphone, seconds=0.2: [
            "  Удаленное аудио:",
            "    формат устройства: 44100 Гц, 2 кан.",
            "    захват: ошибка: Initialize [формат устройства, буфер 0]: "
            "HRESULT 0x80070057 (E_INVALIDARG)",
        ],
    )

    report = audio.probe_input_devices(Config())

    assert "Перенаправленный микрофон RDP не открывается" in report
    assert "IAudioClient::Initialize возвращает E_INVALIDARG" in report
    assert "audiocapturemode:i:1" in report
    assert "== Вывод ==" in report


def test_probe_points_at_windows_blockers(monkeypatch: pytest.MonkeyPatch) -> None:
    devices = [_device(9, "Удаленное аудио", hostapi=1, rate=44100.0)]
    fake = FakeSoundDevice(devices, default_index=9, wasapi_default=9, failing_devices={9})
    monkeypatch.setattr(audio, "sd", fake)
    monkeypatch.setattr(audio, "HAS_SOUNDDEVICE", True)
    monkeypatch.setattr(audio, "_soundcard_microphones", lambda: [])
    monkeypatch.setattr(
        audio, "_capture_env", lambda: _StubEnv(["микрофон запрещён политикой."])
    )

    report = audio.probe_input_devices(Config())

    assert "Причина в ограничениях Windows" in report
    assert "Если ни одно устройство не открылось" not in report


class FakeRecorder:
    samplerate = 16000
    sample_format = "float32"
    _mix_channels = 1
    opened_with = "тестовый формат"
    attempts: list[BaseException] = []

    def __enter__(self) -> "FakeRecorder":
        return self

    def __exit__(self, *args) -> bool:
        return False

    def record(self, numframes: int) -> np.ndarray:
        time.sleep(0.005)
        return np.zeros((numframes, 1), dtype=np.float32)


class FakeMicrophone:
    name = "Удалённое аудио"

    def recorder(self, *, samplerate: int, channels: int, blocksize: int) -> FakeRecorder:
        return FakeRecorder()


def test_media_foundation_fallback_when_portaudio_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    devices = [_device(9, "Удалённое аудио", hostapi=1, channels=1, rate=44100.0)]
    fake = FakeSoundDevice(
        devices, default_index=9, wasapi_default=9, failing_devices={9}
    )
    monkeypatch.setattr(audio, "sd", fake)
    monkeypatch.setattr(audio, "HAS_SOUNDDEVICE", True)
    monkeypatch.setattr(audio, "HAS_SOUNDCARD", True)
    monkeypatch.setattr(audio, "HAS_MEDIA_FOUNDATION", True)
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

    assert session.is_recording
    assert session._native_thread is not None
    assert session._native_mic_name == "Удалённое аудио"
    session.stop()


def test_last_working_capture_configuration_is_tried_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeSoundDevice(_field_machine(), 1, wasapi_default=9)
    monkeypatch.setattr(audio, "sd", fake)
    monkeypatch.setattr(audio, "HAS_SOUNDDEVICE", True)

    first = audio.RecordingSession(Config())
    first.start()
    opened = fake.opened[-1]
    first.stop()

    fake.opened.clear()
    second = audio.RecordingSession(Config())
    second.start()
    second.stop()

    # The remembered (device, rate, channels) is opened immediately instead of
    # walking the rejected candidates again.
    assert fake.opened[0] == opened


def test_remembered_configuration_is_ignored_for_another_microphone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeSoundDevice(_field_machine(), 1, wasapi_default=9)
    monkeypatch.setattr(audio, "sd", fake)
    monkeypatch.setattr(audio, "HAS_SOUNDDEVICE", True)

    first = audio.RecordingSession(Config())
    first.start()
    first.stop()

    fake.opened.clear()
    other = audio.RecordingSession(Config(input_device=14, server_microphone=True))
    other.start()
    other.stop()

    assert fake.opened[0][0] != 9


def test_expired_capture_configuration_is_probed_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeSoundDevice(_field_machine(), 1, wasapi_default=9)
    monkeypatch.setattr(audio, "sd", fake)
    monkeypatch.setattr(audio, "HAS_SOUNDDEVICE", True)

    # A remembered mono configuration that has aged out must not be used: the
    # normal search opens the first candidate (stereo) instead.
    audio._remember_capture(None, 9, 44100, 1, False)
    stale = audio._last_capture
    audio._last_capture = (stale[0] - audio._CAPTURE_CACHE_SECONDS - 1, *stale[1:])

    session = audio.RecordingSession(Config())
    session.start()
    try:
        assert fake.opened[0] == (9, 44100, 2)
    finally:
        session.stop()


def test_capture_mismatch_reports_a_replacement_microphone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The configured device cannot be opened, so another one is used instead.
    fake = FakeSoundDevice(_field_machine(), 1, wasapi_default=9)
    fake.failing_devices = {10}
    monkeypatch.setattr(audio, "sd", fake)
    monkeypatch.setattr(audio, "HAS_SOUNDDEVICE", True)

    session = audio.RecordingSession(Config(input_device=10, server_microphone=True))
    session.start()
    try:
        assert session.capture_mismatch() == session._device_name
    finally:
        session.stop()


def test_capture_mismatch_accepts_the_configured_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeSoundDevice(_field_machine(), 1, wasapi_default=9)
    monkeypatch.setattr(audio, "sd", fake)
    monkeypatch.setattr(audio, "HAS_SOUNDDEVICE", True)

    session = audio.RecordingSession(Config(input_device=9))
    session.start()
    try:
        assert session.capture_mismatch() is None
    finally:
        session.stop()


def test_capture_mismatch_accepts_the_wasapi_twin(monkeypatch: pytest.MonkeyPatch) -> None:
    # The same physical microphone opened through its WASAPI entry is not a
    # fallback to something else.
    devices = [
        _device(1, "Микрофон (USB PnP Audio Device)", hostapi=0),
        _device(2, "Микрофон (USB PnP Audio Device)", hostapi=1, rate=48000.0),
    ]
    fake = FakeSoundDevice(devices, default_index=1, wasapi_default=2, failing_devices={1})
    monkeypatch.setattr(audio, "sd", fake)
    monkeypatch.setattr(audio, "HAS_SOUNDDEVICE", True)

    session = audio.RecordingSession(Config(input_device=1))
    session.start()
    try:
        assert session._stream.device == 2
        assert session.capture_mismatch() is None
    finally:
        session.stop()
