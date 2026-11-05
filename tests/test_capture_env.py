from __future__ import annotations

import pytest

from voicetyper import capture_env as ce


@pytest.fixture()
def windows(monkeypatch: pytest.MonkeyPatch):
    """Pretend to be Windows with a controllable registry and service manager."""
    state: dict[str, object] = {
        "registry": {},
        "services": {},
        "windows": True,
    }

    def fake_hives():
        if not state["windows"]:
            return []
        return [("HKLM", "HKLM"), ("HKCU", "HKCU")]

    def fake_reg_value(hive, path, name):
        if not state["windows"]:
            raise AssertionError("registry must not be read outside Windows")
        return state["registry"].get((hive, path, name))

    def fake_reg_values(hive, path):
        if not state["windows"]:
            raise AssertionError("registry must not be read outside Windows")
        prefix = (hive, path)
        return {
            name: value
            for (h, p, name), value in state["registry"].items()
            if (h, p) == prefix
        }

    def fake_states(names):
        return [(name, state["services"][name]) for name in names if name in state["services"]]

    monkeypatch.setattr(ce, "_is_windows", lambda: bool(state["windows"]))
    monkeypatch.setattr(ce, "_hives", fake_hives)
    monkeypatch.setattr(ce, "_reg_value", fake_reg_value)
    monkeypatch.setattr(ce, "_reg_values", fake_reg_values)
    monkeypatch.setattr(ce, "_query_service_states", fake_states)
    monkeypatch.setattr(ce, "_is_remote_session", lambda: True)
    monkeypatch.setenv("SESSIONNAME", "RDP-Tcp#32")
    monkeypatch.setenv("CLIENTNAME", "PC")
    return state


def _policy(state, hive, name, value):
    state["registry"][(hive, ce.TERMINAL_SERVICES_POLICY_KEY, name)] = value


def _station(state, name, value):
    path = rf"{ce.WINSTATIONS_KEY}\RDP-Tcp"
    state["registry"][("HKLM", path, name)] = value


def _consent(state, hive, label, value):
    path = ce.MICROPHONE_CONSENT_KEY + (r"\NonPackaged" if "NonPackaged" in label else "")
    state["registry"][(hive, path, "Value")] = value


def _running(state: dict) -> None:
    state["services"] = {"Audiosrv": "запущена", "AudioEndpointBuilder": "запущена"}


def test_the_policy_value_name_is_the_documented_one(windows) -> None:
    _policy(windows, "HKLM", "fDisableAudioCapture", "1")
    _station(windows, "fDisableAudioCapture", "0")
    _station(windows, "AudioEnumeratorDll", "rdpendp.dll")
    _running(windows)

    env = ce.collect()

    assert env.policies == [("HKLM", "fDisableAudioCapture", "1")]
    assert env.station == "RDP-Tcp"
    assert len(env.blockers) == 1
    assert "HKLM\\fDisableAudioCapture = 1" in env.blockers[0]
    assert "Разрешить переадресацию аудиозаписи" in env.blockers[0]


def test_station_settings_are_read_for_the_current_station(windows) -> None:
    _station(windows, "fDisableAudioCapture", "1")
    _station(windows, "AudioEnumeratorDll", "rdpendp.dll")
    _running(windows)

    env = ce.collect()
    report = "\n".join(env.report_lines())

    assert env.station_settings == [
        ("HKLM\\RDP-Tcp", "fDisableAudioCapture", "1"),
        ("HKLM\\RDP-Tcp", "AudioEnumeratorDll", "rdpendp.dll"),
    ]
    assert "станция RDP-Tcp:" in report
    assert "AudioEnumeratorDll = rdpendp.dll" in report
    assert "ВНИМАНИЕ:" in report


def test_microphone_privacy_deny_for_desktop_apps(windows) -> None:
    _station(windows, "AudioEnumeratorDll", "rdpendp.dll")
    _consent(windows, "HKCU", "microphone\\NonPackaged", "Deny")
    _running(windows)

    env = ce.collect()

    assert len(env.blockers) == 1
    assert "приложений рабочего стола" in env.blockers[0]


def test_stopped_audio_service_is_a_blocker(windows) -> None:
    windows["services"] = {"Audiosrv": "остановлена", "AudioEndpointBuilder": "запущена"}

    report = "\n".join(ce.report_lines())

    assert "Audiosrv: остановлена" in report
    assert "Windows Audio" in report


def test_healthy_environment_says_so(windows) -> None:
    _station(windows, "fDisableAudioCapture", "0")
    _station(windows, "AudioEnumeratorDll", "rdpendp.dll")
    _consent(windows, "HKLM", "microphone", "Allow")
    _running(windows)

    report = "\n".join(ce.report_lines())

    assert "ограничений, которые мешают записи, не найдено" in report
    assert "ВНИМАНИЕ" not in report


def test_service_state_codes_match_the_windows_api() -> None:
    assert ce._RUNNING == 4
    assert ce._SERVICE_STATES[ce._RUNNING] == "запущена"
    assert ce._SERVICE_STATES[1] == "остановлена"


def test_console_session_has_no_station(windows, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ce, "_is_remote_session", lambda: False)
    monkeypatch.setenv("SESSIONNAME", "Console")
    _running(windows)

    env = ce.collect()
    report = "\n".join(env.report_lines())

    assert env.station == ""
    assert env.station_settings == []
    assert "станция" not in report


def test_non_windows_collects_nothing(windows) -> None:
    windows["windows"] = False

    env = ce.collect()

    assert env.policies == []
    assert env.station_settings == []
    assert env.privacy == []
    assert env.services == []


def test_station_name_helper() -> None:
    assert ce._station_name("RDP-Tcp#32") == "RDP-Tcp"
    assert ce._station_name("Console") == ""
    assert ce._station_name("") == ""


def test_summary_lines_are_compact(windows) -> None:
    _policy(windows, "HKCU", "fDisableAudioPlayback", "1")
    _station(windows, "AudioEnumeratorDll", "rdpendp.dll")
    windows["services"] = {"Audiosrv": "запущена"}

    lines = ce.summary_lines()

    assert lines[0] == "Политика RDP: HKCU fDisableAudioPlayback=1"
    assert lines[1] == "Станция RDP: AudioEnumeratorDll=rdpendp.dll"
    assert lines[2] == "Доступ к микрофону: не задан"
    assert lines[3] == "Службы звука: Audiosrv=запущена"
