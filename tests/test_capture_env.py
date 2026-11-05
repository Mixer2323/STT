from __future__ import annotations

import pytest

from voicetyper import capture_env as ce


@pytest.fixture()
def windows(monkeypatch: pytest.MonkeyPatch):
    """Pretend to be Windows with a controllable registry and service manager."""
    state: dict[str, object] = {"registry": {}, "services": {}, "windows": True}

    def fake_hives():
        if not state["windows"]:
            return []
        return [("HKLM", "HKLM"), ("HKCU", "HKCU")]

    def fake_reg_value(hive, path, name):
        if not state["windows"]:
            raise AssertionError("registry must not be read outside Windows")
        return state["registry"].get((hive, path, name))

    def fake_states(names):
        return [(name, state["services"][name]) for name in names if name in state["services"]]

    monkeypatch.setattr(ce, "_is_windows", lambda: bool(state["windows"]))
    monkeypatch.setattr(ce, "_hives", fake_hives)
    monkeypatch.setattr(ce, "_reg_value", fake_reg_value)
    monkeypatch.setattr(ce, "_query_service_states", fake_states)
    monkeypatch.setattr(ce, "_is_remote_session", lambda: True)
    monkeypatch.setenv("SESSIONNAME", "RDP-Tcp#0")
    monkeypatch.setenv("CLIENTNAME", "OFFICE-PC")
    return state


def _policy(state, hive, name, value):
    state["registry"][(hive, ce.TERMINAL_SERVICES_POLICY_KEY, name)] = value


def _consent(state, hive, label, value):
    path = ce.MICROPHONE_CONSENT_KEY + (r"\NonPackaged" if "NonPackaged" in label else "")
    state["registry"][(hive, path, "Value")] = value


def test_collect_reads_policy_privacy_and_services(windows) -> None:
    _policy(windows, "HKLM", "DisableAudioCapture", "0")
    _consent(windows, "HKLM", "microphone", "Allow")
    _consent(windows, "HKLM", "microphone\\NonPackaged", "Allow")
    windows["services"] = {"Audiosrv": "запущена", "AudioEndpointBuilder": "запущена"}

    env = ce.collect()

    assert env.is_remote
    assert env.session_name == "RDP-Tcp#0"
    assert env.client_name == "OFFICE-PC"
    assert env.policies == [("HKLM", "DisableAudioCapture", "0")]
    assert env.privacy == [
        ("HKLM", "microphone", "Allow"),
        ("HKLM", "microphone\\NonPackaged", "Allow"),
    ]
    assert env.services == [("Audiosrv", "запущена"), ("AudioEndpointBuilder", "запущена")]
    assert env.blockers == []


def test_capture_redirection_policy_is_reported_as_a_blocker(windows) -> None:
    _policy(windows, "HKLM", "DisableAudioCapture", "1")
    windows["services"] = {"Audiosrv": "запущена", "AudioEndpointBuilder": "запущена"}

    report = "\n".join(ce.report_lines())

    assert "HKLM\\DisableAudioCapture = 1" in report
    assert "ВНИМАНИЕ:" in report
    assert "DisableAudioCapture=1" in report


def test_microphone_privacy_deny_for_desktop_apps(windows) -> None:
    _consent(windows, "HKCU", "microphone\\NonPackaged", "Deny")
    windows["services"] = {"Audiosrv": "запущена", "AudioEndpointBuilder": "запущена"}

    env = ce.collect()

    assert len(env.blockers) == 1
    assert "приложений рабочего стола" in env.blockers[0]


def test_stopped_audio_service_is_a_blocker(windows) -> None:
    windows["services"] = {"Audiosrv": "остановлена", "AudioEndpointBuilder": "запущена"}

    report = "\n".join(ce.report_lines())

    assert "Audiosrv: остановлена" in report
    assert "Windows Audio" in report


def test_service_state_codes_match_the_windows_api() -> None:
    assert ce._RUNNING == 4
    assert ce._SERVICE_STATES[ce._RUNNING] == "запущена"
    assert ce._SERVICE_STATES[1] == "остановлена"


def test_healthy_environment_says_so(windows) -> None:
    windows["services"] = {"Audiosrv": "запущена", "AudioEndpointBuilder": "запущена"}

    report = "\n".join(ce.report_lines())

    assert "ограничений, которые мешают записи, не найдено" in report
    assert "ВНИМАНИЕ" not in report


def test_non_windows_collects_nothing(windows) -> None:
    windows["windows"] = False

    env = ce.collect()

    assert env.policies == []
    assert env.privacy == []
    assert env.services == []


def test_summary_lines_are_compact(windows) -> None:
    _policy(windows, "HKCU", "DisableAudioPlayback", "1")
    windows["services"] = {"Audiosrv": "запущена"}

    lines = ce.summary_lines()

    assert lines[0] == "Политика RDP: HKCU DisableAudioPlayback=1"
    assert lines[1] == "Доступ к микрофону: не задан"
    assert lines[2] == "Службы звука: Audiosrv=запущена"
