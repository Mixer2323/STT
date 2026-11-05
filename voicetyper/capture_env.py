"""Windows settings that can make every microphone unopenable.

Read by ``VoiceTyper.exe --audio-test``: an RDP session often enumerates the
redirected endpoint but refuses to activate it. The refusal is normally caused
by audio-capture redirection policy, the microphone privacy consent store or a
stopped Windows Audio service — none of which the device list shows.
"""

from __future__ import annotations

import ctypes
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

TERMINAL_SERVICES_POLICY_KEY = r"SOFTWARE\Policies\Microsoft\Windows NT\Terminal Services"
WINSTATIONS_KEY = r"SYSTEM\CurrentControlSet\Control\Terminal Server\WinStations"
VOLATILE_ENVIRONMENT_KEY = r"Volatile Environment"
MICROPHONE_CONSENT_KEY = (
    r"SOFTWARE\Microsoft\Windows\CurrentVersion"
    r"\CapabilityAccessManager\ConsentStore\microphone"
)
# The ADMX policy "Allow audio recording redirection" writes fDisableAudioCapture
# (0 = allowed); the unprefixed names are kept because older templates use them.
CAPTURE_POLICY_VALUES = (
    "fDisableAudioCapture",
    "fDisableAudioPlayback",
    "DisableAudioCapture",
    "DisableAudioPlayback",
)
STATION_VALUES = ("fDisableAudioCapture", "fDisableAudioPlayback", "AudioEnumeratorDll")
AUDIO_SERVICES = ("Audiosrv", "AudioEndpointBuilder")

_SERVICE_STATES = {
    1: "остановлена",
    2: "запускается",
    3: "останавливается",
    4: "запущена",
    5: "возобновляется",
    6: "ставится на паузу",
    7: "на паузе",
}
_RUNNING = 4


@dataclass
class CaptureEnvironment:
    """Snapshot of the capture-related Windows configuration."""

    session_name: str = ""
    client_name: str = ""
    station: str = ""
    is_remote: bool = False
    policies: list[tuple[str, str, str]] = field(default_factory=list)
    station_settings: list[tuple[str, str, str]] = field(default_factory=list)
    negotiated: list[tuple[str, str, str]] = field(default_factory=list)
    privacy: list[tuple[str, str, str]] = field(default_factory=list)
    services: list[tuple[str, str]] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)

    def report_lines(self) -> list[str]:
        lines = ["== Ограничения записи Windows =="]
        kind = "RDP / удалённый сеанс" if self.is_remote else "локальный сеанс"
        suffix = f", клиент: {self.client_name}" if self.client_name else ""
        lines.append(f"  сеанс: {self.session_name or '?'} ({kind}{suffix})")
        lines.extend(self._section("политика RDP", self.policies))
        if self.is_remote:
            label = f"станция {self.station}" if self.station else "настройки станции RDP"
            lines.extend(self._section(label, self.station_settings))
            lines.extend(self._section("согласовано с клиентом", self.negotiated))
        lines.extend(self._section("доступ к микрофону", self.privacy))
        if self.services:
            lines.append("  службы:")
            for name, state in self.services:
                lines.append(f"    {name}: {state}")
        else:
            lines.append("  службы: состояние не определено")
        if self.blockers:
            for blocker in self.blockers:
                lines.append(f"  ВНИМАНИЕ: {blocker}")
        else:
            lines.append("  ограничений, которые мешают записи, не найдено")
        return lines

    @staticmethod
    def _section(label: str, entries) -> list[str]:
        if not entries:
            return [f"  {label}: значения не заданы"]
        lines = [f"  {label}:"]
        for source, name, value in entries:
            lines.append(f"    {source}\\{name} = {value}")
        return lines


def _reg_value(hive, path: str, name: str) -> Optional[str]:
    try:
        import winreg

        with winreg.OpenKey(hive, path) as key:
            value, _kind = winreg.QueryValueEx(key, name)
        return str(value)
    except FileNotFoundError:
        return None
    except OSError as exc:
        logger.debug("Реестр %s\\%s (%s): %s", getattr(hive, "name", hive), path, name, exc)
        return None


def _reg_values(hive, path: str) -> dict[str, str]:
    try:
        import winreg

        with winreg.OpenKey(hive, path) as key:
            out: dict[str, str] = {}
            for index in range(winreg.QueryInfoKey(key)[1]):
                name, value, _kind = winreg.EnumValue(key, index)
                out[str(name)] = str(value)
            return out
    except OSError:
        return {}


def _reg_subkeys(hive, path: str) -> list[str]:
    try:
        import winreg

        with winreg.OpenKey(hive, path) as key:
            return [winreg.EnumKey(key, index) for index in range(winreg.QueryInfoKey(key)[0])]
    except OSError:
        return []


def _hives() -> list[tuple[str, object]]:
    try:
        import winreg
    except ImportError:
        return []
    return [("HKLM", winreg.HKEY_LOCAL_MACHINE), ("HKCU", winreg.HKEY_CURRENT_USER)]


def _local_machine():
    for name, hive in _hives():
        if name == "HKLM":
            return hive
    return None


def _collect_policies() -> list[tuple[str, str, str]]:
    found: list[tuple[str, str, str]] = []
    for hive_name, hive in _hives():
        for name in CAPTURE_POLICY_VALUES:
            value = _reg_value(hive, TERMINAL_SERVICES_POLICY_KEY, name)
            if value is not None:
                found.append((hive_name, name, value))
    return found


def _station_name(session_name: str) -> str:
    """WinStations subkey for the current session: RDP-Tcp#32 -> RDP-Tcp."""
    if not session_name or session_name.strip().lower() == "console":
        return ""
    return session_name.split("#", 1)[0].strip()


def _collect_station(station: str) -> list[tuple[str, str, str]]:
    hive = _local_machine()
    if hive is None or not station:
        return []
    values = _reg_values(hive, rf"{WINSTATIONS_KEY}\{station}")
    return [
        (f"HKLM\\{station}", name, values[name])
        for name in STATION_VALUES
        if name in values
    ]


def _collect_negotiated() -> list[tuple[str, str, str]]:
    """CLIENTAUDIO* values the session host stored from the client's request."""
    found: list[tuple[str, str, str]] = []
    for hive_name, hive in _hives():
        for subkey in _reg_subkeys(hive, VOLATILE_ENVIRONMENT_KEY):
            for name, value in _reg_values(hive, rf"{VOLATILE_ENVIRONMENT_KEY}\{subkey}").items():
                if name.upper().startswith("CLIENTAUDIO"):
                    found.append((f"{hive_name}\\{subkey}", name, value))
    return found


def _collect_privacy() -> list[tuple[str, str, str]]:
    found: list[tuple[str, str, str]] = []
    for hive_name, hive in _hives():
        for suffix, label in (("", "microphone"), (r"\NonPackaged", "microphone\\NonPackaged")):
            value = _reg_value(hive, MICROPHONE_CONSENT_KEY + suffix, "Value")
            if value is not None:
                found.append((hive_name, label, value))
    return found


def _is_windows() -> bool:
    return os.name == "nt"


def _collect_services() -> list[tuple[str, str]]:
    if not _is_windows():
        return []
    try:
        return _query_service_states(AUDIO_SERVICES)
    except Exception as exc:
        logger.debug("Не удалось опросить службы Windows: %s", exc)
        return []


def _query_service_states(names) -> list[tuple[str, str]]:
    class ServiceStatus(ctypes.Structure):
        _fields_ = [
            ("dwServiceType", ctypes.c_uint32),
            ("dwCurrentState", ctypes.c_uint32),
            ("dwControlsAccepted", ctypes.c_uint32),
            ("dwWin32ExitCode", ctypes.c_uint32),
            ("dwServiceSpecificExitCode", ctypes.c_uint32),
            ("dwCheckPoint", ctypes.c_uint32),
            ("dwWaitHint", ctypes.c_uint32),
        ]

    advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
    advapi32.OpenSCManagerW.restype = ctypes.c_void_p
    advapi32.OpenSCManagerW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_uint32]
    advapi32.OpenServiceW.restype = ctypes.c_void_p
    advapi32.OpenServiceW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_uint32]
    advapi32.QueryServiceStatus.restype = ctypes.c_int32
    advapi32.QueryServiceStatus.argtypes = [ctypes.c_void_p, ctypes.POINTER(ServiceStatus)]
    advapi32.CloseServiceHandle.restype = ctypes.c_int32
    advapi32.CloseServiceHandle.argtypes = [ctypes.c_void_p]

    manager = advapi32.OpenSCManagerW(None, None, 0x0001)  # SC_MANAGER_CONNECT
    if not manager:
        raise ctypes.WinError(ctypes.get_last_error())
    states: list[tuple[str, str]] = []
    try:
        for name in names:
            service = advapi32.OpenServiceW(manager, name, 0x0004)  # SERVICE_QUERY_STATUS
            if not service:
                states.append((name, f"нет доступа (код {ctypes.get_last_error()})"))
                continue
            try:
                status = ServiceStatus()
                if not advapi32.QueryServiceStatus(service, ctypes.byref(status)):
                    states.append((name, "состояние недоступно"))
                else:
                    states.append(
                        (name, _SERVICE_STATES.get(status.dwCurrentState, "неизвестно"))
                    )
            finally:
                advapi32.CloseServiceHandle(service)
    finally:
        advapi32.CloseServiceHandle(manager)
    return states


def _is_remote_session() -> bool:
    if os.environ.get("SESSIONNAME", "").strip().upper().startswith("RDP-"):
        return True
    if not _is_windows():
        return False
    try:
        user32 = ctypes.windll.user32
        return bool(user32.GetSystemMetrics(0x1000))  # SM_REMOTESESSION
    except Exception:
        return False


def _find_blockers(env: CaptureEnvironment) -> list[str]:
    blockers: list[str] = []
    for source, name, value in list(env.policies) + list(env.station_settings):
        if name.lower().endswith("audiocapture") and value.strip() == "1":
            blockers.append(
                f"{source}\\{name} = 1 запрещает перенаправление записи в RDP. "
                "Микрофон клиента в сеанс не поступает: на сервере включите "
                "«Разрешить переадресацию аудиозаписи» (Components → Remote Desktop "
                "Session Host → Device and Resource Redirection)."
            )
    if env.is_remote and env.station_settings and not any(
        name == "AudioEnumeratorDll" for _source, name, _value in env.station_settings
    ):
        blockers.append(
            f"на станции {env.station} не задан AudioEnumeratorDll — сервер не "
            "создаёт перенаправленные аудиоконечные точки. В компонентах Windows "
            "нужен «Пользовательские интерфейсы и инфраструктура» "
            "(User Interfaces and Infrastructure), служба Windows Audio должна "
            "работать в сеансе."
        )
    for hive, label, value in env.privacy:
        if value.strip().casefold() != "deny":
            continue
        if label.endswith("NonPackaged"):
            blockers.append(
                f"доступ к микрофону для приложений рабочего стола запрещён "
                f"({hive} microphone\\NonPackaged = Deny). Параметры Windows → "
                "Конфиденциальность → Микрофон → «Разрешить приложениям рабочего стола "
                "доступ к микрофону»."
            )
        else:
            blockers.append(
                f"доступ к микрофону системно запрещён ({hive} microphone = Deny). "
                "Параметры Windows → Конфиденциальность → Микрофон."
            )
    for name, state in env.services:
        if name == "Audiosrv" and state != _SERVICE_STATES[_RUNNING]:
            blockers.append(
                f"служба Windows Audio ({name}) не запущена: {state}. "
                "Запустите её в services.msc — без неё не открывается ни одно устройство."
            )
    return blockers


def collect() -> CaptureEnvironment:
    """Read the capture environment; on non-Windows systems return an empty snapshot."""
    env = CaptureEnvironment(
        session_name=os.environ.get("SESSIONNAME", ""),
        client_name=os.environ.get("CLIENTNAME", ""),
        is_remote=_is_remote_session(),
    )
    if not _is_windows():
        return env
    env.station = _station_name(env.session_name) if env.is_remote else ""
    env.policies = _collect_policies()
    env.station_settings = _collect_station(env.station)
    env.negotiated = _collect_negotiated()
    env.privacy = _collect_privacy()
    env.services = _collect_services()
    env.blockers = _find_blockers(env)
    return env


def report_lines() -> list[str]:
    try:
        return collect().report_lines()
    except Exception as exc:  # the report must still be produced
        logger.debug("Опрос окружения записи не удался", exc_info=True)
        return ["== Ограничения записи Windows ==", f"  ошибка опроса: {exc}"]


def summary_lines(env: Optional[CaptureEnvironment] = None) -> list[str]:
    """Compact one-line-per-check version for the diagnostics report."""
    env = env or collect()
    lines = [
        "Политика RDP: "
        + (", ".join(f"{hive} {name}={value}" for hive, name, value in env.policies)
           or "не задана"),
        "Станция RDP: "
        + (
            ", ".join(
                f"{name}={value}" for _source, name, value in env.station_settings
            )
            or "не задана"
        ),
        "Доступ к микрофону: "
        + (", ".join(f"{hive} {label}={value}" for hive, label, value in env.privacy)
           or "не задан"),
        "Службы звука: "
        + (", ".join(f"{name}={state}" for name, state in env.services)
           or "не определены"),
    ]
    lines.extend(f"  ВНИМАНИЕ: {blocker}" for blocker in env.blockers)
    return lines
