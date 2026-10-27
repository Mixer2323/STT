"""Hardware detection and automatic runtime recommendations (no torch required)."""

from __future__ import annotations

import ctypes
import importlib.metadata
import importlib.util
import logging
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Optional

from .config import Config

logger = logging.getLogger(__name__)

_CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0


@dataclass(frozen=True)
class HardwareInfo:
    cpu_cores: int = 1
    ram_gb: Optional[float] = None
    cuda: bool = False
    vram_gb: Optional[float] = None
    gpu_name: Optional[str] = None
    free_disk_gb: Optional[float] = None
    machine: str = ""

    def summary(self) -> str:
        gpu = "нет"
        if self.cuda:
            gpu = self.gpu_name or "NVIDIA"
            if self.vram_gb:
                gpu += f" / {self.vram_gb:.1f} ГБ"
        ram = f"{self.ram_gb:.1f} ГБ" if self.ram_gb else "?"
        disk = f"{self.free_disk_gb:.1f} ГБ" if self.free_disk_gb else "?"
        return (
            f"{self.machine or platform.system()} | CPU: {self.cpu_cores} потоков | "
            f"RAM: {ram} | GPU: {gpu} | свободно: {disk}"
        )


@dataclass(frozen=True)
class EngineAvailability:
    installed: bool
    version: Optional[str] = None
    reason: Optional[str] = None


def package_version(*names: str) -> Optional[str]:
    for name in names:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
        except Exception:
            continue
    return None


def module_installed(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def engine_availability() -> dict[str, EngineAvailability]:
    giga_version = package_version("gigastt")
    whisper_version = package_version("faster-whisper")
    tone_version = package_version("tone")
    return {
        "gigaam": EngineAvailability(module_installed("gigastt_uniffi"), giga_version),
        "faster_whisper": EngineAvailability(
            module_installed("faster_whisper"), whisper_version
        ),
        "tone": EngineAvailability(module_installed("tone"), tone_version),
    }


def _ram_gb() -> Optional[float]:
    try:
        import psutil

        return psutil.virtual_memory().total / (1024**3)
    except Exception:
        pass
    if sys.platform == "win32":
        try:

            class MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = MemoryStatus()
            status.dwLength = ctypes.sizeof(MemoryStatus)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return status.ullTotalPhys / (1024**3)
        except Exception:
            return None
    elif sys.platform.startswith("linux"):
        try:
            with open("/proc/meminfo", encoding="ascii") as handle:
                for line in handle:
                    if line.startswith("MemTotal:"):
                        return int(line.split()[1]) / (1024**2)
        except OSError:
            return None
    return None


def _free_disk_gb(path: str) -> Optional[float]:
    try:
        return shutil.disk_usage(path).free / (1024**3)
    except OSError:
        return None


def _nvidia_smi_path() -> Optional[str]:
    found = shutil.which("nvidia-smi")
    if found:
        return found
    if sys.platform == "win32":
        candidate = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "nvidia-smi.exe")
        if os.path.isfile(candidate):
            return candidate
    return None


def _detect_nvidia_smi() -> tuple[bool, Optional[str], Optional[float]]:
    path = _nvidia_smi_path()
    if not path:
        return False, None, None
    try:
        result = subprocess.run(
            [
                path,
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=6,
            creationflags=_CREATE_NO_WINDOW,
        )
    except (OSError, subprocess.SubprocessError):
        logger.debug("nvidia-smi не ответил", exc_info=True)
        return False, None, None
    if result.returncode != 0 or not result.stdout.strip():
        return False, None, None
    first = result.stdout.strip().splitlines()[0]
    parts = [part.strip() for part in first.split(",")]
    name = parts[0] if parts and parts[0] else None
    vram = None
    if len(parts) > 1:
        try:
            vram = float(parts[1]) / 1024
        except ValueError:
            vram = None
    return True, name, vram


def _detect_ctranslate2() -> bool:
    try:
        import ctranslate2

        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


def detect_hardware(probe_path: str | None = None) -> HardwareInfo:
    from . import paths

    cuda, gpu_name, vram = _detect_nvidia_smi()
    if not cuda:
        cuda = _detect_ctranslate2()
    return HardwareInfo(
        cpu_cores=os.cpu_count() or 1,
        ram_gb=_ram_gb(),
        cuda=cuda,
        vram_gb=vram,
        gpu_name=gpu_name,
        free_disk_gb=_free_disk_gb(probe_path or str(paths.models_dir().parent)),
        machine=f"{platform.system()} {platform.release()} ({platform.machine()})",
    )


def resolve_whisper_runtime(
    cfg: Config, hw: HardwareInfo, cudnn_ready: Optional[bool] = None
) -> tuple[str, str, list[str]]:
    """Return ``(device, compute_type, warnings)`` for faster-whisper.

    On Windows the CUDA path is only chosen when a complete cuDNN 9 runtime is
    available: CTranslate2 ships just the loader stub and fails fast
    (``0xc0000409``) without the split libraries. Preloading happens here so
    that it runs before ``faster_whisper`` is imported.
    """
    warnings: list[str] = []
    device = cfg.device
    if device == "cuda" and not hw.cuda:
        warnings.append("CUDA запрошена, но NVIDIA GPU не обнаружен — используется CPU")
        device = "cpu"
    if device == "auto":
        device = "cuda" if hw.cuda else "cpu"
        if hw.cuda and (hw.vram_gb or 0) < 3:
            warnings.append("Меньше 3 ГБ VRAM: Whisper может не поместиться в память GPU")

    if device == "cuda" and sys.platform == "win32":
        if cudnn_ready is None:
            from .cudnn_runtime import ensure_cudnn

            cudnn_ready = ensure_cudnn() is not None
        if not cudnn_ready:
            warnings.append(
                "cuDNN 9 не найден — Whisper работает на CPU. Установите cuDNN 9 "
                "от NVIDIA, чтобы распознавать на GPU"
            )
            logger.warning("Whisper: %s", warnings[-1])
            device = "cpu"

    if cfg.compute_type != "auto":
        compute = cfg.compute_type
    elif device == "cuda":
        vram = hw.vram_gb or 0
        compute = "float16" if vram >= 6 or vram == 0 else "int8_float16"
    else:
        compute = "int8"

    if device == "cpu" and compute in ("float16", "int8_float16"):
        warnings.append(f"Compute type {compute} не поддерживается на CPU — используется int8")
        compute = "int8"
    return device, compute, warnings


def choose_tone_decoder(cfg: Config, hw: HardwareInfo) -> str:
    if cfg.tone_decoder in ("beam_search", "greedy"):
        return cfg.tone_decoder
    ram = hw.ram_gb or 0
    free = hw.free_disk_gb
    if ram >= 8 and (free is None or free >= 5):
        return "beam_search"
    return "greedy"


def choose_engine_name(cfg: Config, hw: HardwareInfo, availability: dict[str, EngineAvailability]) -> str:
    if cfg.model_type != "auto":
        return cfg.model_type

    installed = {name: info.installed for name, info in availability.items()}
    ram = hw.ram_gb or 0

    def first_available(*names: str) -> Optional[str]:
        for name in names:
            if installed.get(name):
                return name
        return None

    # "auto" is Russian-only now, so GigaAM gives the best quality/size ratio,
    # T-one is the low-memory fallback and Whisper the multilingual fallback.
    if installed.get("gigaam") and (ram == 0 or ram >= 4):
        return "gigaam"
    if installed.get("tone") and (ram == 0 or ram >= 4):
        return "tone"
    return first_available("faster_whisper", "gigaam", "tone") or "none"


def format_bytes(size: float) -> str:
    for unit in ("Б", "КБ", "МБ", "ГБ", "ТБ"):
        if abs(size) < 1024 or unit == "ТБ":
            return f"{size:.1f} {unit}" if unit != "Б" else f"{int(size)} Б"
        size /= 1024
    return f"{size:.1f} ТБ"
