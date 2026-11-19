from __future__ import annotations

import sys
from pathlib import Path

import pytest

import voicetyper.cudnn_runtime as cudnn_runtime
from voicetyper.config import Config
from voicetyper.hardware import HardwareInfo, resolve_whisper_runtime

_COMPLETE = (
    "cudnn64_9.dll",
    "cudnn_ops64_9.dll",
    "cudnn_cnn64_9.dll",
    "cudnn_graph64_9.dll",
    "cudnn_heuristic64_9.dll",
    "cudnn_engines_precompiled64_9.dll",
)


def _install(directory: Path, names: tuple[str, ...]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    for name in names:
        (directory / name).write_bytes(b"")
    return directory


def test_find_cudnn_skips_loader_only_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        cudnn_runtime,
        "_candidate_dirs",
        lambda: [
            _install(tmp_path / "bundled", ("cudnn64_9.dll",)),
            _install(tmp_path / "system", _COMPLETE),
        ],
    )
    assert cudnn_runtime.find_cudnn_directory() == tmp_path / "system"


def test_find_cudnn_requires_split_libraries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    partial = _install(tmp_path / "partial", _COMPLETE[:-1])
    monkeypatch.setattr(cudnn_runtime, "_candidate_dirs", lambda: [partial])
    assert cudnn_runtime.find_cudnn_directory() is None


def test_candidate_dirs_start_with_cudnn_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CUDNN_PATH", str(tmp_path / "cudnn"))
    monkeypatch.setenv("PATH", "")
    monkeypatch.setenv("ProgramFiles", str(tmp_path / "program"))
    monkeypatch.setenv("ProgramW6432", str(tmp_path / "program"))
    for variable in ("CUDNN_HOME", "CUDA_PATH", "CUDA_HOME"):
        monkeypatch.delenv(variable, raising=False)

    directories = cudnn_runtime._candidate_dirs()

    assert directories[0] == tmp_path / "cudnn"
    assert directories[1] == tmp_path / "cudnn" / "bin"


def test_gpu_without_cudnn_falls_back_to_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    hw = HardwareInfo(cpu_cores=8, cuda=True, vram_gb=12)

    device, compute, warnings = resolve_whisper_runtime(
        Config(device="cuda"), hw, cudnn_ready=False
    )

    assert (device, compute) == ("cpu", "int8")
    assert any("cuDNN" in warning for warning in warnings)


def test_gpu_with_cudnn_uses_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    hw = HardwareInfo(cpu_cores=8, cuda=True, vram_gb=12)

    device, compute, warnings = resolve_whisper_runtime(
        Config(device="auto"), hw, cudnn_ready=True
    )

    assert (device, compute) == ("cuda", "float16")
    assert warnings == []


def test_explicit_compute_is_downgraded_on_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    hw = HardwareInfo(cpu_cores=8, cuda=True, vram_gb=12)

    device, compute, _ = resolve_whisper_runtime(
        Config(device="cuda", compute_type="float16"), hw, cudnn_ready=False
    )

    assert (device, compute) == ("cpu", "int8")


def test_language_support_follows_the_resolved_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from voicetyper import hardware
    from voicetyper.hardware import EngineAvailability

    monkeypatch.setattr(hardware, "_ram_gb", lambda: 16.0)
    only_whisper = {
        "gigaam": EngineAvailability(False),
        "tone": EngineAvailability(False),
        "faster_whisper": EngineAvailability(True),
    }
    assert hardware.config_accepts_language(Config(model_type="auto"), only_whisper)

    with_gigaam = dict(only_whisper)
    with_gigaam["gigaam"] = EngineAvailability(True)
    assert not hardware.config_accepts_language(Config(model_type="auto"), with_gigaam)
    assert hardware.config_accepts_language(Config(model_type="faster_whisper"), with_gigaam)


def test_non_windows_is_not_gated_by_cudnn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    hw = HardwareInfo(cpu_cores=8, cuda=True, vram_gb=12)

    device, _, warnings = resolve_whisper_runtime(
        Config(device="cuda"), hw, cudnn_ready=False
    )

    assert device == "cuda"
    assert warnings == []


def test_nvidia_smi_prefers_the_system_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from voicetyper import hardware

    system32 = tmp_path / "System32"
    system32.mkdir()
    trusted = system32 / "nvidia-smi.exe"
    trusted.write_bytes(b"")
    monkeypatch.setenv("SystemRoot", str(tmp_path))
    monkeypatch.setattr(sys, "platform", "win32")
    planted = str(tmp_path / "planted" / "nvidia-smi.exe")
    monkeypatch.setattr(hardware.shutil, "which", lambda _name: planted)

    assert hardware._nvidia_smi_path() == str(trusted)


def test_nvidia_smi_falls_back_to_path_when_system_copy_is_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from voicetyper import hardware

    monkeypatch.setenv("SystemRoot", str(tmp_path / "nowhere"))
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(hardware.shutil, "which", lambda _name: "C:/tools/nvidia-smi.exe")

    assert hardware._nvidia_smi_path() == "C:/tools/nvidia-smi.exe"


def test_cuda_probe_is_skipped_without_the_nvidia_driver(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Importing ctranslate2 costs seconds on a cold install; a machine without
    # the NVIDIA driver must not pay for it on every start.
    from voicetyper import hardware

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("SystemRoot", str(tmp_path))
    assert hardware._has_nvidia_driver_libraries() is False
    assert hardware._detect_ctranslate2() is False

    (tmp_path / "System32").mkdir()
    (tmp_path / "System32" / "nvcuda.dll").write_bytes(b"")
    assert hardware._has_nvidia_driver_libraries() is True
