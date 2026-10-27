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
