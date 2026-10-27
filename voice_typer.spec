# -*- mode: python ; coding: utf-8 -*-

import glob
import os
import sysconfig

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
)

datas = []
binaries = []
hiddenimports = []

if os.path.exists("icon.ico"):
    datas.append(("icon.ico", "."))

# Engines are resolved through importlib, so PyInstaller cannot see them.
hiddenimports += collect_submodules("voicetyper")

# GigaAM (uniffi bindings plus the native DLL).
for collector in (collect_data_files, collect_dynamic_libs, collect_submodules):
    try:
        collected = collector("gigastt_uniffi")
        if collector is collect_data_files:
            datas += collected
        elif collector is collect_dynamic_libs:
            binaries += collected
        else:
            hiddenimports += collected
    except Exception:
        pass

# Python modules with lazy imports or plugin discovery. ONNX Runtime is added
# as a hidden import only: its tools/transformers subpackages pull in numba,
# llvmlite and other dev-only dependencies (~100 MB) that are never used here.
for package in (
    "faster_whisper",
    "ctranslate2",
    "tokenizers",
    "huggingface_hub",
    "pystray",
    "PIL",
    "pyctcdecode",
):
    try:
        hiddenimports += collect_submodules(package)
    except Exception:
        pass

hiddenimports += [
    "tone",
    "av",
    "kenlm",
    "keyboard",
    "pyperclip",
    "psutil",
    "sounddevice",
    "soundfile",
    "numpy",
    "onnxruntime",
]

for package in ("ctranslate2", "tokenizers", "av", "onnxruntime.capi", "PIL"):
    try:
        binaries += collect_dynamic_libs(package)
    except Exception:
        pass

# faster_whisper ships the Silero VAD ONNX models under assets/; without them
# every transcription fails at runtime when vad_filter is enabled.
for package in ("faster_whisper", "ctranslate2", "av", "onnxruntime", "pyctcdecode", "PIL"):
    try:
        datas += collect_data_files(package)
    except Exception:
        pass

# sounddevice / soundfile keep their native libraries in sibling packages.
for package in ("_sounddevice_data", "_soundfile_data"):
    try:
        datas += collect_data_files(package)
        hiddenimports += [package]
    except Exception:
        pass

# kenlm is a top-level extension module with a companion DLL.
site_packages = sysconfig.get_paths()["purelib"]
for pattern in ("kenlm*.pyd", "kenlm*.dll"):
    for path in glob.glob(os.path.join(site_packages, pattern)):
        binaries.append((path, "."))

a = Analysis(
    ["voice_typer.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "torch",
        "torchaudio",
        "torchvision",
        "transformers",
        "matplotlib",
        "whisper",
        "vosk",
        "sklearn",
        "pandas",
        "numba",
        "llvmlite",
        "onnxruntime.tools",
        "onnxruntime.transformers",
        "onnxruntime.quantization",
        "onnxruntime.training",
        "IPython",
        "pytest",
        "hypothesis",
    ],
    noarchive=False,
)

# ctranslate2 ships only the cuDNN 9 loader stub without the split runtime
# libraries. Loading that stub in a frozen build kills the process with
# 0xc0000409, so the app preloads a complete system cuDNN 9 instead
# (voicetyper.cudnn_runtime). Filter after Analysis because the DLL arrives
# through collect_data_files and PyInstaller's binary reclassification as well.
def _without_cudnn_loader(entries):
    return [
        entry for entry in entries
        if os.path.basename(entry[0]).lower() != "cudnn64_9.dll"
    ]


a.binaries = _without_cudnn_loader(a.binaries)
a.datas = _without_cudnn_loader(a.datas)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    exclude_binaries=True,
    name="VoiceTyper",
    debug=False,
    strip=False,
    upx=False,
    console=False,
    icon="icon.ico",
    version="version_info.txt",
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="VoiceTyper",
)
