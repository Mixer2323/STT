# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_all, collect_submodules

block_cipher = None

datas = []
binaries = []
hiddenimports = []

# --- torch ---
torch_datas, torch_bins, torch_hidden = collect_all('torch')
datas += torch_datas
binaries += torch_bins
hiddenimports += torch_hidden

# --- whisper ---
w_datas, w_bins, w_hidden = collect_all('whisper')
datas += w_datas
binaries += w_bins
hiddenimports += w_hidden

# --- faster whisper ---
hiddenimports += collect_submodules('faster_whisper')

# --- vosk ---
v_datas, v_bins, v_hidden = collect_all('vosk')
datas += v_datas
binaries += v_bins
hiddenimports += v_hidden

# --- audio ---
hiddenimports += ['sounddevice', 'soundfile', 'numpy', 'scipy']

# --- tray / images ---
hiddenimports += collect_submodules('pystray')
hiddenimports += collect_submodules('PIL')

# --- input ---
hiddenimports += ['keyboard', 'pyperclip', 'pyautogui']

# --- CUDA ---
hiddenimports += ['torch.cuda', 'torch.backends.cudnn']

a = Analysis(
    ['voice_typer.py'],
    binaries=binaries,
    datas=datas + [
        ('config.json', '.'),
    ],
    hiddenimports=hiddenimports,
    excludes=['tkinter', 'matplotlib'],
    noarchive=False
)

pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz,
    a.scripts,
    exclude_binaries=True,
    name='VoiceTyper',
    console=False,
    icon=r'C:\PY\STT\icon.ico'
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    name='VoiceTyper'
)
