#!/usr/bin/env python3
"""
Скрипт для сборки одного exe файла без консоли
"""

import os
import sys
import subprocess
import shutil
import site
import glob

def clean_build():
    """Очистка предыдущих сборок"""
    print("🧹 Очистка предыдущих сборок...")
    
    dirs_to_clean = ['build', 'dist']
    for dir_name in dirs_to_clean:
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)
            print(f"   Удалена папка {dir_name}")

def install_dependencies():
    """Установка зависимостей"""
    print("📦 Установка зависимостей...")
    
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'], 
                      check=True, capture_output=True, text=True)
        print("   ✅ Зависимости установлены")
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Ошибка установки зависимостей: {e}")
        return False
    
    return True

def find_vosk_dlls():
    """Поиск DLL файлов Vosk"""
    print("🔍 Поиск DLL файлов Vosk...")
    
    vosk_dlls = []
    
    # Ищем в site-packages
    for site_dir in site.getsitepackages():
        vosk_dir = os.path.join(site_dir, 'vosk')
        if os.path.exists(vosk_dir):
            dll_files = glob.glob(os.path.join(vosk_dir, '*.dll'))
            vosk_dlls.extend(dll_files)
            print(f"   Найдены DLL в {vosk_dir}: {len(dll_files)} файлов")
    
    # Ищем в пользовательских пакетах
    user_site = site.getusersitepackages()
    if os.path.exists(user_site):
        vosk_dir = os.path.join(user_site, 'vosk')
        if os.path.exists(vosk_dir):
            dll_files = glob.glob(os.path.join(vosk_dir, '*.dll'))
            vosk_dlls.extend(dll_files)
            print(f"   Найдены DLL в {vosk_dir}: {len(dll_files)} файлов")
    
    return vosk_dlls

def create_vosk_hook():
    """Создание hook для Vosk"""
    print("🔧 Создание hook для Vosk...")
    
    hook_content = '''
from PyInstaller.utils.hooks import collect_dynamic_libs

# Собираем все DLL файлы Vosk
binaries = collect_dynamic_libs('vosk')
'''
    
    hook_dir = 'hooks'
    if not os.path.exists(hook_dir):
        os.makedirs(hook_dir)
    
    hook_file = os.path.join(hook_dir, 'hook-vosk.py')
    with open(hook_file, 'w', encoding='utf-8') as f:
        f.write(hook_content)
    
    print(f"   ✅ Hook создан: {hook_file}")

def build_exe_onefile():
    """Сборка одного exe файла"""
    print("🔨 Сборка одного exe файла...")
    
    try:
        # Создаем временный spec файл для одного файла
        spec_content = '''# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

# Собираем DLL файлы Vosk
import os
import glob
import site

vosk_binaries = []
for site_dir in site.getsitepackages():
    vosk_dir = os.path.join(site_dir, 'vosk')
    if os.path.exists(vosk_dir):
        dll_files = glob.glob(os.path.join(vosk_dir, '*.dll'))
        for dll in dll_files:
            vosk_binaries.append((dll, '.'))

a = Analysis(
    ['voice_typer.py'],
    pathex=[],
    binaries=vosk_binaries,
    datas=[
        ('config.json', '.'),
        ('settings.ini', '.'),
        ('icon.ico', '.'),
        ('icon.png', '.'),
        ('tbb12.dll', '.')
    ],
    hiddenimports=[
        'whisper', 'faster_whisper', 'keyboard', 'pyautogui', 'sounddevice', 
        'soundfile', 'pystray', 'PIL', 'pyperclip', 'torch', 'vosk', 
        'numpy', 'psutil', 'comtypes', 'pycaw', 'torchaudio',
        'whisper.model', 'whisper.decoding', 'whisper.audio',
        'faster_whisper.transcribe', 'faster_whisper.model',
        'vosk.model', 'vosk.recognizer',
        'pystray._util_win32', 'pystray._util_gtk', 'pystray._util_x11',
        'PIL._tkinter_finder', 'PIL.Image', 'PIL.ImageDraw', 'PIL.ImageFont',
        'keyboard._winkeyboard', 'keyboard._nixkeyboard',
        'sounddevice._sounddevice', 'soundfile._soundfile',
        'pyautogui._pyautogui_win', 'pyautogui._pyautogui_x11', 'pyautogui._pyautogui_osx'
    ],
    hookspath=['hooks'],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'matplotlib', 'scipy', 'pandas', 'seaborn', 'jupyter', 'IPython',
        'tkinter', 'wx', 'PyQt5', 'PyQt6', 'PySide2', 'PySide6'
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='voice_typer',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # Без консоли
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='icon.ico'
)
'''
        
        with open('voice_typer_onefile.spec', 'w', encoding='utf-8') as f:
            f.write(spec_content)
        
        # Собираем с помощью pyinstaller
        subprocess.run([sys.executable, '-m', 'PyInstaller', 'voice_typer_onefile.spec', '--clean'], 
                      check=True, capture_output=True, text=True)
        print("   ✅ Exe файл собран успешно")
        return True
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Ошибка сборки: {e}")
        if e.stdout:
            print(f"   stdout: {e.stdout}")
        if e.stderr:
            print(f"   stderr: {e.stderr}")
        return False

def check_exe():
    """Проверка созданного exe файла"""
    print("🔍 Проверка exe файла...")
    
    exe_path = os.path.join('dist', 'voice_typer.exe')
    if os.path.exists(exe_path):
        size_mb = os.path.getsize(exe_path) / (1024 * 1024)
        print(f"   ✅ Exe файл создан: {exe_path}")
        print(f"   📏 Размер: {size_mb:.1f} MB")
        return True
    else:
        print("   ❌ Exe файл не найден")
        return False

def main():
    """Основная функция"""
    print("🚀 Начинаем сборку одного exe файла без консоли...")
    print("=" * 60)
    
    # Очистка
    clean_build()
    
    # Установка зависимостей
    if not install_dependencies():
        print("❌ Не удалось установить зависимости")
        return
    
    # Поиск DLL файлов Vosk
    vosk_dlls = find_vosk_dlls()
    if vosk_dlls:
        print(f"   ✅ Найдено {len(vosk_dlls)} DLL файлов Vosk")
    else:
        print("   ⚠️ DLL файлы Vosk не найдены")
    
    # Создание hook
    create_vosk_hook()
    
    # Сборка
    if not build_exe_onefile():
        print("❌ Не удалось собрать exe файл")
        return
    
    # Проверка
    if not check_exe():
        print("❌ Exe файл не был создан")
        return
    
    print("=" * 60)
    print("🎉 Сборка завершена успешно!")
    print(f"📁 Exe файл находится в: {os.path.abspath('dist')}")
    print("💡 Особенности сборки:")
    print("   - Один exe файл (без дополнительных папок)")
    print("   - Без консольного окна")
    print("   - Исправлена проблема с Vosk DLL")
    print("   - Улучшена поддержка CUDA")

if __name__ == "__main__":
    main()
