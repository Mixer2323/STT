#!/usr/bin/env python3
"""
Тестовый скрипт для диагностики проблем с приложением
"""

import sys
import os

print("=== ДИАГНОСТИКА ПРИЛОЖЕНИЯ ===")
print(f"Python версия: {sys.version}")
print(f"Текущая директория: {os.getcwd()}")

# Тестируем импорты
print("\n=== ТЕСТ ИМПОРТОВ ===")

try:
    import os
    print("✅ os")
except Exception as e:
    print(f"❌ os: {e}")

try:
    import json
    print("✅ json")
except Exception as e:
    print(f"❌ json: {e}")

try:
    import threading
    print("✅ threading")
except Exception as e:
    print(f"❌ threading: {e}")

try:
    import tempfile
    print("✅ tempfile")
except Exception as e:
    print(f"❌ tempfile: {e}")

try:
    import time
    print("✅ time")
except Exception as e:
    print(f"❌ time: {e}")

try:
    import subprocess
    print("✅ subprocess")
except Exception as e:
    print(f"❌ subprocess: {e}")

try:
    import platform
    print("✅ platform")
except Exception as e:
    print(f"❌ platform: {e}")

# Тестируем основные библиотеки
print("\n=== ТЕСТ ОСНОВНЫХ БИБЛИОТЕК ===")

try:
    import keyboard
    print("✅ keyboard")
except Exception as e:
    print(f"❌ keyboard: {e}")

try:
    import pyautogui
    print("✅ pyautogui")
except Exception as e:
    print(f"❌ pyautogui: {e}")

try:
    import sounddevice as sd
    print("✅ sounddevice")
except Exception as e:
    print(f"❌ sounddevice: {e}")

try:
    import soundfile as sf
    print("✅ soundfile")
except Exception as e:
    print(f"❌ soundfile: {e}")

try:
    import pystray
    print("✅ pystray")
except Exception as e:
    print(f"❌ pystray: {e}")

try:
    from PIL import Image, ImageDraw, ImageFont
    print("✅ PIL")
except Exception as e:
    print(f"❌ PIL: {e}")

try:
    import whisper
    print("✅ whisper")
except Exception as e:
    print(f"❌ whisper: {e}")

try:
    import pyperclip
    print("✅ pyperclip")
except Exception as e:
    print(f"❌ pyperclip: {e}")

try:
    import torch
    print("✅ torch")
    print(f"   Версия PyTorch: {torch.__version__}")
    print(f"   CUDA доступна: {torch.cuda.is_available()}")
except Exception as e:
    print(f"❌ torch: {e}")

try:
    from faster_whisper import WhisperModel
    print("✅ faster_whisper")
except Exception as e:
    print(f"❌ faster_whisper: {e}")

try:
    from vosk import Model as VoskModel, KaldiRecognizer
    print("✅ vosk")
except Exception as e:
    print(f"❌ vosk: {e}")

# Тестируем файлы конфигурации
print("\n=== ТЕСТ ФАЙЛОВ КОНФИГУРАЦИИ ===")

config_file = "config.json"
if os.path.exists(config_file):
    print(f"✅ {config_file} найден")
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        print(f"   Конфигурация загружена: {list(config.keys())}")
    except Exception as e:
        print(f"   ❌ Ошибка загрузки конфигурации: {e}")
else:
    print(f"❌ {config_file} не найден")

settings_file = "settings.ini"
if os.path.exists(settings_file):
    print(f"✅ {settings_file} найден")
else:
    print(f"❌ {settings_file} не найден")

# Тестируем пути к ресурсам
print("\n=== ТЕСТ ПУТЕЙ К РЕСУРСАМ ===")

def get_resource_path(relative_path):
    """Получить путь к ресурсу, работает как для .py так и для .exe"""
    try:
        # PyInstaller создает временную папку и хранит путь в _MEIPASS
        base_path = sys._MEIPASS
        print(f"   Используется PyInstaller путь: {base_path}")
    except Exception:
        base_path = os.path.dirname(os.path.abspath(__file__))
        print(f"   Используется обычный путь: {base_path}")
    return os.path.join(base_path, relative_path)

# Тестируем функцию get_resource_path
try:
    config_path = get_resource_path("config.json")
    print(f"✅ Путь к config.json: {config_path}")
    if os.path.exists(config_path):
        print("   Файл существует")
    else:
        print("   Файл не существует")
except Exception as e:
    print(f"❌ Ошибка получения пути: {e}")

# Тестируем CUDA
print("\n=== ТЕСТ CUDA ===")

try:
    if torch is not None:
        print(f"PyTorch версия: {torch.__version__}")
        print(f"CUDA доступна: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA версия: {torch.version.cuda}")
            print(f"Количество устройств: {torch.cuda.device_count()}")
            print(f"Текущее устройство: {torch.cuda.current_device()}")
        else:
            print("CUDA недоступна")
    else:
        print("PyTorch не загружен")
except Exception as e:
    print(f"❌ Ошибка проверки CUDA: {e}")

print("\n=== ДИАГНОСТИКА ЗАВЕРШЕНА ===")
input("Нажмите Enter для выхода...")
