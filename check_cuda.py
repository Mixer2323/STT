#!/usr/bin/env python3
"""
Скрипт для проверки CUDA
"""

import sys
import subprocess
import os

print("=== ПРОВЕРКА CUDA ===")
print(f"Python версия: {sys.version}")

# Проверяем PyTorch
try:
    import torch
    print(f"PyTorch версия: {torch.__version__}")
    print(f"CUDA доступна: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA версия: {torch.version.cuda}")
        print(f"Количество устройств: {torch.cuda.device_count()}")
        print(f"Текущее устройство: {torch.cuda.current_device()}")
        print(f"Имя устройства: {torch.cuda.get_device_name(0)}")
    else:
        print("❌ CUDA недоступна в PyTorch")
        # Проверяем, не CPU ли это версия
        if "+cpu" in torch.__version__:
            print("⚠️ Установлена CPU версия PyTorch")
        else:
            print("⚠️ PyTorch установлен, но CUDA не работает")
except ImportError:
    print("❌ PyTorch не установлен")

# Проверяем CUDA через nvidia-smi
print("\n=== ПРОВЕРКА NVIDIA-SMI ===")
try:
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
    if result.returncode == 0:
        print("✅ nvidia-smi работает:")
        print(result.stdout)
    else:
        print("❌ nvidia-smi не работает")
        print(f"Ошибка: {result.stderr}")
except FileNotFoundError:
    print("❌ nvidia-smi не найден (NVIDIA драйверы не установлены)")
except subprocess.TimeoutExpired:
    print("❌ nvidia-smi завис")

# Проверяем переменные окружения
print("\n=== ПЕРЕМЕННЫЕ ОКРУЖЕНИЯ ===")
cuda_vars = ['CUDA_HOME', 'CUDA_PATH', 'CUDA_VISIBLE_DEVICES']
for var in cuda_vars:
    value = os.environ.get(var, 'Не установлена')
    print(f"{var}: {value}")

input("\nНажмите Enter для выхода...")
