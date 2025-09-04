#!/usr/bin/env python3
"""
Скрипт для установки CUDA версии PyTorch
"""

import subprocess
import sys
import os

def run_command(command, description):
    print(f"\n{description}...")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} успешно")
            if result.stdout:
                print(f"Вывод: {result.stdout}")
        else:
            print(f"❌ {description} неудачно")
            if result.stderr:
                print(f"Ошибка: {result.stderr}")
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Ошибка при выполнении команды: {e}")
        return False

def main():
    print("=== УСТАНОВКА CUDA ВЕРСИИ PYTORCH ===")
    
    # Проверяем текущую версию PyTorch
    print("\n1. Проверяем текущую версию PyTorch...")
    success = run_command("pip show torch", "Проверка текущей версии PyTorch")
    
    # Удаляем текущую версию PyTorch
    print("\n2. Удаляем текущую версию PyTorch...")
    success = run_command("pip uninstall torch torchvision torchaudio -y", "Удаление PyTorch")
    
    # Устанавливаем CUDA версию PyTorch
    print("\n3. Устанавливаем CUDA версию PyTorch...")
    # Используем команду для установки CUDA версии PyTorch
    cuda_command = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
    success = run_command(cuda_command, "Установка CUDA версии PyTorch")
    
    # Проверяем установку
    print("\n4. Проверяем установку...")
    success = run_command("python -c \"import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA доступна: {torch.cuda.is_available()}')\"", "Проверка CUDA")
    
    if success:
        print("\n✅ CUDA версия PyTorch установлена успешно!")
    else:
        print("\n❌ Ошибка при установке CUDA версии PyTorch")
    
    input("\nНажмите Enter для выхода...")

if __name__ == "__main__":
    main()
