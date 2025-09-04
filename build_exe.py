#!/usr/bin/env python3
"""
Скрипт для сборки exe приложения с улучшенной конфигурацией
"""

import os
import sys
import subprocess
import shutil

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

def build_exe():
    """Сборка exe файла"""
    print("🔨 Сборка exe файла...")
    
    try:
        # Сначала собираем с помощью pyinstaller
        subprocess.run([sys.executable, '-m', 'PyInstaller', 'voice_typer.spec', '--clean'], 
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
    print("🚀 Начинаем сборку exe приложения...")
    print("=" * 50)
    
    # Очистка
    clean_build()
    
    # Установка зависимостей
    if not install_dependencies():
        print("❌ Не удалось установить зависимости")
        return
    
    # Сборка
    if not build_exe():
        print("❌ Не удалось собрать exe файл")
        return
    
    # Проверка
    if not check_exe():
        print("❌ Exe файл не был создан")
        return
    
    print("=" * 50)
    print("🎉 Сборка завершена успешно!")
    print(f"📁 Exe файл находится в: {os.path.abspath('dist')}")

if __name__ == "__main__":
    main()
