#!/usr/bin/env python3
"""
Простой тест для проверки вывода
"""

import sys
import os

print("Тест вывода 1")
sys.stdout.flush()

print("Тест вывода 2")
sys.stdout.flush()

print("Текущая директория:", os.getcwd())
sys.stdout.flush()

print("Python версия:", sys.version)
sys.stdout.flush()

input("Нажмите Enter для продолжения...")
print("Тест завершен")
