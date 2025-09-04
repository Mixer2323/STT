# Инструкции по сборке EXE приложения

## Проблемы с импортами в EXE

Если у вас возникают проблемы с импортами при запуске EXE файла, выполните следующие шаги:

### 1. Установка зависимостей

```bash
pip install -r requirements.txt
pip install pyinstaller
```

### 2. Способы сборки

#### Способ 1: Автоматическая сборка
```bash
python build_exe.py
```

#### Способ 2: Ручная сборка с улучшенным spec
```bash
pyinstaller voice_typer.spec --clean
```

#### Способ 3: Простая сборка
```bash
pyinstaller voice_typer_simple.spec --clean
```

#### Способ 4: Базовая сборка PyInstaller
```bash
pyinstaller --onefile --windowed --icon=icon.ico --add-data "config.json;." --add-data "settings.ini;." voice_typer.py
```

### 3. Решение частых проблем

#### Проблема: "ModuleNotFoundError"
**Решение:**
1. Добавьте недостающий модуль в `hiddenimports` в spec файле
2. Убедитесь, что модуль установлен: `pip install имя_модуля`

#### Проблема: "DLL load failed"
**Решение:**
1. Установите Visual C++ Redistributable
2. Попробуйте сборку без UPX: `upx=False` в spec файле

#### Проблема: "File not found"
**Решение:**
1. Убедитесь, что все файлы указаны в `datas` в spec файле
2. Проверьте пути к ресурсам в коде

#### Проблема: "Permission denied"
**Решение:**
1. Запустите командную строку от имени администратора
2. Отключите антивирус на время сборки

### 4. Отладка проблем

#### Включение отладочного режима
```bash
pyinstaller --debug=all voice_typer.spec
```

#### Проверка зависимостей
```bash
pip install pipdeptree
pipdeptree -p voice_typer
```

#### Анализ EXE файла
```bash
# Установите pefile для анализа
pip install pefile
python -c "import pefile; pe = pefile.PE('dist/voice_typer.exe'); print([entry.dll for entry in pe.DIRECTORY_ENTRY_IMPORT])"
```

### 5. Альтернативные решения

#### Использование auto-py-to-exe
```bash
pip install auto-py-to-exe
auto-py-to-exe
```

#### Использование cx_Freeze
```bash
pip install cx_Freeze
python setup.py build
```

### 6. Проверка работоспособности

После сборки проверьте:
1. Запуск EXE файла
2. Работа всех функций
3. Доступ к ресурсам (файлы конфигурации, иконки)
4. Работа с аудио устройствами

### 7. Оптимизация размера

Для уменьшения размера EXE:
1. Используйте `--exclude-module` для ненужных модулей
2. Включите UPX сжатие (если нет проблем с DLL)
3. Используйте `--onefile` вместо `--onedir`

### 8. Контакты для поддержки

Если проблемы остаются:
1. Проверьте логи ошибок
2. Убедитесь в совместимости версий Python и библиотек
3. Попробуйте сборку на чистой виртуальной среде
