# Инструкции для загрузки проекта на GitHub

## Шаг 1: Создание репозитория на GitHub

1. Перейдите на https://github.com
2. Нажмите кнопку "New repository" (зеленая кнопка)
3. Заполните поля:
   - **Repository name**: `voice-typer` (или любое другое название)
   - **Description**: `Voice to text application with CUDA support`
   - **Visibility**: Public или Private (на ваш выбор)
   - **НЕ** добавляйте README, .gitignore или лицензию (они уже есть)
4. Нажмите "Create repository"

## Шаг 2: Подключение локального репозитория к GitHub

После создания репозитория GitHub покажет команды. Выполните их в терминале:

```bash
# Добавьте удаленный репозиторий (замените YOUR_USERNAME и REPO_NAME)
git remote add origin https://github.com/YOUR_USERNAME/REPO_NAME.git

# Переименуйте ветку в main (если нужно)
git branch -M main

# Загрузите код на GitHub
git push -u origin main
```

## Шаг 3: Проверка

После выполнения команд ваш код будет доступен на GitHub по адресу:
`https://github.com/YOUR_USERNAME/REPO_NAME`

## Альтернативный способ через GitHub Desktop

1. Скачайте и установите GitHub Desktop
2. Откройте GitHub Desktop
3. Нажмите "Add an Existing Repository from your Hard Drive"
4. Выберите папку `C:\PY\STT`
5. Нажмите "Publish repository"
6. Заполните название и описание
7. Нажмите "Publish repository"

## Текущее состояние

✅ Git репозиторий инициализирован
✅ Все файлы добавлены в git
✅ Создан .gitignore
✅ Создан README.md
✅ Сделаны коммиты
⏳ Ожидается создание репозитория на GitHub и push
