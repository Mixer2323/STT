import os
import json
import threading
import tempfile
import keyboard
import pyautogui
import sounddevice as sd
import soundfile as sf
import pystray
from PIL import Image, ImageDraw, ImageFont
import whisper
import keyboard as kb
import pyperclip
import torch
from faster_whisper import WhisperModel
import time
import subprocess
import sys
import platform
from vosk import Model as VoskModel, KaldiRecognizer
import logging
from datetime import datetime

# Функции для работы с путями
def get_resource_path(relative_path):
    """Получить путь к ресурсу, работает как для .py так и для .exe"""
    try:
        # PyInstaller создает временную папку и хранит путь в _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, relative_path)

def get_app_path():
    """Получить путь к папке с приложением (рядом с .exe)"""
    if getattr(sys, 'frozen', False):
        # Если запущено как .exe
        return os.path.dirname(sys.executable)
    else:
        # Если запущено как .py
        return os.path.dirname(os.path.abspath(__file__))

# ==== Настройка логирования ====
def setup_logging():
    """Настройка системы логирования"""
    # Создаем папку для логов если её нет
    log_dir = os.path.join(get_app_path(), "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # Имя файла лога с датой
    log_file = os.path.join(log_dir, f"voice_typer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    # Проверяем доступность stdout
    stdout_available = False
    try:
        if sys.stdout is not None and hasattr(sys.stdout, 'isatty') and sys.stdout.isatty():
            stdout_available = True
    except:
        pass
    
    # Настройка логирования
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout) if stdout_available else logging.NullHandler()
        ]
    )
    
    return logging.getLogger(__name__)

# Инициализируем логгер
logger = setup_logging()

# Функция для безопасного вывода (работает и с консолью, и без неё)
def safe_print(message, level="info"):
    """Безопасный вывод сообщений"""
    if level == "info":
        logger.info(message)
    elif level == "warning":
        logger.warning(message)
    elif level == "error":
        logger.error(message)
    elif level == "debug":
        logger.debug(message)
    
    # Также пытаемся вывести в консоль, если она доступна
    try:
        if sys.stdout is not None and hasattr(sys.stdout, 'isatty') and sys.stdout.isatty():
            print(message)
    except:
        pass

CONFIG_FILE = get_resource_path("config.json")

# ==== Настройки по умолчанию ====
default_config = {
    "hotkey": "ctrl+shift",  # сочетание клавиш
    "language": "auto",  # "ru", "en", "auto"
    "model": "small",  # tiny, base, small, medium, large
    "model_type": "whisper",  # whisper, faster_whisper, vosk
    "device": "cuda",  # "cuda" или "cpu"
    "force_cpu": False,  # принудительно использовать CPU
    "vosk_model": "vosk-model-ru-0.42"  # модель Vosk по умолчанию
}

# Варианты для выпадающих списков
LANGUAGE_OPTIONS = ["auto", "ru", "en", "de", "fr", "es", "it", "pt", "zh", "ja"]
MODEL_OPTIONS = ["tiny", "base", "small", "medium", "large"]
DEVICE_OPTIONS = ["cuda", "cpu"]
MODEL_TYPE_OPTIONS = ["Whisper", "Faster Whisper", "Vosk"]
VOSK_MODEL_OPTIONS = [
    "vosk-model-ru-0.42",
    "vosk-model-en-us-0.22",
    "vosk-model-small-ru-0.22",
    "vosk-model-small-en-us-0.15"
]

# Глобальные переменные
model = None
config = {}
icon = None
running = True  # Флаг работы программы

# Функция для сохранения конфигурации
def save_config():
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)

# Функция для загрузки или создания конфигурации
def load_config():
    global config
    # Проверяем наличие конфигурационного файла
    if not os.path.exists(CONFIG_FILE):
        config = default_config.copy()
        save_config()
    else:
        # Если файл существует, загружаем его
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            config = json.load(f)
        # Проверяем наличие всех ключей и добавляем отсутствующие
        config_updated = False
        for key, value in default_config.items():
            if key not in config:
                config[key] = value
                config_updated = True
        # Если конфиг был обновлен, сохраняем изменения
        if config_updated:
            save_config()
            safe_print("🔄 Конфигурация обновлена")

# ==== Проверка и настройка CUDA ====
def check_cuda_installation():
    safe_print("=== Проверка установки CUDA ===")
    # Проверяем наличие nvidia-smi
    try:
        result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0:
            safe_print("✅ nvidia-smi доступен")
            # Выводим только первую часть вывода для краткости
            lines = result.stdout.split('\n')
            for line in lines[:10]:
                safe_print(line)
        else:
            safe_print("❌ nvidia-smi недоступен")
            safe_print(result.stderr)
    except FileNotFoundError:
        safe_print("❌ nvidia-smi не найден. Установите драйверы NVIDIA.")
    # Проверяем переменные окружения CUDA
    cuda_path = os.environ.get('CUDA_PATH')
    if cuda_path:
        safe_print(f"✅ CUDA_PATH: {cuda_path}")
    else:
        safe_print("❌ CUDA_PATH не установлена")
    # Проверяем наличие CUDA DLL
    cuda_dll_paths = [
        os.path.join(os.environ.get('CUDA_PATH', ''), 'bin', 'cudart64_*.dll'),
        os.path.join(os.environ.get('CUDA_PATH', ''), 'bin', 'cudnn64_*.dll'),
        'C:\\Windows\\System32\\cudart64_*.dll',
        'C:\\Windows\\System32\\cudnn64_*.dll'
    ]
    for path_pattern in cuda_dll_paths:
        import glob
        dll_files = glob.glob(path_pattern)
        if dll_files:
            safe_print(f"✅ Найдены CUDA DLL: {dll_files}")
        else:
            safe_print(f"❌ Не найдены CUDA DLL по шаблону: {path_pattern}")

def setup_cuda():
    safe_print("=== Проверка CUDA ===")
    safe_print(f"ОС: {platform.system()} {platform.version()}")
    safe_print(f"Python версия: {sys.version}")
    safe_print(f"PyTorch версия: {torch.__version__}")
    
    # Проверяем установку CUDA
    check_cuda_installation()
    
    # Проверяем доступность CUDA в PyTorch
    cuda_available = torch.cuda.is_available()
    safe_print(f"CUDA доступна в PyTorch: {cuda_available}")
    
    if cuda_available:
        safe_print(f"CUDA версия: {torch.version.cuda}")
        safe_print(f"Устройство CUDA: {torch.cuda.get_device_name(0)}")
        safe_print(f"Количество устройств: {torch.cuda.device_count()}")
        # Устанавливаем переменную окружения для CUDA
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        # Проверяем наличие cuDNN
        try:
            safe_print(f"cuDNN доступна: {torch.backends.cudnn.is_available()}")
            safe_print(f"Версия cuDNN: {torch.backends.cudnn.version()}")
        except:
            safe_print("Не удалось определить версию cuDNN")
        # Принудительно включаем CUDA
        torch.cuda.set_device(0)
        safe_print(f"Текущее устройство CUDA: {torch.cuda.current_device()}")
        return True
    else:
        safe_print("❌ CUDA недоступна в PyTorch. Будет использован CPU.")
        # Дополнительная диагностика
        safe_print("\n=== Дополнительная диагностика ===")
        safe_print(f"Версия PyTorch, собранная с CUDA: {torch.version.cuda}")
        
        # Проверяем, не CPU ли это
        if "+cpu" in torch.__version__:
            safe_print("❌ Установлена CPU-версия PyTorch, а не CUDA-версия!")
            safe_print("\nДля исправления проблемы выполните следующие команды:")
            safe_print("1. pip uninstall torch torchvision torchaudio")
            safe_print("2. pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
            safe_print("\nЕсли это не помогает, попробуйте использовать Python 3.10 или 3.11 вместо 3.13")
        
        # Проверяем версию Python
        if sys.version_info.major == 3 and sys.version_info.minor == 13:
            safe_print("\n⚠️ Вы используете Python 3.13, для которого может не быть сборки PyTorch с CUDA.")
            safe_print("Рекомендуется использовать Python 3.10 или 3.11.")
        
        # Проверяем nvidia-smi
        try:
            import subprocess
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                safe_print("✅ nvidia-smi работает - GPU доступен")
            else:
                safe_print("❌ nvidia-smi не работает - проблема с драйверами")
        except:
            safe_print("❌ nvidia-smi не найден - NVIDIA драйверы не установлены")
        
        return False

# ==== Определение устройства ====
def get_device():
    # Проверяем доступность CUDA
    cuda_available = setup_cuda()
    # Если принудительно включен CPU или CUDA недоступна
    if config.get("force_cpu", False) or not cuda_available:
        config["device"] = "cpu"
        save_config()
        safe_print(f"Используется устройство: CPU")
        return "cpu"
    else:
        safe_print(f"Используется устройство: CUDA")
        return "cuda"

# ==== Загрузка модели ====
def load_model():
    global model
    device = get_device()
    model_type = config.get("model_type", "whisper")
    safe_print(f"🔄 Загружаю модель {model_type} ({config.get('model', 'small')}) на {device}...")
    try:
        if model_type == "whisper":
            # Для обычного Whisper
            if device == "cuda":
                try:
                    # Явно указываем устройство при загрузке модели
                    model = whisper.load_model(config["model"]).to("cuda")
                    safe_print("✅ Модель Whisper загружена на CUDA")
                except Exception as e:
                    safe_print(f"⚠️ Ошибка загрузки CUDA версии whisper: {e}")
                    safe_print("🔄 Переключаюсь на CPU...")
                    config["device"] = "cpu"
                    save_config()
                    model = whisper.load_model(config["model"])
                    safe_print("✅ Модель Whisper загружена на CPU")
            else:
                model = whisper.load_model(config["model"])
                safe_print("✅ Модель Whisper загружена на CPU")
        elif model_type == "faster_whisper":
            # Для Faster Whisper
            if device == "cuda":
                try:
                    # Указываем директорию для моделей
                    model_dir = os.path.join(get_app_path(), "models")
                    os.makedirs(model_dir, exist_ok=True)
                    # Явно указываем использование CUDA
                    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
                    model = WhisperModel(
                        config["model"],
                        device="cuda",  # Явно указываем CUDA
                        compute_type="float16",
                        download_root=model_dir
                    )
                    safe_print("✅ Модель Faster Whisper загружена на CUDA")
                except Exception as e:
                    safe_print(f"⚠️ Ошибка загрузки CUDA версии faster_whisper: {e}")
                    safe_print("🔄 Переключаюсь на CPU...")
                    config["device"] = "cpu"
                    save_config()
                    model = WhisperModel(
                        config["model"],
                        device="cpu",
                        compute_type="int8",
                        download_root=model_dir
                    )
                    safe_print("✅ Модель Faster Whisper загружена на CPU")
            else:
                model_dir = os.path.join(get_app_path(), "models")
                os.makedirs(model_dir, exist_ok=True)
                model = WhisperModel(
                    config["model"],
                    device="cpu",
                    compute_type="int8",
                    download_root=model_dir
                )
                safe_print("✅ Модель Faster Whisper загружена на CPU")
        elif model_type == "vosk":
            # Для Vosk
            vosk_model_path = os.path.join(get_app_path(), "vosk_models", config["vosk_model"])
            # Проверяем, существует ли директория с моделью
            if not os.path.exists(vosk_model_path):
                safe_print(f"❌ Модель Vosk не найдена: {vosk_model_path}")
                safe_print("Пожалуйста, скачайте модель Vosk и распакуйте ее в папку vosk_models")
                safe_print("Модели можно скачать здесь: https://alphacephei.com/vosk/models")
                return False
            try:
                # Загружаем модель Vosk
                model = VoskModel(vosk_model_path)
                safe_print(f"✅ Модель Vosk ({config['vosk_model']}) загружена")
            except Exception as e:
                safe_print(f"❌ Ошибка загрузки модели Vosk: {e}")
                return False
        return True
    except Exception as e:
        safe_print(f"❌ Ошибка загрузки модели: {e}")
        return False

# ==== Флаг записи ====
recording_flag = threading.Event()
recording_thread = None

# ==== Запись звука ====
def record_audio(filename="temp.wav", samplerate=16000):
    safe_print("🎤 Начало записи...")
    try:
        with sf.SoundFile(filename, mode='w', samplerate=samplerate, channels=1, subtype="PCM_16") as file:
            with sd.InputStream(samplerate=samplerate, channels=1, dtype="int16") as stream:
                while recording_flag.is_set():
                    data, _ = stream.read(1024)
                    file.write(data)
        safe_print("🛑 Запись остановлена, файл сохранён:", filename)
    except Exception as e:
        safe_print(f"❌ Ошибка записи: {e}")

# ==== Распознавание + вставка ====
def transcribe_and_paste(file_path):
    try:
        model_type = config.get("model_type", "whisper")
        lang = None if config["language"] == "auto" else config["language"]
        if model_type == "whisper":
            result = model.transcribe(file_path, language=lang)
            text = result["text"].strip()
        elif model_type == "faster_whisper":
            segments, info = model.transcribe(file_path, language=lang)
            text = " ".join([segment.text for segment in segments]).strip()
        elif model_type == "vosk":
            # Для Vosk нужно открыть аудиофайл и распознать его
            rec = KaldiRecognizer(model, 16000)
            with open(file_path, "rb") as f:
                # Читаем WAV заголовок
                f.read(44)  # Пропускаем заголовок WAV
                while True:
                    data = f.read(4096)
                    if len(data) == 0:
                        break
                    if rec.AcceptWaveform(data):
                        result = json.loads(rec.Result())
                        if 'text' in result and result['text']:
                            text = result['text']
                            break
                # Получаем финальный результат
                final_result = json.loads(rec.FinalResult())
                if 'text' in final_result and final_result['text']:
                    text = final_result['text']
                else:
                    text = ""
        safe_print("📝 Распознано:", text)
        # вставляем через буфер обмена
        pyperclip.copy(text)
        kb.press_and_release("ctrl+v")
    except Exception as e:
        safe_print("❌ Ошибка распознавания:", e)

# ==== Управление записью ====
def start_recording():
    global recording_thread
    safe_print("🎤 Горячая клавиша нажата - начало записи")
    if not recording_flag.is_set():
        recording_flag.set()
        recording_thread = threading.Thread(target=record_audio, args=("temp.wav",))
        recording_thread.start()
        safe_print("✅ Запись запущена")

def stop_recording():
    safe_print("🛑 Горячая клавиша отпущена - остановка записи")
    if recording_flag.is_set():
        recording_flag.clear()
        if recording_thread:
            recording_thread.join()
        # Запускаем распознавание
        safe_print("🔄 Запуск распознавания...")
        threading.Thread(target=transcribe_and_paste, args=("temp.wav",)).start()

# ==== Перенастройка горячих клавиш ====
def setup_hotkey():
    # Сначала удаляем старые горячие клавиши
    try:
        keyboard.remove_all_hotkeys()
        safe_print("🗑️ Удалены старые горячие клавиши")
    except Exception as e:
        safe_print(f"⚠️ Ошибка при удалении горячих клавиш: {e}")
    
    # Устанавливаем новые
    hotkey = config["hotkey"]
    safe_print(f"⌨️ Настройка горячей клавиши: {hotkey}")
    
    try:
        keyboard.add_hotkey(hotkey, start_recording, suppress=False, trigger_on_release=False)
        keyboard.on_release_key(hotkey.split("+")[-1], lambda _: stop_recording())
        safe_print(f"✅ Горячая клавиша {hotkey} настроена успешно")
    except Exception as e:
        safe_print(f"❌ Ошибка настройки горячей клавиши: {e}")

# ==== Функции для изменения настроек ====
def set_model_type(icon, item):
    model_type_map = {
        "Whisper": "whisper",
        "Faster Whisper": "faster_whisper",
        "Vosk": "vosk"
    }
    config["model_type"] = model_type_map[item.text]
    save_config()
    safe_print(f"🔄 Тип модели: {item.text}")
    load_model()
    update_menu()

def set_device(icon, item):
    config["device"] = item.text
    config["force_cpu"] = (item.text == "cpu")
    save_config()
    safe_print(f"🔄 Устройство: {item.text}")
    load_model()
    update_menu()

def set_language(icon, item):
    config["language"] = item.text
    save_config()
    safe_print(f"🔄 Язык: {item.text}")
    update_menu()

def set_model(icon, item):
    config["model"] = item.text
    save_config()
    safe_print(f"🔄 Модель: {item.text}")
    load_model()
    update_menu()

def set_vosk_model(icon, item):
    config["vosk_model"] = item.text
    save_config()
    safe_print(f"🔄 Модель Vosk: {item.text}")
    load_model()
    update_menu()

# ==== Функции для проверки состояния пунктов меню ====
def check_model_type(option):
    model_type_map = {
        "Whisper": "whisper",
        "Faster Whisper": "faster_whisper",
        "Vosk": "vosk"
    }
    return lambda _: model_type_map[option] == config.get("model_type", "whisper")

def check_device(option):
    return lambda _: option == config["device"]

def check_language(option):
    return lambda _: option == config["language"]

def check_model(option):
    return lambda _: option == config["model"]

def check_vosk_model(option):
    return lambda _: option == config.get("vosk_model", "vosk-model-ru-0.42")

# ==== Функции для меню трея ====
def create_image(text, width=64, height=64, color=(0, 128, 0), text_color=(255, 255, 255)):
    image = Image.new('RGB', (width, height), color)
    draw = ImageDraw.Draw(image)
    # Попытка загрузить шрифт, если не получается, используем стандартный
    try:
        font = ImageFont.truetype("arial.ttf", 10)
    except:
        font = ImageFont.load_default()
    # Размещаем текст по центру
    try:
        # Для новых версий PIL
        text_width, text_height = draw.textsize(text, font=font)
    except:
        # Для старых версий PIL
        text_width, text_height = draw.textbbox((0, 0), text, font=font)[2:]
    position = ((width - text_width) // 2, (height - text_height) // 2)
    draw.text(position, text, fill=text_color, font=font)
    return image

def on_quit(icon, item):
    global running
    running = False
    icon.stop()
    os._exit(0)

def update_menu():
    global icon
    if icon:
        # Создаем подменю для каждого параметра
        # Подменю для типа модели
        model_type_menu = pystray.Menu(
            *[pystray.MenuItem(
                option,
                set_model_type,
                checked=check_model_type(option)
            ) for option in MODEL_TYPE_OPTIONS]
        )
        # Подменю для устройства (только для Whisper)
        device_menu = pystray.Menu(
            *[pystray.MenuItem(
                option,
                set_device,
                checked=check_device(option)
            ) for option in DEVICE_OPTIONS]
        )
        # Подменю для языка
        language_menu = pystray.Menu(
            *[pystray.MenuItem(
                option,
                set_language,
                checked=check_language(option)
            ) for option in LANGUAGE_OPTIONS]
        )
        # Подменю для модели Whisper
        model_menu = pystray.Menu(
            *[pystray.MenuItem(
                option,
                set_model,
                checked=check_model(option)
            ) for option in MODEL_OPTIONS]
        )
        # Подменю для модели Vosk
        vosk_model_menu = pystray.Menu(
            *[pystray.MenuItem(
                option,
                set_vosk_model,
                checked=check_vosk_model(option)
            ) for option in VOSK_MODEL_OPTIONS]
        )
        # Основное меню - разные пункты в зависимости от типа модели
        if config.get("model_type", "whisper") == "vosk":
            menu_items = [
                pystray.MenuItem("Тип модели", model_type_menu),
                pystray.MenuItem("Язык", language_menu),
                pystray.MenuItem("Модель Vosk", vosk_model_menu),
                pystray.MenuItem(f"Горячая клавиша: {config['hotkey']}", None, enabled=False),
                pystray.Menu.SEPARATOR,
                pystray.MenuItem("Выход", on_quit)
            ]
        else:
            menu_items = [
                pystray.MenuItem("Тип модели", model_type_menu),
                pystray.MenuItem("Устройство", device_menu),
                pystray.MenuItem("Язык", language_menu),
                pystray.MenuItem("Модель", model_menu),
                pystray.MenuItem(f"Горячая клавиша: {config['hotkey']}", None, enabled=False),
                pystray.Menu.SEPARATOR,
                pystray.MenuItem("Выход", on_quit)
            ]
        menu = pystray.Menu(*menu_items)
        icon.menu = menu
        # Обновляем иконку с текущим языком
        icon.icon = create_image(config['language'].upper())

def run_tray():
    global icon
    # Создаем иконку с текущим языком
    icon_image = create_image(config['language'].upper())
    # Создаем подменю для каждого параметра
    # Подменю для типа модели
    model_type_menu = pystray.Menu(
        *[pystray.MenuItem(
            option,
            set_model_type,
            checked=check_model_type(option)
        ) for option in MODEL_TYPE_OPTIONS]
    )
    # Подменю для устройства (только для Whisper)
    device_menu = pystray.Menu(
        *[pystray.MenuItem(
            option,
            set_device,
            checked=check_device(option)
        ) for option in DEVICE_OPTIONS]
    )
    # Подменю для языка
    language_menu = pystray.Menu(
        *[pystray.MenuItem(
            option,
            set_language,
            checked=check_language(option)
        ) for option in LANGUAGE_OPTIONS]
    )
    # Подменю для модели Whisper
    model_menu = pystray.Menu(
        *[pystray.MenuItem(
            option,
            set_model,
            checked=check_model(option)
        ) for option in MODEL_OPTIONS]
    )
    # Подменю для модели Vosk
    vosk_model_menu = pystray.Menu(
        *[pystray.MenuItem(
            option,
            set_vosk_model,
            checked=check_vosk_model(option)
        ) for option in VOSK_MODEL_OPTIONS]
    )
    # Основное меню - разные пункты в зависимости от типа модели
    if config.get("model_type", "whisper") == "vosk":
        menu_items = [
            pystray.MenuItem("Тип модели", model_type_menu),
            pystray.MenuItem("Язык", language_menu),
            pystray.MenuItem("Модель Vosk", vosk_model_menu),
            pystray.MenuItem(f"Горячая клавиша: {config['hotkey']}", None, enabled=False),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Выход", on_quit)
        ]
    else:
        menu_items = [
            pystray.MenuItem("Тип модели", model_type_menu),
            pystray.MenuItem("Устройство", device_menu),
            pystray.MenuItem("Язык", language_menu),
            pystray.MenuItem("Модель", model_menu),
            pystray.MenuItem(f"Горячая клавиша: {config['hotkey']}", None, enabled=False),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Выход", on_quit)
        ]
    menu = pystray.Menu(*menu_items)
    icon = pystray.Icon("Voice Typing", icon_image, "Voice Typing", menu)
    icon.run()

# ==== Основная программа ====
if __name__ == "__main__":
    try:
        safe_print("🚀 Запуск Voice Typer...")
        
        # Проверяем и настраиваем CUDA
        setup_cuda()
        
        # Загружаем конфигурацию
        load_config()
        
        # Загружаем модель
        if not load_model():
            safe_print("❌ Не удалось загрузить модель. Завершение работы.")
            os._exit(1)
        
        # Настраиваем горячие клавиши в основном потоке
        safe_print("⌨️ Настройка горячих клавиш...")
        setup_hotkey()
        
        safe_print("✅ Приложение запущено успешно. Иконка в трее активна.")
        
        # Запускаем трей в основном потоке (блокирующий вызов)
        run_tray()
            
    except Exception as e:
        safe_print(f"❌ Критическая ошибка: {e}", "error")
        # Создаем файл с ошибкой для диагностики
        try:
            error_file = os.path.join(get_app_path(), "error.log")
            with open(error_file, "w", encoding="utf-8") as f:
                f.write(f"Критическая ошибка: {e}\n")
                import traceback
                f.write(traceback.format_exc())
        except:
            pass
        os._exit(1)