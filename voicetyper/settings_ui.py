"""Tkinter settings, history and diagnostics window (runs in its own thread)."""

from __future__ import annotations

import ctypes
import dataclasses
import logging
import queue
import sys
import threading
import tkinter as tk
import webbrowser
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Optional

from . import paths
from .audio import HAS_SOUNDDEVICE, list_input_devices
from .config import ConfigStore
from .constants import (
    APP_TITLE,
    APP_VERSION,
    COMPUTE_TYPES,
    DEVICES,
    ENGINE_LABELS,
    ENGINE_ORDER,
    LANGUAGES,
    MULTILINGUAL_ENGINES,
    TONE_DECODERS,
    WHISPER_MODELS,
)
from .engines import describe_engines
from .hotkeys import (
    HAS_KEYBOARD,
    HotkeyError,
    capture_keyboard,
    hotkey_from_capture,
    hotkey_name_from_event,
    normalize_hotkey,
)
from .model_packs import BACKUP_URLS, MODELS_RELEASE_URL, PACKS
from .postprocess import preview_text

logger = logging.getLogger(__name__)

_LANG_LABELS = {
    "auto": "Авто",
    "ru": "Русский",
    "en": "English",
    "de": "Deutsch",
    "fr": "Français",
    "es": "Español",
    "it": "Italiano",
    "pt": "Português",
    "zh": "中文",
    "ja": "日本語",
}
_DEVICE_LABELS = {"auto": "Авто", "cpu": "CPU", "cuda": "CUDA"}
_COMPUTE_LABELS = {
    "auto": "Авто",
    "int8": "int8",
    "int8_float16": "int8_float16",
    "float16": "float16",
    "float32": "float32",
}
_TONE_LABELS = {
    "auto": "Авто (по железу)",
    "beam_search": "Beam search (точнее)",
    "greedy": "Greedy (легче)",
}
_HOTKEY_MODE_LABELS = {"hold": "Удержание", "toggle": "Переключение"}
_HOTKEY_MODIFIERS = ("ctrl", "alt", "shift", "windows")
_HOTKEY_MODIFIER_LABELS = {
    "ctrl": "Ctrl",
    "alt": "Alt",
    "shift": "Shift",
    "windows": "Win",
}
_HOTKEY_HINT = "Нажмите «Записать» и удерживайте сочетание — вводить вручную не обязательно."
_CAPTURE_HINT = "Слушаю… Нажмите и удерживайте сочетание. Esc — отмена."
_CAPTURE_IDLE_MS = 8000


class _ScrolledFrame(ttk.Frame):
    """Vertical scroll container that shows its scrollbar only when needed."""

    def __init__(self, parent: tk.Misc):
        super().__init__(parent)
        background = parent.winfo_toplevel().cget("background")
        self._canvas = tk.Canvas(
            self,
            background=background,
            highlightthickness=0,
            borderwidth=0,
            takefocus=0,
        )
        self._scrollbar = ttk.Scrollbar(self, orient="vertical", command=self._canvas.yview)
        self._canvas.configure(yscrollcommand=self._scrollbar.set)
        self._canvas.pack(side="left", fill="both", expand=True)
        self._scrollbar_visible = False
        self.body = ttk.Frame(self._canvas)
        self._window = self._canvas.create_window((0, 0), window=self.body, anchor="nw")
        self.body.bind("<Configure>", self._sync_region)
        self._canvas.bind("<Configure>", self._sync_width)

    def _sync_region(self, _event=None) -> None:
        self._canvas.configure(scrollregion=self._canvas.bbox("all"))
        self._update_scrollbar()

    def _sync_width(self, event) -> None:
        self._canvas.itemconfigure(self._window, width=event.width)
        self._canvas.configure(scrollregion=self._canvas.bbox("all"))
        self._update_scrollbar()

    def _update_scrollbar(self) -> None:
        needed = self.body.winfo_reqheight() > self._canvas.winfo_height()
        if needed == self._scrollbar_visible:
            return
        self._scrollbar_visible = needed
        if needed:
            self._scrollbar.pack(side="right", fill="y")
        else:
            self._scrollbar.pack_forget()

    def scroll_units(self, units: int) -> None:
        if units:
            self._canvas.yview_scroll(units, "units")


def _reverse(mapping: dict[str, str]) -> dict[str, str]:
    return {label: value for value, label in mapping.items()}


class SettingsManager:
    def __init__(self, app, config_store: ConfigStore):
        self._app = app
        self._store = config_store
        self._lock = threading.Lock()
        self._commands: queue.Queue[tuple] = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._open = False

    @property
    def is_open(self) -> bool:
        with self._lock:
            return self._open

    def open(self, tab: Optional[str] = None) -> None:
        with self._lock:
            if self._open:
                self._commands.put(("show", tab))
                return
            self._open = True
            self._thread = threading.Thread(
                target=self._run, args=(tab,), name="settings-ui", daemon=True
            )
            self._thread.start()

    def show(self, tab: Optional[str] = None) -> None:
        self.open(tab)

    def reload_if_open(self) -> None:
        if self.is_open:
            self._commands.put(("reload",))

    def close(self) -> None:
        if self.is_open:
            self._commands.put(("close",))

    def _run(self, initial_tab: Optional[str]) -> None:
        if sys.platform == "win32":
            try:
                ctypes.windll.shcore.SetProcessDpiAwareness(1)
            except Exception:
                pass
        root = None
        try:
            root = tk.Tk()
            root.withdraw()
            window = SettingsWindow(root, self._app, self._store, self._commands)
            if initial_tab:
                window.show_tab(initial_tab)
            root.deiconify()
            root.mainloop()
        except Exception:
            logger.exception("Окно настроек завершилось с ошибкой")
        finally:
            if root is not None:
                try:
                    root.destroy()
                except Exception:
                    pass
            with self._lock:
                self._open = False
            logger.info("Окно настроек закрыто")


class SettingsWindow:
    def __init__(self, root: tk.Tk, app, store: ConfigStore, commands: queue.Queue):
        self.root = root
        self._app = app
        self._store = store
        self._commands = commands
        self._dirty = False
        self._loading = False
        self._tabs: dict[str, int] = {}
        self._scrolled_frames: dict[int, _ScrolledFrame] = {}
        self._scale = 1.0
        self._devices = []
        self._device_labels: list[str] = []
        self._device_values: list[object] = []
        self.vars: dict[str, tk.Variable] = {}
        self._history_entries: dict[str, object] = {}

        self._capturing = False
        self._capture_modifiers: set[str] = set()
        self._capture_events: queue.Queue[tuple[str, str, Optional[int]]] = queue.Queue()
        self._capture_hook = None
        self._capture_poll_after: Optional[str] = None
        self._capture_timeout_after: Optional[str] = None
        self._hotkeys_were_running = False

        self._configure_root()
        self._build()
        self._load_from_config()
        self._poll_commands()

    # -- window shell ----------------------------------------------------
    def _configure_root(self) -> None:
        self.root.title(f"{APP_TITLE} — Настройки")
        self.root.option_add("*Font", "{Segoe UI} 10")
        style = ttk.Style(self.root)
        try:
            if "vista" in style.theme_names():
                style.theme_use("vista")
        except Exception:
            pass
        style.configure("Hint.TLabel", foreground="#666666")
        style.configure("Error.TLabel", foreground="#b00020")
        style.configure("Title.TLabel", font=("Segoe UI", 12, "bold"))
        style.configure("Status.TLabel", foreground="#444444")
        try:
            icon = paths.resource_path("icon.ico")
            if icon.exists():
                self.root.iconbitmap(default=str(icon))
        except Exception:
            pass

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.bind("<Escape>", lambda _event: self._on_close())
        self.root.bind("<MouseWheel>", self._on_mousewheel, add="+")
        self.root.bind("<Destroy>", self._on_destroy, add="+")

        self.root.update_idletasks()
        self._scale = max(1.0, self.root.winfo_fpixels("1i") / 96.0)
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        width = min(int(820 * self._scale), int(screen_w * 0.9))
        height = min(int(780 * self._scale), int(screen_h * 0.85))
        self.root.minsize(min(int(680 * self._scale), width), min(int(500 * self._scale), height))
        x = max(0, (screen_w - width) // 2)
        y = max(0, (screen_h - height) // 2)
        self.root.geometry(f"{width}x{height}+{x}+{y}")
        self.root.attributes("-topmost", True)
        self.root.after(250, lambda: self.root.attributes("-topmost", False))

    def _build(self) -> None:
        outer = ttk.Frame(self.root, padding=12)
        outer.pack(fill="both", expand=True)

        header = ttk.Frame(outer)
        header.pack(fill="x", pady=(0, 8))
        ttk.Label(header, text=APP_TITLE, style="Title.TLabel").pack(side="left")
        ttk.Label(
            header,
            text=f"v{APP_VERSION} · локальное распознавание речи",
            style="Hint.TLabel",
        ).pack(side="left", padx=(10, 0))

        # Pin the action row to the window bottom so it stays visible even when
        # the tab content is taller than the window.
        footer = ttk.Frame(outer)
        footer.pack(side="bottom", fill="x", pady=(8, 0))
        ttk.Button(footer, text="Закрыть", command=self._on_close).pack(side="right")
        ttk.Button(footer, text="Применить", command=self._apply).pack(
            side="right", padx=(0, 8)
        )
        self.status_var = tk.StringVar(value="")
        ttk.Label(footer, textvariable=self.status_var, style="Status.TLabel").pack(
            side="left", fill="x", expand=True
        )
        ttk.Separator(outer).pack(side="bottom", fill="x", pady=(10, 0))

        self.notebook = ttk.Notebook(outer)
        self.notebook.pack(side="top", fill="both", expand=True)

        self._tabs["main"] = self._build_main_tab()
        self._tabs["insertion"] = self._build_insertion_tab()
        self._tabs["privacy"] = self._build_privacy_tab()
        self._tabs["history"] = self._build_history_tab()
        self._tabs["diagnostics"] = self._build_diagnostics_tab()
        self._tabs["about"] = self._build_about_tab()
        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)

    def _tab(self, title: str) -> ttk.Frame:
        container = ttk.Frame(self.notebook)
        self.notebook.add(container, text=title)
        frame = ttk.Frame(container, padding=14)
        frame.pack(fill="both", expand=True)
        return frame

    def _scrolled_tab(self, title: str) -> ttk.Frame:
        container = ttk.Frame(self.notebook)
        self.notebook.add(container, text=title)
        scrolled = _ScrolledFrame(container)
        scrolled.pack(fill="both", expand=True)
        self._scrolled_frames[self.notebook.index(container)] = scrolled
        frame = ttk.Frame(scrolled.body, padding=14)
        frame.pack(fill="both", expand=True)
        return frame

    def _build_main_tab(self) -> int:
        frame = self._scrolled_tab("Основное")

        hotkey_box = ttk.LabelFrame(frame, text="Клавиша и микрофон", padding=10)
        hotkey_box.pack(fill="x", pady=(0, 10))
        hotkey_box.columnconfigure(1, weight=1)

        self.vars["hotkey"] = tk.StringVar()
        hotkey_row = ttk.Frame(hotkey_box)
        hotkey_row.columnconfigure(0, weight=1)
        ttk.Entry(hotkey_row, textvariable=self.vars["hotkey"]).grid(
            row=0, column=0, sticky="ew"
        )
        self.capture_button = ttk.Button(
            hotkey_row, text="Записать…", width=12, command=self._toggle_capture
        )
        self.capture_button.grid(row=0, column=1, padx=(6, 0))
        self.hotkey_hint = ttk.Label(
            hotkey_row,
            text=_HOTKEY_HINT,
            style="Hint.TLabel",
            wraplength=int(480 * self._scale),
        )
        self.hotkey_hint.grid(row=1, column=0, columnspan=2, sticky="w", pady=(4, 0))
        self._row(hotkey_box, 0, "Горячая клавиша", hotkey_row)
        if not HAS_KEYBOARD:
            self.capture_button.state(["disabled"])
            self._set_hotkey_hint(
                "Запись с клавиатуры недоступна: библиотека keyboard не установлена.",
                error=True,
            )
        self.vars["hotkey_mode"] = tk.StringVar()
        self._combo_row(
            hotkey_box,
            1,
            "Режим",
            self.vars["hotkey_mode"],
            list(_HOTKEY_MODE_LABELS.values()),
        )
        ttk.Label(
            hotkey_box,
            text="Удержание — запись пока клавиши нажаты. Переключение — нажатие старт/стоп.",
            style="Hint.TLabel",
            wraplength=int(480 * self._scale),
        ).grid(row=2, column=1, sticky="w", pady=(0, 6))

        self.vars["input_device"] = tk.StringVar()
        device_frame = ttk.Frame(hotkey_box)
        device_frame.grid(row=3, column=1, sticky="ew", pady=2)
        device_frame.columnconfigure(0, weight=1)
        self.device_combo = ttk.Combobox(
            device_frame, textvariable=self.vars["input_device"], state="readonly"
        )
        self.device_combo.bind("<MouseWheel>", self._on_form_wheel)
        self.device_combo.grid(row=0, column=0, sticky="ew")
        ttk.Button(device_frame, text="Обновить", width=10, command=self._refresh_devices).grid(
            row=0, column=1, padx=(6, 0)
        )
        ttk.Label(hotkey_box, text="Микрофон", style="Hint.TLabel").grid(
            row=3, column=0, sticky="w", padx=(0, 10)
        )
        self.vars["max_recording_seconds"] = tk.StringVar()
        self._spin_row(hotkey_box, 4, "Максимум записи, с", self.vars["max_recording_seconds"], 10, 3600)

        engine_box = ttk.LabelFrame(frame, text="Распознавание", padding=10)
        engine_box.pack(fill="x")
        engine_box.columnconfigure(1, weight=1)

        self.vars["language"] = tk.StringVar()
        self.language_combo = self._combo_row(
            engine_box,
            0,
            "Язык",
            self.vars["language"],
            [_LANG_LABELS[code] for code in LANGUAGES],
        )
        availability = describe_engines()
        self._engine_labels: list[str] = []
        self._engine_values: list[str] = []
        for code in ENGINE_ORDER:
            label = ENGINE_LABELS.get(code, code)
            info = availability.get(code)
            if code != "auto" and info and not info.installed:
                label += " (не установлен)"
            self._engine_labels.append(label)
            self._engine_values.append(code)
        self.vars["model_type"] = tk.StringVar()
        self.engine_combo = self._combo_row(
            engine_box, 1, "Движок", self.vars["model_type"], self._engine_labels
        )
        self.engine_combo.bind("<<ComboboxSelected>>", self._on_engine_selected)

        self.vars["whisper_model"] = tk.StringVar()
        self._combo_row(engine_box, 2, "Модель Whisper", self.vars["whisper_model"], list(WHISPER_MODELS))
        self.vars["tone_decoder"] = tk.StringVar()
        self._combo_row(
            engine_box,
            3,
            "Декодер T-one",
            self.vars["tone_decoder"],
            [_TONE_LABELS[value] for value in TONE_DECODERS],
        )
        self.vars["device"] = tk.StringVar()
        self._combo_row(
            engine_box,
            4,
            "Устройство",
            self.vars["device"],
            [_DEVICE_LABELS[value] for value in DEVICES],
        )
        self.vars["compute_type"] = tk.StringVar()
        self._combo_row(
            engine_box,
            5,
            "Compute type",
            self.vars["compute_type"],
            [_COMPUTE_LABELS[value] for value in COMPUTE_TYPES],
        )
        self.vars["beam_size"] = tk.StringVar()
        self._spin_row(engine_box, 6, "Beam size (Whisper)", self.vars["beam_size"], 1, 10)
        self.vars["vad_filter"] = tk.BooleanVar()
        ttk.Checkbutton(
            engine_box,
            text="VAD-фильтр Whisper (убирает тишину)",
            variable=self.vars["vad_filter"],
        ).grid(row=7, column=1, sticky="w", pady=2)
        ttk.Label(
            engine_box,
            text=(
                "«Авто» выбирает GigaAM/T-one для русского. GigaAM и T-one "
                "распознают только русский, поэтому выбор языка доступен лишь "
                "для Faster-Whisper."
            ),
            style="Hint.TLabel",
            wraplength=int(480 * self._scale),
        ).grid(row=8, column=1, sticky="w", pady=(4, 0))

        self._register_dirty_traces()
        return 0

    def _build_insertion_tab(self) -> int:
        frame = self._scrolled_tab("Вставка")
        box = ttk.LabelFrame(frame, text="Способ вставки", padding=10)
        box.pack(fill="x", pady=(0, 10))

        self.vars["insert_mode"] = tk.StringVar()
        for value, label, hint in (
            ("paste", "Вставить через Ctrl+V", "Буфер обмена восстанавливается после вставки"),
            ("clipboard", "Только скопировать", "Текст остаётся в буфере обмена, вставьте вручную"),
            ("type", "Напечатать", "Медленнее и хуже для длинного текста"),
        ):
            ttk.Radiobutton(
                box, text=label, value=value, variable=self.vars["insert_mode"]
            ).pack(anchor="w")
            ttk.Label(box, text=hint, style="Hint.TLabel").pack(anchor="w", padx=(22, 0), pady=(0, 4))

        options = ttk.LabelFrame(frame, text="Параметры", padding=10)
        options.pack(fill="x", pady=(0, 10))
        options.columnconfigure(1, weight=1)
        self.vars["restore_clipboard"] = tk.BooleanVar()
        ttk.Checkbutton(
            options,
            text="Восстанавливать текстовый буфер обмена",
            variable=self.vars["restore_clipboard"],
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=2)
        self.vars["restore_target_window"] = tk.BooleanVar()
        ttk.Checkbutton(
            options,
            text="Вставлять в окно, активное в начале записи",
            variable=self.vars["restore_target_window"],
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=2)
        self.vars["paste_delay_ms"] = tk.StringVar()
        self._spin_row(options, 2, "Задержка перед Ctrl+V, мс", self.vars["paste_delay_ms"], 0, 2000)
        self.vars["trim_silence"] = tk.BooleanVar()
        ttk.Checkbutton(
            options, text="Обрезать тишину в начале и конце", variable=self.vars["trim_silence"]
        ).grid(row=3, column=0, columnspan=2, sticky="w", pady=2)
        self.vars["capitalize"] = tk.BooleanVar()
        ttk.Checkbutton(
            options, text="Начинать предложения с заглавной буквы", variable=self.vars["capitalize"]
        ).grid(row=4, column=0, columnspan=2, sticky="w", pady=2)
        self.vars["remove_fillers"] = tk.BooleanVar()
        ttk.Checkbutton(
            options, text="Удалять слова-заполнители (ээ, эм, uh, um, «ну,», «это самое»)", variable=self.vars["remove_fillers"]
        ).grid(row=5, column=0, columnspan=2, sticky="w", pady=2)

        exclude = ttk.LabelFrame(frame, text="Фразы-исключения", padding=10)
        exclude.pack(fill="both", expand=True)
        ttk.Label(
            exclude,
            text="Если распознанный текст состоит из этой фразы (возможно, повторённой), вставка отменяется.\n"
            "Одна фраза на строку. Удобно для типичных галлюцинаций моделей.",
            style="Hint.TLabel",
        ).pack(anchor="w", pady=(0, 6))
        self.exclude_text = tk.Text(exclude, height=7, wrap="word", undo=True)
        self.exclude_text.pack(fill="both", expand=True)
        self._register_dirty_traces()
        return 1

    def _build_privacy_tab(self) -> int:
        frame = self._scrolled_tab("Приватность")
        box = ttk.LabelFrame(frame, text="История", padding=10)
        box.pack(fill="x", pady=(0, 10))
        self.vars["history_enabled"] = tk.BooleanVar()
        ttk.Checkbutton(
            box, text="Сохранять распознанный текст в локальную историю", variable=self.vars["history_enabled"]
        ).pack(anchor="w", pady=2)
        row = ttk.Frame(box)
        row.pack(anchor="w", fill="x", pady=2)
        ttk.Label(row, text="Хранить записей:").pack(side="left")
        self.vars["history_limit"] = tk.StringVar()
        limit_spin = ttk.Spinbox(
            row, from_=0, to=10000, width=8, textvariable=self.vars["history_limit"]
        )
        limit_spin.bind("<MouseWheel>", self._on_form_wheel)
        limit_spin.pack(side="left", padx=8)
        ttk.Button(row, text="Очистить историю…", command=self._clear_history).pack(side="left")

        privacy = ttk.LabelFrame(frame, text="Логи", padding=10)
        privacy.pack(fill="x", pady=(0, 10))
        self.vars["log_transcripts"] = tk.BooleanVar()
        ttk.Checkbutton(
            privacy,
            text="Писать полный текст распознавания в лог (по умолчанию только длина и хеш)",
            variable=self.vars["log_transcripts"],
        ).pack(anchor="w", pady=2)
        ttk.Label(
            privacy,
            text="Полный текст всегда доступен только в локальной истории. "
            "Логи находятся в папке данных и ротируются автоматически.",
            style="Hint.TLabel",
            wraplength=650,
        ).pack(anchor="w", pady=(4, 0))

        notifications = ttk.LabelFrame(frame, text="Поведение", padding=10)
        notifications.pack(fill="x")
        self.vars["notifications"] = tk.BooleanVar()
        ttk.Checkbutton(
            notifications, text="Показывать уведомления трея", variable=self.vars["notifications"]
        ).pack(anchor="w", pady=2)
        self.vars["streaming_preview"] = tk.BooleanVar()
        ttk.Checkbutton(
            notifications,
            text="Показывать предварительный текст T-one во время записи",
            variable=self.vars["streaming_preview"],
        ).pack(anchor="w", pady=2)
        self.vars["autostart"] = tk.BooleanVar()
        ttk.Checkbutton(
            notifications, text="Запускать Voice Typer при входе в Windows", variable=self.vars["autostart"]
        ).pack(anchor="w", pady=2)
        self.vars["single_instance"] = tk.BooleanVar()
        ttk.Checkbutton(
            notifications,
            text="Не запускать второй экземпляр (требуется перезапуск)",
            variable=self.vars["single_instance"],
        ).pack(anchor="w", pady=2)

        self._register_dirty_traces()
        return 2

    def _build_history_tab(self) -> int:
        frame = self._tab("История")
        top = ttk.Frame(frame)
        top.pack(fill="x", pady=(0, 8))
        self.history_query = tk.StringVar()
        search = ttk.Entry(top, textvariable=self.history_query)
        search.pack(side="left", fill="x", expand=True)
        search.bind("<Return>", lambda _event: self._refresh_history())
        ttk.Button(top, text="Найти", command=self._refresh_history).pack(side="left", padx=(6, 0))
        ttk.Button(top, text="Обновить", command=self._refresh_history).pack(side="left", padx=(6, 0))

        columns = ("date", "engine", "duration", "text")
        tree_frame = ttk.Frame(frame)
        tree_frame.pack(fill="both", expand=True)
        self.history_tree = ttk.Treeview(tree_frame, columns=columns, show="headings", height=12)
        for column, title, width, anchor in (
            ("date", "Дата", 130, "w"),
            ("engine", "Движок", 110, "w"),
            ("duration", "Длит.", 60, "e"),
            ("text", "Текст", 380, "w"),
        ):
            self.history_tree.heading(column, text=title)
            self.history_tree.column(column, width=width, anchor=anchor, stretch=(column == "text"))
        tree_scroll = ttk.Scrollbar(tree_frame, orient="vertical", command=self.history_tree.yview)
        self.history_tree.configure(yscrollcommand=tree_scroll.set)
        self.history_tree.pack(side="left", fill="both", expand=True)
        tree_scroll.pack(side="right", fill="y")
        self.history_tree.bind("<<TreeviewSelect>>", self._on_history_select)

        preview_box = ttk.LabelFrame(frame, text="Выбранная запись", padding=6)
        preview_box.pack(fill="both", expand=False, pady=(8, 6))
        self.history_preview = tk.Text(preview_box, height=5, wrap="word", state="disabled")
        self.history_preview.pack(fill="both", expand=True)

        buttons = ttk.Frame(frame)
        buttons.pack(fill="x")
        ttk.Button(buttons, text="Копировать", command=self._copy_history).pack(side="left")
        ttk.Button(buttons, text="Удалить", command=self._delete_history).pack(side="left", padx=(6, 0))
        ttk.Button(buttons, text="Очистить всё…", command=self._clear_history).pack(side="left", padx=(6, 0))
        ttk.Button(buttons, text="Экспорт CSV…", command=self._export_history).pack(side="left", padx=(6, 0))
        ttk.Label(
            frame,
            text="История хранится локально в SQLite и не отправляется в сеть.",
            style="Hint.TLabel",
        ).pack(anchor="w", pady=(6, 0))
        return 3

    def _build_diagnostics_tab(self) -> int:
        frame = self._tab("Диагностика")
        top = ttk.Frame(frame)
        top.pack(fill="x", pady=(0, 8))
        ttk.Button(top, text="Обновить", command=self._refresh_diagnostics).pack(side="left")
        ttk.Button(top, text="Копировать", command=self._copy_diagnostics).pack(side="left", padx=(6, 0))
        ttk.Button(top, text="Сохранить…", command=self._save_diagnostics).pack(side="left", padx=(6, 0))
        ttk.Button(top, text="Открыть логи", command=self._app.open_logs).pack(side="right")
        ttk.Button(top, text="Папка моделей", command=self._app.open_models_dir).pack(side="right", padx=(0, 6))

        text_frame = ttk.Frame(frame)
        text_frame.pack(fill="both", expand=True)
        self.diagnostics_text = tk.Text(text_frame, wrap="word", state="disabled")
        scroll = ttk.Scrollbar(text_frame, orient="vertical", command=self.diagnostics_text.yview)
        self.diagnostics_text.configure(yscrollcommand=scroll.set)
        self.diagnostics_text.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        self._refresh_diagnostics()
        return 4

    def _build_about_tab(self) -> int:
        frame = self._tab("О программе")
        text_frame = ttk.Frame(frame)
        text_frame.pack(fill="both", expand=True)
        text = tk.Text(
            text_frame,
            wrap="word",
            relief="flat",
            background=self.root.cget("background"),
            cursor="arrow",
        )
        scroll = ttk.Scrollbar(text_frame, orient="vertical", command=text.yview)
        text.configure(yscrollcommand=scroll.set)
        text.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        text.tag_configure("heading", font=("Segoe UI", 10, "bold"))
        text.tag_configure("mono", font=("Consolas", 9))
        text.tag_configure("link", foreground="#0a66c2", underline=True)
        text.tag_bind("link", "<Button-1>", self._open_about_link)
        text.tag_bind("link", "<Enter>", lambda _event: text.configure(cursor="hand2"))
        text.tag_bind("link", "<Leave>", lambda _event: text.configure(cursor="arrow"))
        self._fill_about_text(text)
        text.configure(state="disabled")
        return 5

    def _fill_about_text(self, text: tk.Text) -> None:
        def para(value: str = "", tag=None) -> None:
            text.insert("end", value + "\n", tag or ())

        def link(label: str, url: str) -> None:
            text.insert("end", label, ("link", url))

        models_root = paths.models_dir()

        para(f"{APP_TITLE} {APP_VERSION}", "heading")
        para(
            "Локальное распознавание речи для Windows. Аудио не покидает компьютер; "
            "сеть используется только для загрузки моделей."
        )
        para()
        para("Движки:", "heading")
        para("  • GigaAM v3 — русский язык, ONNX, лучшее качество на CPU.")
        para("  • T-one — русский язык, потоковое распознавание, лёгкая модель.")
        para("  • Faster-Whisper — многоязычный, CUDA при наличии NVIDIA GPU.")
        para()
        para("Горячая клавиша по умолчанию: Ctrl+Alt+Space. В режиме удержания "
             "запись идёт, пока клавиши нажаты; в режиме переключения — до повторного нажатия.")
        para()
        para("Рекомендуемая установка моделей", "heading")
        para(
            "Готовые архивы моделей можно скачать одним файлом со страницы релиза "
            "на GitHub — это быстрее и надёжнее, чем пофайловая загрузка из Hugging Face."
        )
        text.insert("end", "Страница релиза моделей: ")
        link(MODELS_RELEASE_URL, MODELS_RELEASE_URL)
        text.insert("end", "\n\n")
        para(f"Распакуйте архивы в папку моделей: {models_root}")
        para("Файлы должны лежать прямо в указанных подпапках:")
        for pack in PACKS.values():
            text.insert("end", "  • ")
            text.insert("end", f"{pack.title} → {models_root / Path(*pack.folder)}", "mono")
            text.insert("end", "\n")
        para()
        para("Запасные источники (если GitHub недоступен):", "heading")
        for pack in PACKS.values():
            text.insert("end", f"  • {pack.title}: ")
            urls = BACKUP_URLS.get(pack.key, ())
            for index, url in enumerate(urls):
                if index:
                    text.insert("end", " · ")
                link(url, url)
            text.insert("end", "\n")
        para()
        para("Модели также скачиваются автоматически при первом запуске движка.")
        para()
        para(f"Данные: {paths.data_dir()}")
        para(f"Модели: {models_root}")
        para(f"Логи: {paths.logs_dir()}")

    def _open_about_link(self, event) -> str:
        widget = event.widget
        try:
            index = widget.index(f"@{event.x},{event.y}")
        except tk.TclError:
            return "break"
        for tag in widget.tag_names(index):
            if tag.startswith(("http://", "https://")):
                webbrowser.open(tag)
                break
        return "break"

    # -- form helpers ----------------------------------------------------
    def _row(self, parent, row: int, label: str, widget) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(0, 10), pady=4)
        widget.grid(row=row, column=1, sticky="ew", pady=4)

    def _combo_row(self, parent, row: int, label: str, variable, values) -> ttk.Combobox:
        combo = ttk.Combobox(parent, textvariable=variable, values=values, state="readonly")
        combo.bind("<MouseWheel>", self._on_form_wheel)
        self._row(parent, row, label, combo)
        return combo

    def _engine_code(self) -> str:
        return dict(zip(self._engine_labels, self._engine_values, strict=True)).get(
            self.vars["model_type"].get(), self._store.get().model_type
        )

    def _update_language_options(self) -> None:
        if self._engine_code() in MULTILINGUAL_ENGINES:
            self.language_combo.configure(
                values=[_LANG_LABELS[code] for code in LANGUAGES], state="readonly"
            )
        else:
            self.language_combo.configure(values=["Русский"], state="disabled")
            if self.vars["language"].get() != "Русский":
                self.vars["language"].set("Русский")

    def _on_engine_selected(self, _event=None) -> None:
        self._update_language_options()

    def _spin_row(self, parent, row: int, label: str, variable, minimum: int, maximum: int):
        spin = ttk.Spinbox(parent, from_=minimum, to=maximum, textvariable=variable, width=10)
        spin.bind("<MouseWheel>", self._on_form_wheel)
        self._row(parent, row, label, spin)
        return spin

    def _register_dirty_traces(self) -> None:
        for variable in self.vars.values():
            try:
                variable.trace_add("write", self._mark_dirty)
            except Exception:
                pass
        if hasattr(self, "exclude_text"):
            self.exclude_text.bind("<<Modified>>", self._on_text_modified)

    def _mark_dirty(self, *_args) -> None:
        if not self._loading:
            self._dirty = True
            self.status_var.set("Есть несохранённые изменения")

    def _on_text_modified(self, _event=None) -> None:
        # edit_modified(False) itself emits <<Modified>>, so ignore the event
        # that arrives after the flag has already been reset.
        if not self.exclude_text.edit_modified():
            return
        if not self._loading:
            self._dirty = True
            self.status_var.set("Есть несохранённые изменения")
        self.exclude_text.edit_modified(False)

    # -- loading ---------------------------------------------------------
    def _load_from_config(self) -> None:
        cfg = self._store.get()
        self._loading = True
        try:
            self._refresh_devices()
            self.vars["hotkey"].set(cfg.hotkey)
            self.vars["hotkey_mode"].set(_HOTKEY_MODE_LABELS.get(cfg.hotkey_mode, "Удержание"))
            self.vars["language"].set(_LANG_LABELS.get(cfg.language, cfg.language))
            self.vars["model_type"].set(
                dict(zip(self._engine_values, self._engine_labels, strict=True)).get(
                    cfg.model_type, ENGINE_LABELS.get(cfg.model_type, cfg.model_type)
                )
            )
            self._update_language_options()
            self.vars["whisper_model"].set(cfg.whisper_model)
            self.vars["tone_decoder"].set(_TONE_LABELS.get(cfg.tone_decoder, cfg.tone_decoder))
            self.vars["device"].set(_DEVICE_LABELS.get(cfg.device, cfg.device))
            self.vars["compute_type"].set(_COMPUTE_LABELS.get(cfg.compute_type, cfg.compute_type))
            self.vars["beam_size"].set(str(cfg.beam_size))
            self.vars["vad_filter"].set(cfg.vad_filter)
            self.vars["max_recording_seconds"].set(str(cfg.max_recording_seconds))
            self.vars["insert_mode"].set(cfg.insert_mode)
            self.vars["restore_clipboard"].set(cfg.restore_clipboard)
            self.vars["restore_target_window"].set(cfg.restore_target_window)
            self.vars["paste_delay_ms"].set(str(cfg.paste_delay_ms))
            self.vars["trim_silence"].set(cfg.trim_silence)
            self.vars["capitalize"].set(cfg.capitalize)
            self.vars["remove_fillers"].set(cfg.remove_fillers)
            self.vars["history_enabled"].set(cfg.history_enabled)
            self.vars["history_limit"].set(str(cfg.history_limit))
            self.vars["log_transcripts"].set(cfg.log_transcripts)
            self.vars["notifications"].set(cfg.notifications)
            self.vars["streaming_preview"].set(cfg.streaming_preview)
            self.vars["autostart"].set(cfg.autostart)
            self.vars["single_instance"].set(cfg.single_instance)
            self.exclude_text.delete("1.0", "end")
            self.exclude_text.insert("1.0", "\n".join(cfg.exclude_phrases))
            self.exclude_text.edit_modified(False)
            self._select_input_device(cfg.input_device)
            self._dirty = False
            self.status_var.set("")
        finally:
            self._loading = False

    def _refresh_devices(self) -> None:
        self._devices = list_input_devices()
        self._device_labels = ["Системный по умолчанию"]
        self._device_values = [None]
        for device in self._devices:
            self._device_labels.append(f"{device.index}: {device.label()}")
            self._device_values.append(device.index)
        if not HAS_SOUNDDEVICE:
            self._device_labels = ["sounddevice не установлен"]
            self._device_values = [None]
        self.device_combo.configure(values=self._device_labels)
        if not self._device_labels:
            self.device_combo.set("")
            return
        current = self.vars.get("input_device")
        if current is None:
            self.vars["input_device"] = tk.StringVar()
            self.device_combo.configure(textvariable=self.vars["input_device"])
        if self.vars["input_device"].get() not in self._device_labels:
            self.device_combo.set(self._device_labels[0])

    def _select_input_device(self, value) -> None:
        try:
            index = self._device_values.index(value)
        except ValueError:
            index = 0
        self.device_combo.set(self._device_labels[index])

    # -- hotkey capture --------------------------------------------------
    def _set_hotkey_hint(self, text: str, *, error: bool = False) -> None:
        self.hotkey_hint.configure(
            text=text, style="Error.TLabel" if error else "Hint.TLabel"
        )

    def _toggle_capture(self) -> None:
        if self._capturing:
            self._finish_capture(None, "Запись комбинации отменена")
        else:
            self._start_capture()

    def _start_capture(self) -> None:
        if not HAS_KEYBOARD:
            return
        self._capturing = True
        self._capture_modifiers.clear()
        self._drain_capture_events()
        self.capture_button.configure(text="Отмена")
        self._set_hotkey_hint(_CAPTURE_HINT)
        hotkeys = getattr(self._app, "hotkeys", None)
        self._hotkeys_were_running = bool(hotkeys is not None and hotkeys.running)
        if self._hotkeys_were_running:
            try:
                hotkeys.close()
            except Exception:
                logger.debug("Не удалось приостановить горячую клавишу", exc_info=True)
                self._hotkeys_were_running = False
        try:
            self._capture_hook = capture_keyboard(self._on_key_event)
        except Exception as exc:
            self._capturing = False
            self.capture_button.configure(text="Записать…")
            self._set_hotkey_hint(f"Не удалось перехватить клавиатуру: {exc}", error=True)
            self._resume_hotkeys()
            return
        self._schedule_capture_timeout()
        self._poll_capture()

    def _on_key_event(self, event) -> bool:
        # Runs on the keyboard hook thread; keep it cheap and swallow the key.
        try:
            self._capture_events.put_nowait(
                (event.event_type, event.name, getattr(event, "scan_code", None))
            )
        except Exception:
            pass
        return False

    def _poll_capture(self) -> None:
        self._capture_poll_after = None
        if not self._capturing:
            return
        try:
            while self._capturing:
                event_type, name, scan_code = self._capture_events.get_nowait()
                self._handle_capture_event(event_type, name, scan_code)
        except queue.Empty:
            pass
        if self._capturing:
            self._capture_poll_after = self.root.after(30, self._poll_capture)

    def _handle_capture_event(
        self, event_type: str, name: str, scan_code: Optional[int] = None
    ) -> None:
        if event_type == "up":
            self._capture_modifiers.discard(hotkey_name_from_event(name))
            self._refresh_capture_hint()
            return
        name = (name or "").strip().lower()
        if name == "esc":
            self._finish_capture(None, "Запись комбинации отменена")
            return
        modifier = hotkey_name_from_event(name)
        if modifier in _HOTKEY_MODIFIERS:
            self._capture_modifiers.add(modifier)
            self._refresh_capture_hint()
            self._schedule_capture_timeout()
            return
        try:
            combo = hotkey_from_capture(self._capture_modifiers, name, scan_code)
        except HotkeyError as exc:
            self._set_hotkey_hint(str(exc), error=True)
            self._schedule_capture_timeout()
            return
        self._finish_capture(combo, f"Горячая клавиша: {combo}")

    def _refresh_capture_hint(self) -> None:
        held = [name for name in _HOTKEY_MODIFIERS if name in self._capture_modifiers]
        if not held:
            self._set_hotkey_hint(_CAPTURE_HINT)
            return
        labels = " + ".join(_HOTKEY_MODIFIER_LABELS[name] for name in held)
        self._set_hotkey_hint(f"{labels} — добавьте обычную клавишу, например Space.")

    def _finish_capture(self, combo: Optional[str], message: str) -> None:
        if not self._capturing:
            return
        self._capturing = False
        self._cancel_capture_timeout()
        if self._capture_poll_after is not None:
            try:
                self.root.after_cancel(self._capture_poll_after)
            except Exception:
                pass
            self._capture_poll_after = None
        hook = self._capture_hook
        self._capture_hook = None
        if hook is not None:
            try:
                hook()
            except Exception:
                logger.debug("Не удалось снять перехват клавиатуры", exc_info=True)
        self._drain_capture_events()
        self.capture_button.configure(text="Записать…")
        if combo:
            self.vars["hotkey"].set(combo)
            self._set_hotkey_hint(f"Записано: {combo} — нажмите «Применить»")
        else:
            self._set_hotkey_hint(_HOTKEY_HINT)
        if message:
            self.status_var.set(message)
        self._resume_hotkeys()

    def _drain_capture_events(self) -> None:
        while True:
            try:
                self._capture_events.get_nowait()
            except queue.Empty:
                return

    def _schedule_capture_timeout(self) -> None:
        self._cancel_capture_timeout()
        self._capture_timeout_after = self.root.after(
            _CAPTURE_IDLE_MS, self._on_capture_timeout
        )

    def _on_capture_timeout(self) -> None:
        self._capture_timeout_after = None
        self._finish_capture(None, "Запись комбинации отменена: нет нажатий")

    def _cancel_capture_timeout(self) -> None:
        if self._capture_timeout_after is not None:
            try:
                self.root.after_cancel(self._capture_timeout_after)
            except Exception:
                pass
            self._capture_timeout_after = None

    def _resume_hotkeys(self) -> None:
        if not self._hotkeys_were_running:
            return
        self._hotkeys_were_running = False
        if getattr(self._app, "is_shutting_down", False):
            return
        try:
            self._app.hotkeys.start()
        except Exception as exc:
            logger.exception("Не удалось вернуть глобальную горячую клавишу")
            try:
                self._set_hotkey_hint(
                    f"Горячая клавиша не восстановлена: {exc}", error=True
                )
            except Exception:
                pass

    def _on_form_wheel(self, event):
        # Widget-level wheel bindings run before the class binding, so this
        # keeps the page scrolling instead of silently editing combobox/spinbox values.
        self._scroll_active(event)
        return "break"

    def _on_mousewheel(self, event):
        if isinstance(
            event.widget, (ttk.Combobox, ttk.Spinbox, tk.Text, tk.Listbox, ttk.Treeview)
        ):
            return None
        self._scroll_active(event)
        return "break"

    def _scroll_active(self, event) -> None:
        if self._capturing or not getattr(event, "delta", 0):
            return
        try:
            index = self.notebook.index(self.notebook.select())
        except Exception:
            return
        scrolled = self._scrolled_frames.get(index)
        if scrolled is None:
            return
        scrolled.scroll_units(-1 if event.delta > 0 else 1)

    def _on_destroy(self, event) -> None:
        if event.widget is not self.root:
            return
        if self._capturing:
            self._capturing = False
            hook = self._capture_hook
            self._capture_hook = None
            if hook is not None:
                try:
                    hook()
                except Exception:
                    pass
            self._resume_hotkeys()

    # -- apply -----------------------------------------------------------
    def _int_value(self, name: str, label: str, minimum: int, maximum: int) -> int:
        raw = str(self.vars[name].get()).strip()
        try:
            value = int(raw)
        except ValueError as exc:
            raise ValueError(f"«{label}» должно быть целым числом") from exc
        if not minimum <= value <= maximum:
            raise ValueError(f"«{label}» должно быть от {minimum} до {maximum}")
        return value

    def _apply(self) -> None:
        if self._capturing:
            self._finish_capture(None, "")
        try:
            hotkey, _keys = normalize_hotkey(self.vars["hotkey"].get())
            base = self._store.get()
            engine = dict(zip(self._engine_labels, self._engine_values, strict=True)).get(
                self.vars["model_type"].get(), base.model_type
            )
            language = (
                _reverse(_LANG_LABELS).get(self.vars["language"].get(), "ru")
                if engine in MULTILINGUAL_ENGINES
                else "ru"
            )
            tone = _reverse(_TONE_LABELS).get(self.vars["tone_decoder"].get(), base.tone_decoder)
            device = _reverse(_DEVICE_LABELS).get(self.vars["device"].get(), base.device)
            compute = _reverse(_COMPUTE_LABELS).get(
                self.vars["compute_type"].get(), base.compute_type
            )
            mode = _reverse(_HOTKEY_MODE_LABELS).get(
                self.vars["hotkey_mode"].get(), base.hotkey_mode
            )
            try:
                device_index = self._device_values[self._device_labels.index(self.vars["input_device"].get())]
            except (ValueError, IndexError):
                device_index = base.input_device
            phrases = [
                line.strip()
                for line in self.exclude_text.get("1.0", "end").splitlines()
                if line.strip()
            ]
            if len(phrases) > 100:
                raise ValueError("Слишком много фраз-исключений (максимум 100)")
            cfg = dataclasses.replace(
                base,
                hotkey=hotkey,
                hotkey_mode=mode,
                language=language,
                model_type=engine,
                whisper_model=self.vars["whisper_model"].get(),
                tone_decoder=tone,
                device=device,
                compute_type=compute,
                beam_size=self._int_value("beam_size", "Beam size", 1, 10),
                vad_filter=bool(self.vars["vad_filter"].get()),
                max_recording_seconds=self._int_value(
                    "max_recording_seconds", "Максимум записи", 10, 3600
                ),
                input_device=device_index,
                insert_mode=self.vars["insert_mode"].get(),
                restore_clipboard=bool(self.vars["restore_clipboard"].get()),
                restore_target_window=bool(self.vars["restore_target_window"].get()),
                paste_delay_ms=self._int_value("paste_delay_ms", "Задержка вставки", 0, 2000),
                trim_silence=bool(self.vars["trim_silence"].get()),
                capitalize=bool(self.vars["capitalize"].get()),
                remove_fillers=bool(self.vars["remove_fillers"].get()),
                exclude_phrases=phrases,
                history_enabled=bool(self.vars["history_enabled"].get()),
                history_limit=self._int_value("history_limit", "Лимит истории", 0, 10000),
                log_transcripts=bool(self.vars["log_transcripts"].get()),
                notifications=bool(self.vars["notifications"].get()),
                streaming_preview=bool(self.vars["streaming_preview"].get()),
                autostart=bool(self.vars["autostart"].get()),
                single_instance=bool(self.vars["single_instance"].get()),
            )
        except (ValueError, HotkeyError) as exc:
            messagebox.showerror("Проверьте настройки", str(exc), parent=self.root)
            return

        try:
            ok, message = self._app.apply_config(cfg)
        except Exception as exc:
            logger.exception("Ошибка применения настроек")
            messagebox.showerror("Ошибка", str(exc), parent=self.root)
            return
        if not ok:
            messagebox.showerror("Не удалось применить настройки", message, parent=self.root)
            return
        self._dirty = False
        self.status_var.set(message)
        self._load_from_config()
        if HAS_KEYBOARD:
            self._set_hotkey_hint(_HOTKEY_HINT)
        self.status_var.set(message)

    # -- history ---------------------------------------------------------
    def _refresh_history(self) -> None:
        history = getattr(self._app, "history", None)
        if history is None:
            return
        rows = history.list(limit=200, query=self.history_query.get())
        self._history_entries = {}
        for item in self.history_tree.get_children():
            self.history_tree.delete(item)
        for entry in rows:
            iid = str(entry.id)
            self._history_entries[iid] = entry
            self.history_tree.insert(
                "",
                "end",
                iid=iid,
                values=(
                    entry.created_display,
                    entry.engine,
                    f"{entry.duration:.1f} с",
                    preview_text(entry.text, 110),
                ),
            )
        self._set_preview("")

    def _on_history_select(self, _event=None) -> None:
        selection = self.history_tree.selection()
        if not selection:
            self._set_preview("")
            return
        entry = self._history_entries.get(selection[0])
        self._set_preview(entry.text if entry else "")

    def _set_preview(self, text: str) -> None:
        self.history_preview.configure(state="normal")
        self.history_preview.delete("1.0", "end")
        self.history_preview.insert("1.0", text)
        self.history_preview.configure(state="disabled")

    def _selected_history_text(self) -> Optional[str]:
        selection = self.history_tree.selection()
        if not selection:
            return None
        entry = self._history_entries.get(selection[0])
        return entry.text if entry else None

    def _copy_history(self) -> None:
        text = self._selected_history_text()
        if not text:
            self.status_var.set("Выберите запись в истории")
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self.status_var.set("Текст скопирован в буфер обмена")

    def _delete_history(self) -> None:
        selection = self.history_tree.selection()
        if not selection:
            return
        if not messagebox.askyesno("Удалить", "Удалить выбранные записи?", parent=self.root):
            return
        for iid in selection:
            self._app.history.delete(int(iid))
        self._refresh_history()

    def _clear_history(self) -> None:
        if not messagebox.askyesno(
            "Очистить историю",
            "Удалить всю историю распознавания? Это действие необратимо.",
            parent=self.root,
        ):
            return
        self._app.history.clear()
        self._refresh_history()
        self.status_var.set("История очищена")

    def _export_history(self) -> None:
        if not self._app.history.count():
            self.status_var.set("История пуста")
            return
        destination = filedialog.asksaveasfilename(
            parent=self.root,
            title="Экспорт истории",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
            initialfile="voicetyper-history.csv",
        )
        if not destination:
            return
        try:
            self._app.history.export_csv(Path(destination))
            self.status_var.set(f"История экспортирована: {destination}")
        except Exception as exc:
            messagebox.showerror("Ошибка экспорта", str(exc), parent=self.root)

    # -- diagnostics -----------------------------------------------------
    def _refresh_diagnostics(self) -> None:
        try:
            content = self._app.collect_diagnostics_text()
        except Exception as exc:
            content = f"Не удалось собрать диагностику: {exc}"
        self.diagnostics_text.configure(state="normal")
        self.diagnostics_text.delete("1.0", "end")
        self.diagnostics_text.insert("1.0", content)
        self.diagnostics_text.configure(state="disabled")

    def _copy_diagnostics(self) -> None:
        content = self.diagnostics_text.get("1.0", "end").strip()
        self.root.clipboard_clear()
        self.root.clipboard_append(content)
        self.status_var.set("Диагностика скопирована")

    def _save_diagnostics(self) -> None:
        destination = filedialog.asksaveasfilename(
            parent=self.root,
            title="Сохранить диагностику",
            defaultextension=".txt",
            filetypes=[("Текст", "*.txt")],
            initialfile="voicetyper-diagnostics.txt",
        )
        if not destination:
            return
        try:
            Path(destination).write_text(
                self.diagnostics_text.get("1.0", "end").strip() + "\n", encoding="utf-8"
            )
            self.status_var.set(f"Сохранено: {destination}")
        except Exception as exc:
            messagebox.showerror("Ошибка сохранения", str(exc), parent=self.root)

    # -- tabs / commands -------------------------------------------------
    def show_tab(self, name: Optional[str]) -> None:
        if not name:
            return
        index = self._tabs.get(name)
        if index is not None:
            self.notebook.select(index)
            if name == "history":
                self._refresh_history()
            elif name == "diagnostics":
                self._refresh_diagnostics()

    def _on_tab_changed(self, _event=None) -> None:
        if self._capturing:
            self._finish_capture(None, "Запись комбинации отменена")
        current = self.notebook.index(self.notebook.select())
        if current == self._tabs.get("history"):
            self._refresh_history()
        elif current == self._tabs.get("diagnostics"):
            self._refresh_diagnostics()

    def _poll_commands(self) -> None:
        try:
            while True:
                command = self._commands.get_nowait()
                action = command[0]
                if action == "show":
                    self.root.deiconify()
                    self.root.lift()
                    try:
                        self.root.focus_force()
                    except Exception:
                        pass
                    if len(command) > 1:
                        self.show_tab(command[1])
                elif action == "reload":
                    self._load_from_config()
                    self._refresh_history()
                elif action == "close":
                    self._finish_capture(None, "")
                    self.root.destroy()
                    return
        except queue.Empty:
            pass
        self.root.after(200, self._poll_commands)

    def _on_close(self) -> None:
        if self._capturing:
            self._finish_capture(None, "")
        if self._dirty:
            if not messagebox.askyesno(
                "Не сохранено",
                "Есть несохранённые изменения. Закрыть окно без сохранения?",
                parent=self.root,
            ):
                return
        self.root.destroy()
