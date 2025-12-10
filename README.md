# Voice Typer

A speech-to-text application that converts spoken words into written text automatically. The application runs in the system tray and uses configurable hotkeys to activate voice recording.

## Features

- **Multiple Speech Recognition Engines**:
  - OpenAI Whisper
  - Faster Whisper (optimized version)
  - Vosk (offline speech recognition)

- **Configurable Settings**:
  - Customizable hotkey activation
  - Language selection (auto-detection or specific language)
  - Model size options (tiny, base, small, medium, large)
  - Device selection (CPU or GPU with CUDA support)
  - Excludable phrases that won't be typed

- **System Tray Interface**:
  - Visual indicator when recording (RED "REC" icon)
  - Easy access to settings through the system tray menu
  - Language indicator in the icon

- **Advanced Features**:
  - CUDA support for GPU acceleration
  - Comprehensive logging system
  - Automatic model downloading
  - Phrase exclusion filtering
  - Cross-platform path handling (works as .py or .exe)

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd voice-typer
   ```

2. Install required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Download speech recognition models (automatically done on first run)

## Usage

1. Run the application:
   ```
   python voice_typer.py
   ```
   
   Or run the compiled executable:
   ```
   ./dist/voice_typer.exe
   ```

2. The application will appear in your system tray (look for the language indicator)

3. Press the configured hotkey (default: `Win+Ctrl`) to start recording

4. Speak into your microphone

5. Release the hotkey to stop recording and have the text automatically typed at the cursor position

## Configuration

The application creates a `config.json` file on first run with the following default settings:

```json
{
    "hotkey": "win+ctrl",
    "language": "auto",
    "model": "large",
    "model_type": "faster_whisper",
    "device": "cuda",
    "force_cpu": false,
    "vosk_model": "vosk-model-ru-0.42",
    "exclude_phrases": [
        "Thank you.",
        "We'll be right back.",
        "Thanks for watching!"
    ]
}
```

You can modify these settings directly in the `config.json` file or through the system tray menu.

## Building Executable

To build a standalone executable:

1. Install PyInstaller:
   ```
   pip install pyinstaller
   ```

2. Build the executable:
   ```
   pyinstaller voice_typer.spec
   ```

The executable will be created in the `dist` folder.

## Troubleshooting

### DLL Loading Issues

If you encounter DLL loading errors (like "OSError: [WinError 1114] Произошел сбой в программе инициализации библиотеки динамической компоновки (DLL)"), this is typically caused by missing or incompatible Torch library dependencies.

**Solution**: The updated `voice_typer.spec` file and custom Torch hook in this repository properly handle Torch dependencies. The build process now includes:

1. Custom Torch hook that ensures all necessary DLLs are included
2. Proper collection of Torch binaries and data files
3. Explicit inclusion of the problematic `c10.dll` and other critical libraries

### CUDA Setup

For GPU acceleration with CUDA:
1. Ensure you have compatible NVIDIA drivers installed
2. Install CUDA toolkit compatible with your PyTorch version
3. Make sure your GPU is supported by the Torch CUDA version

## Requirements

- Python 3.7+
- CUDA-compatible GPU (optional, for GPU acceleration)
- Microphone for voice input

## Dependencies

- openai-whisper or faster-whisper
- vosk (for offline speech recognition)
- sounddevice
- soundfile
- pyautogui
- keyboard
- pystray
- pillow
- torch
- pyperclip

See `requirements.txt` for complete list of dependencies.

## License

This project is licensed under the MIT License - see the LICENSE file for details.