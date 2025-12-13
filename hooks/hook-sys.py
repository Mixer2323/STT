# Hook for sys module to ensure Python DLLs are included
import os
import sys
from PyInstaller.utils.hooks import collect_dynamic_libs

# Collect Python DLLs
binaries = []

# Get the Python installation directory
python_dir = os.path.dirname(sys.executable)

# Common Python DLLs that might be needed
python_dlls = [
    'python{}.dll'.format(sys.version_info.major),
    'python{}{}.dll'.format(sys.version_info.major, sys.version_info.minor),
]

for dll in python_dlls:
    dll_path = os.path.join(python_dir, dll)
    if os.path.exists(dll_path):
        binaries.append((dll_path, '.'))