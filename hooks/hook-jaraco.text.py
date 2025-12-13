# Hook for jaraco.text to handle missing Lorem ipsum.txt file
from PyInstaller.utils.hooks import collect_data_files
import os

# Collect data files for jaraco.text
datas = collect_data_files('jaraco.text')

# Try to locate and include the missing Lorem ipsum.txt file
try:
    import jaraco.text
    jaraco_path = os.path.dirname(jaraco.text.__file__)
    lorem_file = os.path.join(jaraco_path, 'Lorem ipsum.txt')
    if os.path.exists(lorem_file):
        datas.append((lorem_file, 'jaraco/text'))
except ImportError:
    pass