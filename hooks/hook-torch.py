from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs
import os
import torch

# Collect all Torch binaries and data files
datas = collect_data_files('torch')
binaries = collect_dynamic_libs('torch')

# Specifically ensure the lib directory is included
torch_path = os.path.dirname(torch.__file__)
torch_lib_path = os.path.join(torch_path, 'lib')

if os.path.exists(torch_lib_path):
    for file in os.listdir(torch_lib_path):
        file_path = os.path.join(torch_lib_path, file)
        if os.path.isfile(file_path):
            binaries.append((file_path, 'torch/lib'))