#!/bin/bash
# -*- coding: utf-8 -*-
pip install -r"requirements.txt"
# 切换到 mamba 目录
cd causal-conv1d || { echo "Failed to change directory to 'causal-conv1d'"; exit 1; }

# 使用 pip 安装当前目录中的包
pip install . || { echo "Failed to install the package"; exit 1; }

cd ..
cd mamba || { echo "Failed to change directory to 'mamba'"; exit 1; }

# 使用 pip 安装当前目录中的包
pip install . || { echo "Failed to install the package"; exit 1; }

# 复制 mamba_ssm 目录到目标路径
cp -r mamba_ssm /usr/local/lib/python3.10/dist-packages/ || { echo "Failed to copy 'mamba_ssm' to target directory"; exit 1; }


echo "Script executed successfully!"