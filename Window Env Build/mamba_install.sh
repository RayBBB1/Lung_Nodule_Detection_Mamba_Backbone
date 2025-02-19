#!/bin/bash
# -*- coding: utf-8 -*-

# 切换到 mamba 目录
cd mamba || { echo "Failed to change directory to 'mamba'"; exit 1; }

# 使用 pip 安装当前目录中的包
pip install . || { echo "Failed to install the package"; exit 1; }

# 复制 mamba_ssm 目录到目标路径
cp -r mamba_ssm /usr/local/lib/python3.8/dist-packages/ || { echo "Failed to copy 'mamba_ssm' to target directory"; exit 1; }

pip install causal_conv1d==1.0.2
echo "Script executed successfully!"
