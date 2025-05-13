FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# 設定工作目錄
WORKDIR /app

# 更新套件並安裝 Python 3.10 及 pip
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-dev \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 確保 Python 3.10 為預設版本 
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# 確認 Python 版本 
RUN python --version

# 安裝 PyTorch (與 CUDA 11.8 相容的版本)
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# (可選) 安裝其他 Python 套件
# RUN pip install --no-cache-dir numpy pandas ...

# 複製程式碼到容器內部
# COPY . .

# 設定容器啟動時執行的命令 (可選)
# CMD ["python", "your_script.py"]
