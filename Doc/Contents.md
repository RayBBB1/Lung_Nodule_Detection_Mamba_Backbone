# Window Env Build - Contents
Mamba只能在linux系統執行
將模型環境架設在Docker之中

## Contents
一、Docker 基礎知識與環境準備
* 1.1 Docker 簡介與優勢：了解 Docker 是什麼，以及為什麼使用 Docker 來進行 AI 模型訓練。
* 1.2 Windows 上 Docker 環境安裝：安裝 Docker Desktop for Windows 並配置 WSL2 後端。
* 1.3 Docker 常用命令：學習 Docker 基礎命令，例如 docker run, docker ps, docker images 等。

二、Linux Docker 容器操作
* 2.1 選擇 Linux 鏡像：選擇適合 AI 開發的 Linux 發行版鏡像，例如 Ubuntu。
* 2.2 創建與運行 Linux 容器：使用 docker run 命令創建並運行基於 Linux 鏡像的容器。
* 2.3 進入容器 Shell 環境：學習如何進入正在運行的 Docker 容器的 bash shell 環境。
* 2.4 容器檔案系統與主機檔案系統交互：了解如何使用 Volume 將主機目錄掛載到容器中，方便程式碼和數據的共享。

三、在 Docker 中配置 CUDA Toolkit
* 3.1 NVIDIA Container Toolkit 安裝與配置：安裝 NVIDIA Container Toolkit，以便在 Docker 容器中使用 NVIDIA GPU。
* 3.2 驗證 Docker 中 CUDA 是否可用：確認 Docker 容器可以訪問並使用主機的 NVIDIA GPU。
* 3.3 選擇或創建包含 CUDA 的 Docker 鏡像：選擇或創建預裝 CUDA Toolkit 的 Docker 鏡像，或學習如何在 Dockerfile 中安裝 CUDA。

四、在 Docker Linux 容器中安裝 PyTorch 與 CUDA
* 4.1 選擇 PyTorch Docker 鏡像：選擇預裝 PyTorch 和 CUDA 的 Docker 鏡像，簡化安裝過程。
* 4.2 手動安裝 PyTorch 與 CUDA (可選)：學習如何在 Docker 容器中手動安裝 PyTorch 和 CUDA Toolkit，更深入理解安裝過程。
* 4.3 驗證 PyTorch CUDA 是否可用：在 Docker 容器中運行 PyTorch 程式碼，驗證 CUDA 加速是否正常工作。

五、在 Docker Linux 容器中安裝 Mamba SSM
* 5.1 安裝 Mamba SSM 及其依賴：學習如何在 Docker 容器中使用 pip 安裝 Mamba SSM 以及其所需的依賴庫。
* 5.2 驗證 Mamba SSM 安裝：在 Docker 容器中導入 Mamba SSM 庫，驗證安裝是否成功。

六、在 Docker Linux 環境中訓練 AI 模型
* 6.1 準備模型訓練程式碼與數據集：將您的 PyTorch 模型訓練程式碼和數據集準備好，並放置在主機的目錄中，以便通過 Volume 掛載到容器。
* 6.2 在 Docker 容器中運行訓練腳本：在 Docker 容器中執行 PyTorch 訓練腳本，利用 CUDA 加速訓練模型。
* 6.3 管理與保存訓練結果：學習如何在 Docker 環境中管理訓練過程，並將訓練好的模型和結果保存到主機目錄中。

七、最佳實踐與常見問題
* 7.1 Dockerfile 優化：學習如何編寫高效的 Dockerfile，優化鏡像大小和構建速度。
* 7.2 容器資源管理：了解如何限制 Docker 容器的資源使用，例如 CPU 和記憶體。
* 7.3 常見問題排查：總結在 Windows Docker 環境下訓練 AI 模型時可能遇到的常見問題和解決方案。
