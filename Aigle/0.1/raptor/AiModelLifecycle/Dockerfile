# Dockerfile - AI Model Lifecycle Management Platform
# Multi-stage build for optimized production deployment

# --- Stage 1: Base Image ---
# 使用 NVIDIA PyTorch/CUDA 映像檔以支持 GPU 和 DeepSpeed
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

# Upgrade system packages to address vulnerabilities
RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libaio-dev \
    time \
    && rm -rf /var/lib/apt/lists/*

# --- 環境變數 ---
# 設定工作目錄
WORKDIR /app

# 防止 Python 寫入 .pyc 檔案
ENV PYTHONDONTWRITEBYTECODE=1
# 確保 Python 輸出是即時的，方便在 Docker logs 中查看
ENV PYTHONUNBUFFERED=1

# --- 安裝 Python 依賴 ---
# 使用 requirements.lock.txt 確保版本一致（從 uv.lock 導出）
# 這確保了與本地開發環境完全相同的依賴版本
COPY requirements.lock.txt ./

COPY pyproject.toml ./

# --- 複製應用程式程式碼 ---
# 將 src 目錄下的所有內容複製到工作目錄的 src 子目錄中
COPY ./src ./src

# 安裝 uv 並使用鎖定文件安裝依賴
# --system: 安裝到系統 Python（Docker 容器中不需要虛擬環境）
# --no-cache: 不使用快取，確保全新安裝
RUN pip install --no-cache-dir uv && \
    uv pip install --system --no-cache -r requirements.lock.txt

# --- DeepSpeed 專門編譯步驟 ---
# 在安裝完其他依賴後，編譯 DeepSpeed 的自定義 C++/CUDA 擴展
# 這樣可以避免運行時的 JIT 編譯，提高穩定性。
RUN DS_BUILD_CPU_ADAM=1 DS_BUILD_FUSED_ADAM=1 pip install deepspeed==0.18.2 --no-cache-dir \
    --global-option="build_ext" --global-option="-j$(nproc)" \
    && rm -rf /root/.cache/pip

# --- 端口暴露 ---
# 我們的 FastAPI 應用預計在 8000 端口運行
EXPOSE ${FASTAPI_PORT}

# --- 啟動命令 ---
# 當容器啟動時，執行此命令來啟動 Uvicorn 伺服器
# 綁定到 0.0.0.0 以便從容器外部訪問
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "${FASTAPI_PORT}", "--reload"]