# Easy-Wav2Lip + YOLO11 統合 Dockerfile
# WSL2 + RTX 3050 最適化版

# Stage 1: NVIDIA NGC PyTorch Base (CUDA 12.1 compatible)
FROM nvcr.io/nvidia/pytorch:24.01-py3 AS base

# 環境変数設定
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=0

# システム依存関係インストール
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    libgtk-3-0 \
    cmake \
    git \
    wget \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Stage 2: Python Dependencies
FROM base AS dependencies

# 作業ディレクトリ設定
WORKDIR /app

# requirements.txt コピー
COPY requirements-minimal.txt .

# Python依存関係インストール（最小構成）
RUN pip install --no-cache-dir -r requirements-minimal.txt

# YOLO11インストール
RUN pip install --no-cache-dir ultralytics

# TensorRT最適化用
RUN pip install --no-cache-dir torch-tensorrt nvidia-tensorrt

# 追加の依存関係
RUN pip install --no-cache-dir \
    onnx \
    onnxruntime-gpu \
    opencv-python-headless \
    pillow \
    tqdm \
    pyyaml \
    easydict

# Stage 3: Model Download
FROM dependencies AS models

# モデルディレクトリ作成
RUN mkdir -p /app/models/{wav2lip,yolo,face_detection,gfpgan}

# Wav2Lipモデルダウンロード
RUN gdown --id 1KQPKpuXHGdpGPiYcJwSZZIJLcMVFEUuC -O /app/models/wav2lip/wav2lip.pth
RUN gdown --id 1fQtBSYEyuai_jHRhfKJYkpBhZJmryHDw -O /app/models/wav2lip/wav2lip_gan.pth

# YOLO11モデルダウンロード
RUN wget -O /app/models/yolo/yolo11n.pt https://github.com/ultralytics/assets/releases/download/v8.2.0/yolo11n.pt
RUN wget -O /app/models/yolo/yolo11s.pt https://github.com/ultralytics/assets/releases/download/v8.2.0/yolo11s.pt

# Face detectionモデル
RUN gdown --id 1BPqVN3ql7ybvGPRGEcGJtfYfVyBvQkBr -O /app/models/face_detection/retinaface_resnet50.pth

# GFPGANモデル
RUN gdown --id 1GnpFZMojYlAGELiXBnqBGQSfDFGGb8e2 -O /app/models/gfpgan/GFPGANv1.4.pth

# Stage 4: Application Runtime
FROM models AS runtime

# アプリケーションコードコピー
COPY . /app/

# ディレクトリ作成
RUN mkdir -p /app/{input,output,logs,config,scripts}

# 実行権限設定
RUN chmod +x /app/scripts/*.py

# ポート公開（API用）
EXPOSE 8000

# 非rootユーザー作成（セキュリティ向上）
RUN useradd -m -u 1000 wav2lip && \
    chown -R wav2lip:wav2lip /app
USER wav2lip

# ヘルスチェック
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import torch; print('CUDA Available:', torch.cuda.is_available())"

# デフォルトコマンド
CMD ["python", "/app/scripts/wav2lip_cli.py", "--help"]