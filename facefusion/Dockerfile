# Stage 1: リポジトリクローン用ステージ
FROM ubuntu:22.04 AS repo-downloader

# 環境変数設定
ENV DEBIAN_FRONTEND=noninteractive

# Git のインストール
RUN apt-get update && apt-get install -y \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# 作業ディレクトリ設定
WORKDIR /repos

# FaceFusionリポジトリのクローン
RUN git clone https://github.com/facefusion/facefusion.git /app/facefusion

# Stage 2: Python環境構築
FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04 AS python-builder

# 環境変数設定
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda-12.2
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# システムパッケージのインストール
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    python3-pip \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Pythonのデフォルト設定
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# pipのアップグレード
RUN pip install --upgrade pip setuptools wheel

# 作業ディレクトリ設定
WORKDIR /app

# FaceFusionのコピー（Stage 1から）
COPY --from=repo-downloader /app/facefusion /app

# PyTorchのインストール（CUDA 12.1対応）
RUN pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121

# FaceFusion依存関係のインストール
RUN pip install \
    gradio-rangeslider==0.0.8 \
    gradio==5.25.2 \
    numpy==2.2.4 \
    onnx==1.17.0 \
    onnxruntime-gpu==1.22.0 \
    opencv-python==4.11.0.86 \
    psutil==7.0.0 \
    tqdm==4.67.1 \
    scipy==1.15.2 \
    requests \
    pillow \
    imageio \
    face-alignment \
    insightface \
    mediapipe

# Stage 3: 実行環境
FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

# 環境変数設定
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV CUDA_HOME=/usr/local/cuda-12.2
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/torch/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# 最小限のシステムパッケージ
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Pythonのデフォルト設定
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# 作業ディレクトリ設定
WORKDIR /app

# アプリケーションとPython環境のコピー
COPY --from=python-builder /app /app
COPY --from=python-builder /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages

# ディレクトリ作成
RUN mkdir -p /app/input /app/output /app/logs /app/scripts

# ボリューム設定
VOLUME ["/app/input", "/app/output", "/app/logs"]

# 実行ユーザー設定（セキュリティ向上）
RUN useradd -m -u 1000 facefusion && \
    chown -R facefusion:facefusion /app

USER facefusion

# デフォルトコマンド
CMD ["python", "facefusion.py", "--help"]