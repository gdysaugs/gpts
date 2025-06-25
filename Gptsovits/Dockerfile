# Stage 1: モデルダウンロード用ステージ
FROM ubuntu:22.04 AS model-downloader

# 環境変数設定
ENV DEBIAN_FRONTEND=noninteractive
ENV GIT_LFS_SKIP_SMUDGE=0

# Git LFSと必要なツールのインストール
RUN apt-get update && apt-get install -y \
    git \
    git-lfs \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Git LFSの初期化
RUN git lfs install

# 作業ディレクトリ設定
WORKDIR /models

# GPT-SoVITSリポジトリのクローン（基本構造）
RUN git clone https://github.com/RVC-Boss/GPT-SoVITS.git /app/GPT-SoVITS

# HuggingFaceからv4モデルをダウンロード
WORKDIR /models/pretrained
RUN git clone https://huggingface.co/lj1995/GPT-SoVITS.git . && \
    git lfs fetch --all && \
    git lfs checkout

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
    python3-pip \
    ffmpeg \
    libsndfile1 \
    libportaudio2 \
    && rm -rf /var/lib/apt/lists/*

# Pythonのデフォルト設定
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# pipのアップグレード
RUN pip install --upgrade pip setuptools wheel

# 作業ディレクトリ設定
WORKDIR /app

# GPT-SoVITSのコピー（Stage 1から）
COPY --from=model-downloader /app/GPT-SoVITS /app

# PyTorchのインストール（安全策: 2.1.2 + CUDA 12.1、既存依存関係を維持）
RUN pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121

# ONNX Runtime + TensorRT最適化パッケージ
RUN pip install onnx==1.15.0 \
    onnxruntime-gpu==1.16.3 \
    onnxoptimizer==0.3.13 \
    tensorrt==8.6.1.post1

# FastAPI + 音声処理パッケージ
RUN pip install fastapi==0.104.1 \
    uvicorn==0.24.0 \
    python-multipart==0.0.6

# Transformersライブラリの互換性修正
RUN pip install transformers==4.35.2

# 依存関係のインストール
RUN pip install -r requirements.txt

# Stage 3: 実行環境
FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

# 環境変数設定
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV CUDA_HOME=/usr/local/cuda-12.2
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/torch/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV IS_HALF=True

# 最小限のシステムパッケージ
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    ffmpeg \
    libsndfile1 \
    libportaudio2 \
    && rm -rf /var/lib/apt/lists/*

# Pythonのデフォルト設定
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# 作業ディレクトリ設定
WORKDIR /app

# アプリケーションとPython環境のコピー
COPY --from=python-builder /app /app
COPY --from=python-builder /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages

# モデルファイルのコピー（全てのモデルをコピー）
COPY --from=model-downloader /models/pretrained/ /app/GPT_SoVITS/pretrained_models/

# ディレクトリ作成
RUN mkdir -p /app/input /app/output /app/logs /app/scripts /app/models/onnx

# ボリューム設定
VOLUME ["/app/input", "/app/output", "/app/logs"]

# pyopenjtalkの権限問題を解決
RUN chmod -R 777 /usr/local/lib/python3.10/dist-packages/pyopenjtalk/ || true

# 実行ユーザー設定（セキュリティ向上）
RUN useradd -m -u 1000 gpt-sovits && \
    chown -R gpt-sovits:gpt-sovits /app

# pyopenjtalkディレクトリの権限を再設定
RUN chown -R gpt-sovits:gpt-sovits /usr/local/lib/python3.10/dist-packages/pyopenjtalk/ || true

USER gpt-sovits

# デフォルトコマンド
CMD ["python", "-u", "/app/scripts/test_voice_clone.py"]