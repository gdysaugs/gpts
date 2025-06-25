# Multi-stage build for CUDA-enabled llama-cpp-python
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04 AS builder

# 環境変数設定
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_DOCKER_ARCH=all
ENV LLAMA_CUBLAS=on
ENV CMAKE_CUDA_ARCHITECTURES=all

# 基本パッケージのインストール
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python環境設定
RUN ln -s /usr/bin/python3 /usr/bin/python
RUN pip3 install --upgrade pip setuptools wheel

# CUDAライブラリのリンクを修正
RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1

# CUDA対応llama-cpp-pythonをソースからビルド
# RTX 3050は compute capability 8.6
RUN CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=86" \
    FORCE_CMAKE=1 \
    CUDA_DOCKER_ARCH=sm_86 \
    LD_LIBRARY_PATH="/usr/local/cuda/lib64/stubs:${LD_LIBRARY_PATH}" \
    pip install 'llama-cpp-python[server]' --no-cache-dir --verbose

# Runtime stage
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# 環境変数設定
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# ランタイム依存関係
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Python環境設定
RUN ln -s /usr/bin/python3 /usr/bin/python

# ビルド済みパッケージをコピー
COPY --from=builder /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# 追加のPython依存関係
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# 作業ディレクトリ設定
WORKDIR /app

# アプリケーションファイルをコピー
COPY scripts/ /app/scripts/
COPY src/ /app/src/
COPY config/ /app/config/

# Pythonパス設定
ENV PYTHONPATH="/app/src:${PYTHONPATH}"

# モデルディレクトリ作成
RUN mkdir -p /app/models /app/logs

# 実行権限設定
RUN chmod +x /app/scripts/*.py

# ポート公開
EXPOSE 8000

# デフォルトコマンド（CLIモード）
CMD ["python", "/app/scripts/chat_cli.py"]