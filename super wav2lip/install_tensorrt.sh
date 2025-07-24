#!/bin/bash
# TensorRTライブラリのみインストール（競合回避版）

echo "🚀 TensorRTライブラリインストール開始..."

# NVIDIA公式リポジトリ追加
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb

# TensorRTライブラリのみインストール（CUDAツールキットなし）
apt-get update
apt-get install -y --no-install-recommends \
    libnvinfer10 \
    libnvinfer-plugin10 \
    libnvparsers10 \
    libnvonnxparsers10 \
    python3-libnvinfer

echo "✅ TensorRTライブラリインストール完了"

# 確認
echo "🔍 インストール確認:"
find /usr -name "libnvinfer.so*" | head -5