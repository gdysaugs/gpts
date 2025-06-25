#!/bin/bash

set -e

# 作業ディレクトリ
WORK_DIR="/app/GPT_SoVITS/pretrained_models"
JA_MODEL_DIR="$WORK_DIR/gpt-sovits-ja-h"

echo "日本語特化モデルをダウンロード中..."

# ディレクトリ作成
mkdir -p "$JA_MODEL_DIR"
cd "$JA_MODEL_DIR"

# 主要なモデルファイルを直接ダウンロード
echo "モデルファイルを直接ダウンロード..."

# GPTモデル
echo "GPTモデルをダウンロード中..."
wget -O s2G2333k.pth "https://huggingface.co/AkitoP/GPT-SoVITS-JA-H/resolve/main/s2G2333k.pth"

# SoVITSモデル
echo "SoVITSモデルをダウンロード中..."
wget -O s2D2333k.pth "https://huggingface.co/AkitoP/GPT-SoVITS-JA-H/resolve/main/s2D2333k.pth"

# 設定ファイルなどもダウンロード
echo "設定ファイルをダウンロード中..."
wget -O config.json "https://huggingface.co/AkitoP/GPT-SoVITS-JA-H/resolve/main/config.json" || echo "config.jsonが見つかりません"
wget -O README.md "https://huggingface.co/AkitoP/GPT-SoVITS-JA-H/resolve/main/README.md" || echo "README.mdが見つかりません"

echo "ダウンロード完了!"
echo "モデルファイル一覧:"
ls -la "$JA_MODEL_DIR"

# ファイルサイズを確認
echo ""
echo "=== ダウンロード済みファイル ==="
find "$JA_MODEL_DIR" -type f -name "*.pth" -o -name "*.ckpt" -o -name "*.safetensors" | xargs ls -lh