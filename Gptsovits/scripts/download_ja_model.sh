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

# 日本語特化感情モデル (hscene-e17.ckpt) - 唯一利用可能なファイル
echo "日本語特化感情モデル(hscene-e17.ckpt)をダウンロード中..."
echo "これは650時間の日本語音声データで訓練された特化モデルです..."

wget -O hscene-e17.ckpt "https://huggingface.co/AkitoP/GPT-SoVITS-JA-H/resolve/main/hscene-e17.ckpt" || {
    echo "❌ エラー: hscene-e17.ckptのダウンロードに失敗しました"
    echo "手動で以下からダウンロードしてください："
    echo "https://huggingface.co/AkitoP/GPT-SoVITS-JA-H"
    exit 1
}

echo "✅ 日本語特化モデルのダウンロードが完了しました"
echo ""
echo "注意: このモデルは標準のGPT-SoVITSモデルと組み合わせて使用します"
echo "標準のv2モデルが必要な場合は、先にdownload_models.shを実行してください"

echo "ダウンロード完了!"
echo "モデルファイル一覧:"
ls -la "$JA_MODEL_DIR"

# ファイルサイズを確認
echo ""
echo "=== ダウンロード済みファイル ==="
find "$JA_MODEL_DIR" -type f -name "*.pth" -o -name "*.ckpt" -o -name "*.safetensors" | xargs ls -lh