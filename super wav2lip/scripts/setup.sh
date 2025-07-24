#!/bin/bash
# Super Wav2Lip セットアップスクリプト
# Windows ファイルの配置と環境準備

set -e

echo "🎭 Super Wav2Lip セットアップ開始"
echo "=================================="

# カレントディレクトリを確認
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "プロジェクトディレクトリ: $PROJECT_DIR"
cd "$PROJECT_DIR"

# 必要なディレクトリ作成
echo "📁 ディレクトリ構造を作成中..."
mkdir -p input/videos input/audio output models/onnx models/enhancers logs temp

# Windowsファイルパスの定義
WIN_VIDEO_PATH="/mnt/c/Users/adama/Videos/画面録画/画面録画 2025-05-16 222902.mp4"
WIN_AUDIO_PATH="/mnt/c/Users/adama/Downloads/ohayougozaimasu_10.wav" 
# ONNX モデルファイル (複数確認)
WIN_ONNX_MODEL1="/mnt/c/Users/adama/Downloads/wav2lip_384.onnx"
WIN_ONNX_MODEL2="/mnt/c/Users/adama/Downloads/model-512.onnx"

echo "📥 Windowsファイルをコピー中..."

# 動画ファイルのコピー
if [ -f "$WIN_VIDEO_PATH" ]; then
    cp "$WIN_VIDEO_PATH" "input/videos/source_video.mp4"
    echo "✅ 動画ファイルをコピーしました: source_video.mp4"
    
    # ファイル情報表示
    VIDEO_SIZE=$(du -h "input/videos/source_video.mp4" | cut -f1)
    echo "   ファイルサイズ: $VIDEO_SIZE"
else
    echo "❌ 動画ファイルが見つかりません: $WIN_VIDEO_PATH"
    echo "   手動でファイルを input/videos/ に配置してください"
fi

# 音声ファイルのコピー
if [ -f "$WIN_AUDIO_PATH" ]; then
    cp "$WIN_AUDIO_PATH" "input/audio/target_audio.wav"
    echo "✅ 音声ファイルをコピーしました: target_audio.wav"
    
    # ファイル情報表示
    AUDIO_SIZE=$(du -h "input/audio/target_audio.wav" | cut -f1)
    echo "   ファイルサイズ: $AUDIO_SIZE"
else
    echo "❌ 音声ファイルが見つかりません: $WIN_AUDIO_PATH"
    echo "   手動でファイルを input/audio/ に配置してください"
fi

# ONNXモデルファイルのコピー  
echo ""
echo "🤖 ONNX モデルファイルを確認中..."

# wav2lip_384.onnx の確認
if [ -f "$WIN_ONNX_MODEL1" ]; then
    cp "$WIN_ONNX_MODEL1" "models/onnx/wav2lip_384.onnx"
    echo "✅ Wav2Lip ONNXモデルをコピーしました: wav2lip_384.onnx"
    
    # ファイル情報表示
    MODEL_SIZE=$(du -h "models/onnx/wav2lip_384.onnx" | cut -f1)
    echo "   ファイルサイズ: $MODEL_SIZE"
fi

# その他のONNXモデルの確認
if [ -f "$WIN_ONNX_MODEL2" ]; then
    cp "$WIN_ONNX_MODEL2" "models/onnx/model-512.onnx"
    echo "✅ 追加ONNXモデルをコピーしました: model-512.onnx"
fi

# 全てのONNXファイルを確認してコピー
echo ""
echo "🔍 すべてのONNXモデルファイルを確認中..."
ONNX_COUNT=0
for onnx_file in "/mnt/c/Users/adama/Downloads/"*.onnx; do
    if [ -f "$onnx_file" ]; then
        filename=$(basename "$onnx_file")
        cp "$onnx_file" "models/onnx/$filename"
        echo "✅ ONNXモデルをコピーしました: $filename"
        ONNX_COUNT=$((ONNX_COUNT + 1))
    fi
done

if [ $ONNX_COUNT -eq 0 ]; then
    echo "❌ ONNXモデルファイルが見つかりません"
    echo "   📋 必要なファイル:"
    echo "   - wav2lip_384.onnx (メインモデル)"
    echo "   - recognition.onnx (顔認識)"
    echo "   - scrfd_2.5g_bnkps.onnx (顔検出)"
    echo ""
    echo "   📥 Google Drive からダウンロード:"
    echo "   https://drive.google.com/drive/folders/1BGl9bmMtlGEMx_wwKufJrZChFyqjnlsQ"
fi

echo ""
echo "🔧 環境確認中..."

# GPU確認
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU が検出されました"
    nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits
else
    echo "⚠️  NVIDIA GPU が検出されませんでした"
fi

# Docker確認
if command -v docker &> /dev/null; then
    echo "✅ Docker が利用可能です"
    docker --version
    
    # NVIDIA Container Toolkit確認
    if docker run --gpus all --rm nvidia/cuda:12.1-runtime-ubuntu22.04 nvidia-smi &> /dev/null; then
        echo "✅ Docker GPU support が動作しています"
    else
        echo "⚠️  Docker GPU support に問題があります"
        echo "   NVIDIA Container Toolkit の確認が必要です"
    fi
else
    echo "❌ Docker が見つかりません"
    echo "   Docker をインストールしてください"
fi

echo ""
echo "📊 ファイル一覧:"
echo "入力動画:"
ls -lh input/videos/ 2>/dev/null || echo "  (なし)"

echo "入力音声:"
ls -lh input/audio/ 2>/dev/null || echo "  (なし)"

echo "モデルファイル:"
ls -lh models/*.pth 2>/dev/null || echo "  (なし)"
ls -lh models/onnx/ 2>/dev/null || echo "  (なし)"

echo ""
echo "✅ セットアップ完了！"
echo ""
echo "次のステップ:"
echo "1. Dockerイメージをビルド:"
echo "   DOCKER_BUILDKIT=1 docker build -t super-wav2lip:v1 ."
echo ""
echo "2. テスト実行:"
echo "   ./scripts/test_lipsync.sh"
echo ""
echo "3. 手動実行:"
echo "   docker run --gpus all --rm --privileged \\"
echo "     -v /usr/lib/wsl:/usr/lib/wsl \\"
echo "     -e LD_LIBRARY_PATH=/usr/lib/wsl/lib \\"
echo "     -v \$(pwd)/input:/app/input \\"
echo "     -v \$(pwd)/output:/app/output \\"
echo "     -v \$(pwd)/models:/app/models \\"
echo "     super-wav2lip:v1 python src/lipsync_cli.py \\"
echo "     --checkpoint_path /app/models/wav2lip.pth \\"
echo "     --face /app/input/videos/source_video.mp4 \\"
echo "     --audio /app/input/audio/target_audio.wav \\"
echo "     --outfile /app/output/result.mp4"