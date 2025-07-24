#!/bin/bash
# Super Wav2Lip テスト実行スクリプト
# 口パク動画生成のテスト

set -e

echo "🎭 Super Wav2Lip テスト実行"
echo "=========================="

# カレントディレクトリを確認
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "プロジェクトディレクトリ: $PROJECT_DIR"
cd "$PROJECT_DIR"

# 必要ファイルの確認
echo "📋 必要ファイルを確認中..."

# 入力動画確認
if [ ! -f "input/videos/source_video.mp4" ]; then
    echo "❌ 入力動画が見つかりません: input/videos/source_video.mp4"
    echo "   setup.sh を実行してファイルを配置してください"
    exit 1
fi

# 入力音声確認
if [ ! -f "input/audio/target_audio.wav" ]; then
    echo "❌ 入力音声が見つかりません: input/audio/target_audio.wav"
    echo "   setup.sh を実行してファイルを配置してください"
    exit 1
fi

# ONNXモデルファイル確認
MODEL_FILE=""
if [ -f "models/onnx/model-512.onnx" ]; then
    MODEL_FILE="models/onnx/model-512.onnx" 
    echo "✅ ONNX (512)モデルを使用: $MODEL_FILE"
elif [ -f "models/onnx/wav2lip_384.onnx" ]; then
    MODEL_FILE="models/onnx/wav2lip_384.onnx"
    echo "✅ Wav2Lip ONNX (384)モデルを使用: $MODEL_FILE"
else
    echo "❌ ONNXモデルファイルが見つかりません"
    echo "   必要なファイル: wav2lip_384.onnx"
    echo "   Google Drive からダウンロード:"
    echo "   https://drive.google.com/drive/folders/1BGl9bmMtlGEMx_wwKufJrZChFyqjnlsQ"
    echo ""
    echo "   setup.sh を実行してファイルを配置してください"
    exit 1
fi

# Dockerイメージ確認
echo "🐳 Dockerイメージを確認中..."
if ! docker images | grep -q "super-wav2lip"; then
    echo "📦 Dockerイメージをビルド中..."
    DOCKER_BUILDKIT=1 docker build -t super-wav2lip:v1 .
else
    echo "✅ Dockerイメージが見つかりました"
fi

# GPU確認
echo "🔧 GPU環境を確認中..."
if ! docker run --gpus all --rm --privileged \
    -v /usr/lib/wsl:/usr/lib/wsl \
    -e LD_LIBRARY_PATH=/usr/lib/wsl/lib \
    super-wav2lip:v1 nvidia-smi > /dev/null 2>&1; then
    echo "⚠️  GPU環境に問題があります。CPUで実行します。"
    GPU_FLAGS=""
else
    echo "✅ GPU環境が正常です"
    GPU_FLAGS="--gpus all --privileged -v /usr/lib/wsl:/usr/lib/wsl -e LD_LIBRARY_PATH=/usr/lib/wsl/lib"
fi

# 出力ディレクトリクリア
echo "🧹 出力ディレクトリをクリア中..."
rm -rf output/*
mkdir -p output

# ファイル情報表示
echo ""
echo "📊 入力ファイル情報:"
echo "動画ファイル:"
ls -lh input/videos/source_video.mp4

echo "音声ファイル:"
ls -lh input/audio/target_audio.wav

echo "モデルファイル:"
ls -lh "$MODEL_FILE"

# テスト実行開始
echo ""
echo "🚀 口パク動画生成テストを開始..."
echo "処理時間を計測します..."

START_TIME=$(date +%s)

# Docker実行
echo "実行コマンド:"
echo "docker run $GPU_FLAGS --rm \\"
echo "  -v \$(pwd)/input:/app/input \\"
echo "  -v \$(pwd)/output:/app/output \\"
echo "  -v \$(pwd)/models:/app/models \\"
echo "  super-wav2lip:v1 python3 src/lipsync_cli.py \\"
echo "  --checkpoint_path /app/$MODEL_FILE \\"
echo "  --face /app/input/videos/source_video.mp4 \\"
echo "  --audio /app/input/audio/target_audio.wav \\"
echo "  --outfile /app/output/result.mp4 \\"
echo "  --verbose"

echo ""
echo "実行中..."

# 実際の実行
if docker run --gpus all --privileged \
    -v /usr/lib/wsl:/usr/lib/wsl \
    -e LD_LIBRARY_PATH=/usr/lib/wsl/lib \
    --rm \
    -v "$(pwd)/input":/app/input \
    -v "$(pwd)/output":/app/output \
    -v "$(pwd)/models":/app/models \
    super-wav2lip:v1 python3 src/lipsync_cli.py \
    --checkpoint_path "/app/$MODEL_FILE" \
    --face /app/input/videos/source_video.mp4 \
    --audio /app/input/audio/target_audio.wav \
    --outfile /app/output/result.mp4 \
    --verbose; then
    
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    
    echo ""
    echo "✅ 口パク動画生成が完了しました！"
    echo "処理時間: ${DURATION}秒"
    
    # 結果確認
    if [ -f "output/result.mp4" ]; then
        echo ""
        echo "📊 出力結果:"
        ls -lh output/result.mp4
        
        OUTPUT_SIZE=$(du -h output/result.mp4 | cut -f1)
        echo "出力ファイルサイズ: $OUTPUT_SIZE"
        
        # 動画情報取得 (ffprobeがある場合)
        if command -v ffprobe &> /dev/null; then
            echo ""
            echo "📹 動画情報:"
            ffprobe -v quiet -show_format -show_streams output/result.mp4 | grep -E "(duration|width|height|codec_name)" | head -10
        fi
        
        echo ""
        echo "🎉 テスト完了！"
        echo "出力ファイル: output/result.mp4"
        echo ""
        echo "次のステップ:"
        echo "1. ファイルをWindowsに転送:"
        echo "   cp output/result.mp4 '/mnt/c/Users/adama/Downloads/super_wav2lip_result.mp4'"
        echo ""
        echo "2. 高品質版を試す場合:"
        echo "   ./scripts/test_hq_lipsync.sh"
        
    else
        echo "❌ 出力ファイルが生成されませんでした"
        echo "ログを確認してください"
        exit 1
    fi
    
else
    echo "❌ 口パク動画生成に失敗しました"
    exit 1
fi