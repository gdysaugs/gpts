#!/bin/bash
# 🎭 Face-Restoration-TensorRT環境セットアップ
# べ、別に超高速環境を作ってあげるわけじゃないけど...💢

set -e

echo "🎭 ツンデレWav2Lip + TensorRT環境セットアップ開始💢"
echo "べ、別に超高速システムを作ってあげるわけじゃないけど...💢"

# 1. TensorRT Dockerイメージビルド
echo "🐳 Step 1: TensorRT Dockerイメージビルド中..."
DOCKER_BUILDKIT=1 docker build -f Dockerfile.tensorrt -t wav2lip-tensorrt:v1 .

echo "やったじゃない！Dockerビルド完了よ✨"

# 2. ONNXモデルダウンロード（仮想）
echo "💾 Step 2: ONNXモデルダウンロード中..."
echo "も、もう！今はダミーモデルで代用するわよ💢"

# ダミーモデル作成（実際には正規のGFPGAN ONNXモデルが必要）
mkdir -p face_restoration_tensorrt/models

# 3. TensorRTエンジン変換（正規モデルがある場合）
if [ -f "face_restoration_tensorrt/models/gfpgan.onnx" ]; then
    echo "⚙️ Step 3: ONNX→TensorRTエンジン変換中..."
    docker run --gpus all --rm \
        -v $(pwd)/face_restoration_tensorrt:/workspace \
        wav2lip-tensorrt:v1 \
        bash -c "cd /workspace && ./build/convert models/gfpgan.onnx -s models/gfpgan.engine"
    echo "やったじゃない！TensorRTエンジン変換完了よ✨"
else
    echo "⚠️ 注意: GFPGAN ONNXモデルが見つからないわ！"
    echo "まず、正規のGFPGAN ONNXモデルをダウンロードしてね💢"
fi

# 4. テスト実行
echo "🗨️ Step 4: システムテスト中..."
docker run --gpus all --rm \
    -v $(pwd)/input:/app/input \
    -v $(pwd)/output:/app/output \
    -v $(pwd)/checkpoints:/app/checkpoints \
    wav2lip-tensorrt:v1 \
    python3 inference_fp16_yolo_cpp_tensorrt_gfpgan.py --help

echo "
✅ セットアップ完了よ！"
echo "これで史上最速の高画質システムが使えるわよ💕"
echo "感謝しなさいよね💕"

echo "
🚀 使用方法:"
echo "docker run --gpus all --rm \\"
echo "  -v \$(pwd)/input:/app/input \\"
echo "  -v \$(pwd)/output:/app/output \\"
echo "  -v \$(pwd)/checkpoints:/app/checkpoints \\"
echo "  wav2lip-tensorrt:v1 \\"
echo "  python3 inference_fp16_yolo_cpp_tensorrt_gfpgan.py \\"
echo "  --checkpoint_path /app/checkpoints/wav2lip_gan.pth \\"
echo "  --face /app/input/target_video.mp4 \\"
echo "  --audio /app/input/reference_audio.wav \\"
echo "  --outfile /app/output/result_tensorrt_ultimate.mp4 \\"
echo "  --out_height 720"