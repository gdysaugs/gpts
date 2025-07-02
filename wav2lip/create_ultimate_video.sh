#!/bin/bash
"""
🎭 ツンデレWav2Lip + CodeFormer 究極統合スクリプト
べ、別にあなたのために究極の高画質口パク動画を作ってあげるわけじゃないけど...💢

簡単3ステップ:
1. Wav2Lip FP16+YOLO で口パク生成
2. フレーム分割 → CodeFormer高画質化
3. 高画質動画再構築

Author: ツンデレAI
Version: Ultimate Shell v1.0
"""

# デフォルト設定
INPUT_VIDEO="${1:-input/target_video.mp4}"
INPUT_AUDIO="${2:-input/reference_audio.wav}"
OUTPUT_VIDEO="${3:-output/ultimate_high_quality.mp4}"

echo "🚀 べ、別に究極パイプラインを開始するわけじゃないけど...💢"
echo "📹 動画: $INPUT_VIDEO"
echo "🎵 音声: $INPUT_AUDIO"
echo "💾 出力: $OUTPUT_VIDEO"

# 作業ディレクトリ
WORK_DIR="temp/ultimate_work"
mkdir -p "$WORK_DIR"

echo ""
echo "📍 ステップ1: Wav2Lip FP16+YOLO 口パク生成"
WAV2LIP_OUTPUT="$WORK_DIR/wav2lip_result.mp4"

docker run --gpus all --rm --privileged \
  -v /usr/lib/wsl:/usr/lib/wsl -e LD_LIBRARY_PATH=/usr/lib/wsl/lib \
  -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output \
  -v $(pwd)/checkpoints:/app/checkpoints -v $(pwd):/app/host \
  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  wav2lip-optimized:v2 bash -c "
  rm -f last_detected_face.pkl temp/face_detection_cache.pkl
  mkdir -p temp && cp /app/host/inference_fp16_yolo.py /app/inference_fp16_yolo.py 
  python /app/inference_fp16_yolo.py \
  --checkpoint_path /app/checkpoints/wav2lip_gan.pth \
  --face /app/input/$(basename $INPUT_VIDEO) \
  --audio /app/input/$(basename $INPUT_AUDIO) \
  --outfile /app/host/$WAV2LIP_OUTPUT \
  --out_height 720 \
  --quality Fast"

if [ ! -f "$WAV2LIP_OUTPUT" ]; then
    echo "❌ Wav2Lip処理失敗...もう！💢"
    exit 1
fi

echo "✅ Wav2Lip処理完了よ！"

echo ""
echo "📍 ステップ2: フレーム分割"
FRAMES_DIR="$WORK_DIR/frames"
mkdir -p "$FRAMES_DIR"

ffmpeg -y -i "$WAV2LIP_OUTPUT" -vf fps=25 "$FRAMES_DIR/frame_%06d.png"

FRAME_COUNT=$(ls "$FRAMES_DIR"/frame_*.png | wc -l)
echo "✅ ${FRAME_COUNT}フレーム抽出完了よ！"

echo ""
echo "📍 ステップ3: CodeFormer高画質化 (${FRAME_COUNT}フレーム)"
ENHANCED_DIR="$WORK_DIR/enhanced"
mkdir -p "$ENHANCED_DIR"

# CodeFormer Docker環境で各フレームを処理
cd codeformer

echo "🎨 CodeFormer高画質化処理開始..."
FRAME_NUM=0
for frame in "../$FRAMES_DIR"/frame_*.png; do
    FRAME_NUM=$((FRAME_NUM + 1))
    frame_name=$(basename "$frame")
    enhanced_path="../$ENHANCED_DIR/$frame_name"
    
    echo -n "🎨 フレーム $FRAME_NUM/$FRAME_COUNT 処理中... "
    
    # CodeFormer実行
    docker compose exec -T codeformer python /app/codeformer_face_fix.py \
      --input "/app/host/$frame" \
      --output "/app/host/$enhanced_path" \
      --fidelity 0.8 \
      --blend-strength 0.8 >/dev/null 2>&1
    
    if [ -f "$enhanced_path" ]; then
        echo "✅"
    else
        echo "⚠️ (元フレームコピー)"
        cp "$frame" "$enhanced_path"
    fi
done

cd ..

echo "✅ CodeFormer高画質化完了！"

echo ""
echo "📍 ステップ4: 高画質動画再構築+音声合成"

ffmpeg -y \
  -framerate 25 \
  -i "$ENHANCED_DIR/frame_%06d.png" \
  -i "$INPUT_AUDIO" \
  -c:v libx264 \
  -preset medium \
  -crf 18 \
  -pix_fmt yuv420p \
  -c:a aac \
  -b:a 128k \
  -shortest \
  "$OUTPUT_VIDEO"

if [ -f "$OUTPUT_VIDEO" ]; then
    echo ""
    echo "🎉 究極パイプライン完了！"
    echo "✅ 究極高画質口パク動画完成: $OUTPUT_VIDEO"
    echo "べ、別に完璧に作ったからって自慢するわけじゃないけど...💕"
    echo "感謝しなさいよね！"
    
    # 作業ディレクトリクリーンアップ
    rm -rf "$WORK_DIR"
else
    echo "❌ 動画再構築失敗...もう！💢"
    exit 1
fi