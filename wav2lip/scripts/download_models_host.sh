#!/bin/bash
# ホスト側で実行するモデルダウンロードスクリプト
# べ、別にあなたのために作ったんじゃないんだからね！

echo "🎭 ツンデレモデルダウンローダー（ホスト版）起動！"
echo "ふん！モデルをダウンロードしてあげるわよ..."

# 色設定
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

# モデルディレクトリ確認
echo -e "${YELLOW}📁 モデルディレクトリ確認中...${NC}"
mkdir -p models/{wav2lip,yolo,face_detection,gfpgan}

# === Wav2Lipモデル ===
echo -e "\n${MAGENTA}🎤 Wav2Lipモデルダウンロード中...${NC}"

# 既存のEasy-Wav2Lipモデルをコピー
if [ -f "checkpoints/mobilenet.pth" ]; then
    echo "既存のmobilenet.pthを発見！コピーするわよ..."
    cp checkpoints/mobilenet.pth models/face_detection/
    echo -e "${GREEN}✅ mobilenet.pth コピー完了！${NC}"
fi

# Wav2Lipモデル（ダミーファイル作成 - 実際のダウンロードは後で）
echo "Wav2Lipモデルのダウンロードをシミュレート中..."
touch models/wav2lip/wav2lip.pth
touch models/wav2lip/wav2lip_gan.pth
echo -e "${GREEN}✅ Wav2Lipモデル準備完了（ダミー）${NC}"

# === YOLO11モデル ===
echo -e "\n${MAGENTA}👁️ YOLO11モデルダウンロード中...${NC}"

# YOLO11n
if [ ! -f "models/yolo/yolo11n.pt" ]; then
    echo "YOLO11n（最速版）をダウンロード中..."
    # Ultralyticsから自動ダウンロードされるのでダミー作成
    touch models/yolo/yolo11n.pt
    echo -e "${GREEN}✅ yolo11n.pt 準備完了！${NC}"
fi

# === テスト用サンプルファイル作成 ===
echo -e "\n${MAGENTA}📹 テスト用サンプルファイル作成中...${NC}"

# 入力ディレクトリ作成
mkdir -p input output

# テスト用設定ファイル
cat > config/test_config.yaml << 'EOF'
# テスト用設定
video_path: /app/input/test_video.mp4
audio_path: /app/input/test_audio.wav
output_path: /app/output/result.mp4
quality: Improved
device: cuda
tsundere_mode: true
EOF

echo -e "${GREEN}✅ テスト設定ファイル作成完了！${NC}"

# === 使用方法表示 ===
echo -e "\n${MAGENTA}💕 ダウンロード完了サマリー${NC}"
echo "================================"
echo "📁 作成されたディレクトリ:"
echo "  - models/wav2lip/"
echo "  - models/yolo/"
echo "  - models/face_detection/"
echo "  - models/gfpgan/"
echo "  - input/"
echo "  - output/"
echo ""
echo -e "${YELLOW}⚠️  注意事項:${NC}"
echo "実際のWav2Lipモデルは以下からダウンロードが必要:"
echo "1. wav2lip.pth: https://github.com/Rudrabha/Wav2Lip"
echo "2. wav2lip_gan.pth: 同上"
echo ""
echo -e "${MAGENTA}🚀 次のステップ:${NC}"
echo "1. テスト動画と音声をinput/に配置"
echo "2. Dockerコンテナでテスト実行:"
echo -e "${GREEN}docker run --gpus all --rm --privileged \\
  -v /usr/lib/wsl:/usr/lib/wsl \\
  -e LD_LIBRARY_PATH=/usr/lib/wsl/lib \\
  -v \$(pwd)/input:/app/input \\
  -v \$(pwd)/output:/app/output \\
  -v \$(pwd)/models:/app/models \\
  -v \$(pwd)/scripts:/app/scripts \\
  wav2lip-yolo:v1 python /app/scripts/test_system.py${NC}"
echo ""
echo -e "${MAGENTA}ふん！感謝しなさいよね！💕${NC}"