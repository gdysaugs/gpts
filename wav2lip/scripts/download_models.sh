#!/bin/bash
# ツンデレモデルダウンローダー
# べ、別にあなたのためじゃないけど...モデルをダウンロードしてあげるわよ！

set -e

echo "🎭 ツンデレモデルダウンローダー起動！"
echo "ふん！必要なモデルをダウンロードしてあげるわよ..."

# モデルディレクトリ作成
mkdir -p /app/models/{wav2lip,yolo,face_detection,gfpgan}

echo "📁 モデルディレクトリ準備完了！"

# === Wav2Lipモデル ===
echo "🎤 Wav2Lipモデルダウンロード中..."

if [ ! -f "/app/models/wav2lip/wav2lip.pth" ]; then
    echo "べ、別に頑張ってるわけじゃないけど...標準モデルをダウンロード中..."
    gdown --id 1KQPKpuXHGdpGPiYcJwSZZIJLcMVFEUuC -O /app/models/wav2lip/wav2lip.pth
    echo "✅ wav2lip.pth ダウンロード完了！"
else
    echo "✅ wav2lip.pth 既に存在！"
fi

if [ ! -f "/app/models/wav2lip/wav2lip_gan.pth" ]; then
    echo "ふん！GANモデルも必要でしょ？ダウンロードしてあげるわ..."
    gdown --id 1fQtBSYEyuai_jHRhfKJYkpBhZJmryHDw -O /app/models/wav2lip/wav2lip_gan.pth
    echo "✅ wav2lip_gan.pth ダウンロード完了！"
else
    echo "✅ wav2lip_gan.pth 既に存在！"
fi

# === YOLO11モデル ===
echo "👁️ YOLO11モデルダウンロード中..."

# YOLO11n (最速)
if [ ! -f "/app/models/yolo/yolo11n.pt" ]; then
    echo "最速のYOLO11nをダウンロード中..."
    wget -O /app/models/yolo/yolo11n.pt https://github.com/ultralytics/assets/releases/download/v8.2.0/yolo11n.pt
    echo "✅ yolo11n.pt ダウンロード完了！"
else
    echo "✅ yolo11n.pt 既に存在！"
fi

# YOLO11s (バランス)
if [ ! -f "/app/models/yolo/yolo11s.pt" ]; then
    echo "バランス型のYOLO11sをダウンロード中..."
    wget -O /app/models/yolo/yolo11s.pt https://github.com/ultralytics/assets/releases/download/v8.2.0/yolo11s.pt
    echo "✅ yolo11s.pt ダウンロード完了！"
else
    echo "✅ yolo11s.pt 既に存在！"
fi

# YOLO11m (高精度) - オプション
if [ "${DOWNLOAD_YOLO11M:-false}" = "true" ]; then
    if [ ! -f "/app/models/yolo/yolo11m.pt" ]; then
        echo "高精度のYOLO11mをダウンロード中..."
        wget -O /app/models/yolo/yolo11m.pt https://github.com/ultralytics/assets/releases/download/v8.2.0/yolo11m.pt
        echo "✅ yolo11m.pt ダウンロード完了！"
    else
        echo "✅ yolo11m.pt 既に存在！"
    fi
fi

# === 顔検出モデル ===
echo "👤 顔検出モデルダウンロード中..."

# RetinaFace
if [ ! -f "/app/models/face_detection/retinaface_resnet50.pth" ]; then
    echo "RetinaFaceモデルをダウンロード中..."
    gdown --id 1BPqVN3ql7ybvGPRGEcGJtfYfVyBvQkBr -O /app/models/face_detection/retinaface_resnet50.pth
    echo "✅ RetinaFace ダウンロード完了！"
else
    echo "✅ RetinaFace 既に存在！"
fi

# dlib 68点モデル (オプション)
if [ "${DOWNLOAD_DLIB68:-false}" = "true" ]; then
    if [ ! -f "/app/models/face_detection/shape_predictor_68_face_landmarks.dat" ]; then
        echo "dlib 68点特徴点モデルをダウンロード中..."
        wget -O /app/models/face_detection/shape_predictor_68_face_landmarks.dat.bz2 \
            http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
        bunzip2 /app/models/face_detection/shape_predictor_68_face_landmarks.dat.bz2
        echo "✅ dlib 68点モデル ダウンロード完了！"
    else
        echo "✅ dlib 68点モデル 既に存在！"
    fi
fi

# === GFPGANモデル ===
echo "✨ GFPGAN顔画質向上モデルダウンロード中..."

if [ ! -f "/app/models/gfpgan/GFPGANv1.4.pth" ]; then
    echo "べ、別に画質向上が好きなわけじゃないけど...GFPGANをダウンロード中..."
    gdown --id 1GnpFZMojYlAGELiXBnqBGQSfDFGGb8e2 -O /app/models/gfpgan/GFPGANv1.4.pth
    echo "✅ GFPGAN ダウンロード完了！"
else
    echo "✅ GFPGAN 既に存在！"
fi

# === モバイル用軽量モデル (オプション) ===
if [ "${DOWNLOAD_MOBILE:-false}" = "true" ]; then
    echo "📱 モバイル用軽量モデルダウンロード中..."
    
    # MobileNet顔検出
    if [ ! -f "/app/models/face_detection/mobilenet.pth" ]; then
        echo "MobileNet顔検出をダウンロード中..."
        # checkpoints/mobilenet.pthが既に存在するので移動
        if [ -f "/app/checkpoints/mobilenet.pth" ]; then
            cp /app/checkpoints/mobilenet.pth /app/models/face_detection/mobilenet.pth
            echo "✅ MobileNet (ローカルから) 配置完了！"
        fi
    fi
fi

# === ダウンロード後検証 ===
echo "🔍 ダウンロード検証中..."

total_size=0
file_count=0

for model_dir in /app/models/*/; do
    echo "📂 $(basename "$model_dir"):"
    for file in "$model_dir"*.{pth,pt,dat} 2>/dev/null; do
        if [ -f "$file" ]; then
            size=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null || echo "0")
            size_mb=$((size / 1024 / 1024))
            total_size=$((total_size + size_mb))
            file_count=$((file_count + 1))
            echo "  ✅ $(basename "$file") (${size_mb}MB)"
        fi
    done
done

echo ""
echo "📊 ダウンロード完了サマリー:"
echo "  📁 ファイル数: ${file_count}個"
echo "  💾 総サイズ: ${total_size}MB"

# === RTX 3050最適化チェック ===
echo ""
echo "⚡ RTX 3050最適化チェック..."

if [ "$total_size" -gt 3000 ]; then
    echo "⚠️  大容量モデルが検出されました。RTX 3050では以下を推奨:"
    echo "   - YOLO11n使用 (最速)"
    echo "   - FP16最適化有効"
    echo "   - バッチサイズ1"
fi

echo ""
echo "🎉 べ、別にすごくないけど...全モデルのダウンロード完了よ！"
echo "💕 感謝しなさいよね！"

# === 使用例表示 ===
echo ""
echo "📝 使用例:"
echo "# 基本的な使い方"
echo "python /app/scripts/tsundere_cli.py generate \\"
echo "  --video input.mp4 \\"
echo "  --audio speech.wav \\"
echo "  --output result.mp4"
echo ""
echo "# 高品質設定"
echo "python /app/scripts/tsundere_cli.py generate \\"
echo "  --video input.mp4 \\"
echo "  --audio speech.wav \\"
echo "  --output result.mp4 \\"
echo "  --quality Enhanced \\"
echo "  --tensorrt"

echo ""
echo "🎭 ツンデレWav2Lip準備完了！"