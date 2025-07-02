#!/bin/bash
# 🎭 ツンデレCodeFormerモデルダウンローダー
# べ、別にあなたのためにモデルをダウンロードしてあげるわけじゃないけど...💕

echo "ふん！CodeFormerモデルをダウンロードしてあげるわよ..."

# ディレクトリ作成
mkdir -p checkpoints
cd checkpoints

# CodeFormerモデルダウンロード
echo "📥 CodeFormerモデルダウンロード中..."
if [ ! -f "codeformer.pth" ]; then
    echo "CodeFormer v0.1.0をダウンロード中..."
    wget -q --show-progress https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth
    
    if [ $? -eq 0 ]; then
        echo "✅ CodeFormerモデルダウンロード完了！"
        ls -lh codeformer.pth
    else
        echo "❌ も、もう！ダウンロード失敗よ！"
        echo "手動でダウンロードしてcheckpoints/に配置しなさい："
        echo "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth"
        exit 1
    fi
else
    echo "すでにCodeFormerモデルがあるじゃない！"
    ls -lh codeformer.pth
fi

# 顔検出モデル（必要な場合）
echo "📥 顔検出モデル確認中..."
if [ ! -f "detection_Resnet50_Final.pth" ]; then
    echo "顔検出モデルをダウンロード中..."
    wget -q --show-progress https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/detection_Resnet50_Final.pth
fi

# パーシング モデル（必要な場合）
if [ ! -f "parsing_parsenet.pth" ]; then
    echo "顔パーシングモデルをダウンロード中..."
    wget -q --show-progress https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/parsing_parsenet.pth
fi

cd ..

echo ""
echo "✨ 完了！感謝しなさいよね💕"
echo "📊 モデルサイズ:"
ls -lh checkpoints/*.pth | grep -E "(codeformer|detection|parsing)"

echo ""
echo "💡 使用方法:"
echo "python inference_fp16_yolo_codeformer.py \\"
echo "  --checkpoint_path checkpoints/wav2lip_gan.pth \\"
echo "  --face input/target_video.mp4 \\"
echo "  --audio input/reference_audio.wav \\"
echo "  --outfile output/result_codeformer_ultimate.mp4 \\"
echo "  --fidelity_weight 0.7 \\"
echo "  --out_height 720"