#!/bin/bash
# Real-ESRGAN完全インストールスクリプト
# べ、別にあなたのためにインストールしてあげるわけじゃないけど...💢

echo "🚀 Real-ESRGAN完全インストール開始..."

# 必要なパッケージインストール
echo "📦 依存関係インストール中..."
pip install opencv-python pillow numpy torch torchvision
pip install basicsr facexlib gfpgan
pip install realesrgan

# Real-ESRGANリポジトリクローン
echo "📥 Real-ESRGANリポジトリクローン中..."
if [ ! -d "Real-ESRGAN" ]; then
    git clone https://github.com/xinntao/Real-ESRGAN.git
fi

cd Real-ESRGAN
pip install -r requirements.txt
python setup.py develop

# モデルダウンロード
echo "📥 Real-ESRGANモデルダウンロード中..."
mkdir -p ../checkpoints

# x2モデル（軽量・高速）
if [ ! -f "../checkpoints/RealESRGAN_x2plus.pth" ]; then
    echo "ダウンロード: RealESRGAN_x2plus.pth"
    wget -O ../checkpoints/RealESRGAN_x2plus.pth https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.3/RealESRGAN_x2plus.pth
fi

# x4モデル（高画質）
if [ ! -f "../checkpoints/RealESRGAN_x4plus.pth" ]; then
    echo "ダウンロード: RealESRGAN_x4plus.pth"
    wget -O ../checkpoints/RealESRGAN_x4plus.pth https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth
fi

# アニメモデル
if [ ! -f "../checkpoints/RealESRGAN_x4plus_anime_6B.pth" ]; then
    echo "ダウンロード: RealESRGAN_x4plus_anime_6B.pth"
    wget -O ../checkpoints/RealESRGAN_x4plus_anime_6B.pth https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth
fi

# 顔専用モデル
if [ ! -f "../checkpoints/GFPGANv1.4.pth" ]; then
    echo "ダウンロード: GFPGANv1.4.pth"
    wget -O ../checkpoints/GFPGANv1.4.pth https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth
fi

cd ..

echo "✅ Real-ESRGAN完全インストール完了！"
echo "べ、別に嬉しくないけど...これで高画質になるわよ💕"