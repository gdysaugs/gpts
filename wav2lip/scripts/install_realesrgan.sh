#!/bin/bash
# 🎭 ツンデレReal-ESRGANインストーラー
# べ、別にあなたのためにReal-ESRGANをインストールしてあげるわけじゃないけど...💕

echo "ふん！Real-ESRGANをインストールしてあげるわよ..."

# 作業ディレクトリ作成
mkdir -p /tmp/realesrgan_install
cd /tmp/realesrgan_install

# Real-ESRGAN ncnn-vulkan バイナリダウンロード
echo "📥 Real-ESRGAN バイナリダウンロード中..."
wget -q --show-progress https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-ubuntu.zip

if [ $? -eq 0 ]; then
    echo "✅ Real-ESRGANダウンロード完了！"
    
    # 解凍
    echo "📦 解凍中..."
    unzip -q realesrgan-ncnn-vulkan-20220424-ubuntu.zip
    
    # バイナリを/usr/local/binにコピー
    echo "📁 インストール中..."
    sudo cp realesrgan-ncnn-vulkan-20220424-ubuntu/realesrgan-ncnn-vulkan /usr/local/bin/
    sudo chmod +x /usr/local/bin/realesrgan-ncnn-vulkan
    
    # モデルファイルをコピー
    echo "📋 モデルファイルコピー中..."
    sudo mkdir -p /usr/local/share/realesrgan
    sudo cp -r realesrgan-ncnn-vulkan-20220424-ubuntu/models /usr/local/share/realesrgan/
    
    echo "✅ Real-ESRGANインストール完了！"
    
    # テスト実行
    echo "🧪 動作テスト中..."
    /usr/local/bin/realesrgan-ncnn-vulkan --help
    
    if [ $? -eq 0 ]; then
        echo "✨ Real-ESRGAN動作確認完了💕"
    else
        echo "❌ も、もう！動作テストが失敗したわよ！"
    fi
    
else
    echo "❌ も、もう！ダウンロード失敗よ！"
    echo "手動でダウンロードしてインストールしなさい："
    echo "https://github.com/xinntao/Real-ESRGAN/releases"
    exit 1
fi

# クリーンアップ
cd /
rm -rf /tmp/realesrgan_install

echo ""
echo "💝 完了！感謝しなさいよね💕"
echo "📊 インストール場所:"
echo "  - バイナリ: /usr/local/bin/realesrgan-ncnn-vulkan"
echo "  - モデル: /usr/local/share/realesrgan/models/"

echo ""
echo "💡 使用方法:"
echo "realesrgan-ncnn-vulkan -i input.jpg -o output.jpg -n RealESRGAN_x2plus"