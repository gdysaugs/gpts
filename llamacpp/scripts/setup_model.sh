#!/bin/bash
# LlamaCPP Model Setup Script
# ツンデレLLMモデルセットアップ

set -e

echo "🤖 ツンデレLlamaCPPモデルセットアップ開始！"
echo "ふん！必要なモデルをダウンロードしてあげるわよ..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}$1${NC}"
}

# モデルディレクトリ作成
mkdir -p models

# 必要なモデルファイル
MODEL_FILE="Berghof-NSFW-7B.i1-Q4_K_S.gguf"
MODEL_PATH="models/$MODEL_FILE"
EXPECTED_SIZE=4140374464  # 3.9GB in bytes

print_header "📋 LlamaCPP Model Requirements"
echo "Model: $MODEL_FILE"
echo "Size: 3.9GB (4,140,374,464 bytes)"
echo "Path: $MODEL_PATH"
echo ""

# モデルファイル存在確認
if [ -f "$MODEL_PATH" ]; then
    # サイズ確認
    ACTUAL_SIZE=$(stat -c%s "$MODEL_PATH" 2>/dev/null || stat -f%z "$MODEL_PATH" 2>/dev/null)
    
    if [ "$ACTUAL_SIZE" -eq "$EXPECTED_SIZE" ]; then
        print_status "✅ $MODEL_FILE already exists and size is correct!"
        echo "Size: $(($ACTUAL_SIZE / 1024 / 1024))MB"
        echo ""
        print_status "🎉 Model setup complete! Ready to use."
        exit 0
    else
        print_warning "⚠️ $MODEL_FILE exists but size is incorrect"
        echo "Expected: $(($EXPECTED_SIZE / 1024 / 1024))MB"
        echo "Actual: $(($ACTUAL_SIZE / 1024 / 1024))MB"
        echo "Removing corrupted file..."
        rm -f "$MODEL_PATH"
    fi
fi

print_header "📥 Model Download Options"
echo ""

# Option 1: Hugging Face CLI (推奨)
print_status "Option 1: Hugging Face CLI (Recommended)"
echo "1. Install Hugging Face CLI: pip install huggingface_hub"
echo "2. Download: huggingface-cli download TheBloke/Berghof-NSFW-7B-GGUF $MODEL_FILE --local-dir models/ --local-dir-use-symlinks False"
echo ""

# Option 2: Copy from Windows (if available)
WINDOWS_PATH="/mnt/c/Users/adama/Downloads/$MODEL_FILE"
if [ -f "$WINDOWS_PATH" ]; then
    print_status "Option 2: Copy from Windows Downloads (Available!)"
    echo "Found: $WINDOWS_PATH"
    echo "Execute: cp \"$WINDOWS_PATH\" models/"
    echo ""
    
    read -p "Copy from Windows Downloads now? [y/N]: " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_status "Copying from Windows Downloads..."
        cp "$WINDOWS_PATH" models/
        
        # Verify copied file
        if [ -f "$MODEL_PATH" ]; then
            COPIED_SIZE=$(stat -c%s "$MODEL_PATH" 2>/dev/null || stat -f%z "$MODEL_PATH" 2>/dev/null)
            if [ "$COPIED_SIZE" -eq "$EXPECTED_SIZE" ]; then
                print_status "✅ Copy successful! Size verified."
                echo "🎉 Model setup complete! Ready to use."
                exit 0
            else
                print_error "❌ Copy failed! Size mismatch."
                rm -f "$MODEL_PATH"
            fi
        fi
    fi
else
    print_warning "Option 2: Copy from Windows Downloads (Not Available)"
    echo "File not found: $WINDOWS_PATH"
    echo ""
fi

# Option 3: Direct download with wget (backup)
print_status "Option 3: Direct Download (Fallback)"
echo "If other methods fail, try manual download from:"
echo "https://huggingface.co/TheBloke/Berghof-NSFW-7B-GGUF/resolve/main/$MODEL_FILE"
echo ""

# Try Hugging Face CLI automatically
print_header "🚀 Attempting automatic download..."

if command -v huggingface-cli &> /dev/null; then
    print_status "Hugging Face CLI detected, attempting download..."
    
    # Try download
    if huggingface-cli download TheBloke/Berghof-NSFW-7B-GGUF "$MODEL_FILE" --local-dir models/ --local-dir-use-symlinks False; then
        # Verify downloaded file
        if [ -f "$MODEL_PATH" ]; then
            DOWNLOADED_SIZE=$(stat -c%s "$MODEL_PATH" 2>/dev/null || stat -f%z "$MODEL_PATH" 2>/dev/null)
            if [ "$DOWNLOADED_SIZE" -eq "$EXPECTED_SIZE" ]; then
                print_status "✅ Download successful! Size verified."
                echo "🎉 Model setup complete! Ready to use."
                exit 0
            else
                print_error "❌ Download failed! Size mismatch."
                rm -f "$MODEL_PATH"
            fi
        fi
    else
        print_warning "Automatic download failed"
    fi
else
    print_warning "Hugging Face CLI not installed"
    echo "Install with: pip install huggingface_hub"
fi

echo ""
print_header "🔧 Manual Setup Required"
print_error "Automatic model download failed!"
echo ""
print_status "Please manually download the model using one of these methods:"
echo ""
echo "Method 1 - Hugging Face CLI:"
echo "  pip install huggingface_hub"
echo "  huggingface-cli download TheBloke/Berghof-NSFW-7B-GGUF $MODEL_FILE --local-dir models/ --local-dir-use-symlinks False"
echo ""
echo "Method 2 - Browser Download:"
echo "  1. Visit: https://huggingface.co/TheBloke/Berghof-NSFW-7B-GGUF"
echo "  2. Download: $MODEL_FILE"
echo "  3. Copy to: $(pwd)/models/"
echo ""
echo "Method 3 - Windows Copy:"
echo "  cp \"/mnt/c/Users/adama/Downloads/$MODEL_FILE\" models/"
echo ""
print_status "After downloading, verify with:"
echo "  ls -lh models/$MODEL_FILE"
echo "  # Should show ~3.9GB file"
echo ""
print_status "📖 For detailed instructions, see: ../MODEL_DOWNLOAD_GUIDE.md"

exit 1