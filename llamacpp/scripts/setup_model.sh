#!/bin/bash
# Model Setup Script for Llama.cpp Python

set -e

echo "🔧 Llama.cpp Model Setup Script"
echo "================================"

# 色付きメッセージ用関数
print_success() { echo -e "\033[32m✅ $1\033[0m"; }
print_warning() { echo -e "\033[33m⚠️ $1\033[0m"; }
print_error() { echo -e "\033[31m❌ $1\033[0m"; }
print_info() { echo -e "\033[34m🔍 $1\033[0m"; }

# ディレクトリ存在確認
check_directories() {
    print_info "Checking directories..."
    
    if [ ! -d "/app/models" ]; then
        print_error "Models directory not found!"
        exit 1
    fi
    
    if [ ! -d "/app/config" ]; then
        print_error "Config directory not found!"
        exit 1
    fi
    
    if [ ! -d "/app/logs" ]; then
        mkdir -p /app/logs
        print_success "Created logs directory"
    fi
    
    print_success "All directories present"
}

# モデルファイル確認とダウンロード
check_model_file() {
    print_info "Checking model file..."
    
    MODEL_FILE="/app/models/Berghof-NSFW-7B.i1-Q4_K_S.gguf"
    
    if [ ! -f "$MODEL_FILE" ]; then
        print_warning "Model file not found: $MODEL_FILE"
        print_info "Attempting to download model..."
        
        # モデルダウンロード
        DOWNLOAD_URL="https://huggingface.co/mradermacher/Berghof-NSFW-7B-i1-GGUF/resolve/main/Berghof-NSFW-7B.i1-Q4_K_S.gguf?download=true"
        
        if command -v wget &> /dev/null; then
            print_info "Downloading Berghof-NSFW-7B model (this may take a while)..."
            wget -O "$MODEL_FILE" "$DOWNLOAD_URL" || {
                print_error "Download failed with wget"
                print_info "Please download manually from:"
                print_info "https://huggingface.co/mradermacher/Berghof-NSFW-7B-i1-GGUF?not-for-all-audiences=true&show_file_info=Berghof-NSFW-7B.i1-Q4_K_S.gguf"
                exit 1
            }
        elif command -v curl &> /dev/null; then
            print_info "Downloading Berghof-NSFW-7B model (this may take a while)..."
            curl -L -o "$MODEL_FILE" "$DOWNLOAD_URL" || {
                print_error "Download failed with curl"
                print_info "Please download manually from:"
                print_info "https://huggingface.co/mradermacher/Berghof-NSFW-7B-i1-GGUF?not-for-all-audiences=true&show_file_info=Berghof-NSFW-7B.i1-Q4_K_S.gguf"
                exit 1
            }
        else
            print_error "Neither wget nor curl available for download"
            print_info "Please download manually from:"
            print_info "https://huggingface.co/mradermacher/Berghof-NSFW-7B-i1-GGUF?not-for-all-audiences=true&show_file_info=Berghof-NSFW-7B.i1-Q4_K_S.gguf"
            exit 1
        fi
        
        print_success "Model downloaded successfully"
    fi
    
    # ファイルサイズチェック
    FILE_SIZE=$(stat -c%s "$MODEL_FILE")
    FILE_SIZE_GB=$((FILE_SIZE / 1024 / 1024 / 1024))
    
    if [ $FILE_SIZE_GB -lt 1 ]; then
        print_error "Model file seems too small: ${FILE_SIZE_GB}GB"
        exit 1
    fi
    
    print_success "Model file found: ${FILE_SIZE_GB}GB"
}

# GPU確認
check_gpu() {
    print_info "Checking GPU availability..."
    
    if ! command -v nvidia-smi &> /dev/null; then
        print_error "nvidia-smi not found!"
        exit 1
    fi
    
    if ! nvidia-smi &> /dev/null; then
        print_error "nvidia-smi failed to run!"
        exit 1
    fi
    
    # GPU情報表示
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)
    print_success "GPU detected: $GPU_INFO"
}

# Python依存関係確認
check_python_deps() {
    print_info "Checking Python dependencies..."
    
    # llama-cpp-python確認
    if ! python -c "import llama_cpp" &> /dev/null; then
        print_error "llama-cpp-python not installed!"
        exit 1
    fi
    
    # CUDA対応確認（簡易チェック）
    LLAMA_VERSION=$(python -c "import llama_cpp; print(llama_cpp.__version__)" 2>/dev/null || echo "unknown")
    print_success "llama-cpp-python version: $LLAMA_VERSION"
    
    # その他の依存関係
    for package in yaml rich colorama; do
        if ! python -c "import $package" &> /dev/null; then
            print_warning "$package not found, installing..."
            pip install $package
        fi
    done
    
    print_success "All Python dependencies available"
}

# 設定ファイル確認
check_config() {
    print_info "Checking configuration..."
    
    CONFIG_FILE="/app/config/model_config.yaml"
    
    if [ ! -f "$CONFIG_FILE" ]; then
        print_error "Config file not found: $CONFIG_FILE"
        exit 1
    fi
    
    # YAML形式確認
    if ! python -c "import yaml; yaml.safe_load(open('$CONFIG_FILE'))" &> /dev/null; then
        print_error "Invalid YAML configuration!"
        exit 1
    fi
    
    print_success "Configuration file is valid"
}

# 権限確認
check_permissions() {
    print_info "Checking file permissions..."
    
    # スクリプトファイル実行権限
    chmod +x /app/scripts/*.py 2>/dev/null || true
    
    # ログディレクトリ書き込み権限
    if [ ! -w "/app/logs" ]; then
        print_error "Cannot write to logs directory!"
        exit 1
    fi
    
    print_success "File permissions OK"
}

# メイン実行
main() {
    echo "Starting system check..."
    
    check_directories
    check_model_file
    check_gpu
    check_python_deps
    check_config
    check_permissions
    
    echo
    print_success "✨ All checks passed! System is ready."
    print_info "Run 'python /app/scripts/model_test.py' for detailed testing"
    print_info "Run 'python /app/scripts/chat_cli.py' to start chatting"
}

# エラーハンドリング
trap 'print_error "Setup failed on line $LINENO"' ERR

# 実行
main "$@"