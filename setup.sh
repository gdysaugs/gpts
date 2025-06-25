#!/bin/bash

# =============================================================================
# GPT-SoVITS & LlamaCPP Projects Setup Script
# =============================================================================
# This script downloads all necessary models and sets up the environment
# for both GPT-SoVITS and LlamaCPP projects after cloning the repository.

set -e  # Exit on any error

echo "🚀 Setting up GPT-SoVITS & LlamaCPP projects..."
echo "=============================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
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

# Check if we're in the right directory
if [ ! -f "setup.sh" ]; then
    print_error "Please run this script from the project root directory"
    exit 1
fi

# Create necessary directories
print_header "📁 Creating directory structure..."
mkdir -p Gptsovits/{input,output,logs,models/v4/GPT-SoVITS}
mkdir -p llamacpp/{models,logs,logs/sessions}

print_status "Directory structure created"

# Setup GPT-SoVITS
print_header "🎵 Setting up GPT-SoVITS..."

cd Gptsovits

# Check if download script exists and run it
if [ -f "scripts/download_models.sh" ]; then
    print_status "Running GPT-SoVITS model download script..."
    chmod +x scripts/download_models.sh
    ./scripts/download_models.sh
else
    print_warning "GPT-SoVITS download script not found, manual setup required"
    print_status "Please download models manually:"
    echo "  - Standard v2 models to models/v4/GPT-SoVITS/gsv-v2final-pretrained/"
    echo "  - Japanese model hscene-e17.ckpt to models/v4/GPT-SoVITS/gpt-sovits-ja-h/"
    echo "  - Chinese HuBERT to models/v4/GPT-SoVITS/chinese-hubert-base/"
    echo "  - Chinese RoBERTa to models/v4/GPT-SoVITS/chinese-roberta-wwm-ext-large/"
fi

cd ..

# Setup LlamaCPP
print_header "🤖 Setting up LlamaCPP..."

cd llamacpp

# Check if setup script exists and run it
if [ -f "scripts/setup_model.sh" ]; then
    print_status "Running LlamaCPP model setup script..."
    chmod +x scripts/setup_model.sh
    ./scripts/setup_model.sh
else
    print_warning "LlamaCPP setup script not found, manual setup required"
    print_status "Please download the model manually:"
    echo "  - Download Berghof-NSFW-7B.i1-Q4_K_S.gguf to models/"
    echo "  - Or any other GGUF format model you prefer"
fi

cd ..

# Set permissions for all scripts
print_header "🔧 Setting script permissions..."
find . -name "*.sh" -type f -exec chmod +x {} \;
find . -name "*.py" -type f -path "*/scripts/*" -exec chmod +x {} \;

print_status "Script permissions set"

# Docker setup verification
print_header "🐳 Docker setup verification..."

if command -v docker &> /dev/null; then
    print_status "Docker is installed"
    
    if docker info | grep -q "nvidia"; then
        print_status "NVIDIA Docker runtime detected"
    else
        print_warning "NVIDIA Docker runtime not detected"
        print_status "Install nvidia-container-toolkit for GPU acceleration:"
        echo "  sudo apt update && sudo apt install nvidia-container-toolkit"
        echo "  sudo systemctl restart docker"
    fi
else
    print_error "Docker is not installed"
    print_status "Install Docker for WSL2:"
    echo "  curl -fsSL https://get.docker.com -o get-docker.sh"
    echo "  sudo sh get-docker.sh"
    echo "  sudo usermod -aG docker \$USER"
fi

# Final status
print_header "✅ Setup complete!"
echo ""
print_status "Next steps:"
echo "1. GPT-SoVITS: Build Docker image"
echo "   cd Gptsovits && DOCKER_BUILDKIT=1 docker build -t gpt-sovits:v4 ."
echo ""
echo "2. LlamaCPP: Build Docker image"
echo "   cd llamacpp && DOCKER_BUILDKIT=1 docker build -t llama-cpp-python:cuda ."
echo ""
echo "3. Test GPU access:"
echo "   docker run --gpus all --rm nvidia/cuda:12.1-runtime-ubuntu20.04 nvidia-smi"
echo ""

print_status "Ready to use! Check README.md for usage instructions."