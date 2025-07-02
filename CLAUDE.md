# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Multi-project AI system optimized for WSL2 Ubuntu + RTX 3050 GPU acceleration consisting of three specialized components:

- **GPT-SoVITS**: Japanese voice cloning with emotional expression (port 8000)
- **LlamaCPP**: GPU-optimized local LLM chat system (port 8001) 
- **Wav2Lip**: Lip-sync video generation with YOLO11 face detection (port 8002)

Working directory: `/home/adama/project/gpts/`

## Essential Build Commands

### Initial Setup
```bash
# Full environment setup (creates dirs, downloads models, checks GPU)
cd /home/adama/project/gpts
chmod +x setup.sh && ./setup.sh

# Build all images (ALWAYS use BuildKit cache, NEVER --no-cache)
cd Gptsovits && DOCKER_BUILDKIT=1 docker build -t gpt-sovits:v4 .
cd ../llamacpp && DOCKER_BUILDKIT=1 docker build -t llama-cpp-python:cuda .
cd ../wav2lip && DOCKER_BUILDKIT=1 docker build -t wav2lip-yolo:v1 .

# Verify builds
docker images | grep -E "gpt-sovits|llama-cpp|wav2lip"
```

### WSL2 GPU Requirements
All GPU Docker commands must include these flags:
```bash
--gpus all --privileged \
-v /usr/lib/wsl:/usr/lib/wsl \
-e LD_LIBRARY_PATH=/usr/lib/wsl/lib
```

## Primary Usage Commands

### GPT-SoVITS (Recommended: FastAPI Server)
```bash
cd /home/adama/project/gpts/Gptsovits

# Start server (20s init, then 3s per request)
docker run --gpus all -d -p 8000:8000 --name gpt-sovits-api \
  --privileged -v /usr/lib/wsl:/usr/lib/wsl -e LD_LIBRARY_PATH=/usr/lib/wsl/lib \
  -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output \
  -v $(pwd)/scripts:/app/scripts \
  -v $(pwd)/models/v4/GPT-SoVITS/gpt-sovits-ja-h:/app/GPT_SoVITS/pretrained_models/gpt-sovits-ja-h \
  gpt-sovits:v4 bash -c "pip install fastapi uvicorn python-multipart && python /app/scripts/fastapi_voice_server.py"

# Generate voice (3 second response)
curl -G "http://localhost:8000/clone-voice-simple" \
  --data-urlencode "ref_text=おはようございます" \
  --data-urlencode "target_text=音声生成のテストです" > output/test.wav

# Cleanup
docker stop gpt-sovits-api && docker rm gpt-sovits-api
```

### LlamaCPP (Direct CLI - Fastest)
```bash
cd /home/adama/project/gpts/llamacpp
docker run --gpus all --rm -it --privileged \
  -v /usr/lib/wsl:/usr/lib/wsl -e LD_LIBRARY_PATH=/usr/lib/wsl/lib \
  -v $(pwd)/models:/app/models -v $(pwd)/config:/app/config \
  -v $(pwd)/logs:/app/logs \
  llama-cpp-python:cuda python /app/scripts/chat_cli.py
```

### Wav2Lip (Quality-Fast Lip-Sync) ✅ WORKING
```bash
cd /home/adama/project/gpts/wav2lip

# 🚨 CRITICAL: 必ず --quality Fast を使用（ImprovedやEnhancedは口パクしない）

# 720p高画質（5秒処理、口パク保証）
docker run --gpus all --rm --privileged \
  -v /usr/lib/wsl:/usr/lib/wsl -e LD_LIBRARY_PATH=/usr/lib/wsl/lib \
  -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output \
  -v $(pwd)/checkpoints:/app/checkpoints -v $(pwd):/app/host \
  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  wav2lip-optimized:v1 bash -c "
  mkdir -p temp && cp /app/host/inference.py /app/inference.py 
  python /app/inference.py \
  --checkpoint_path /app/checkpoints/wav2lip_gan.pth \
  --face /app/input/target_video.mp4 \
  --audio /app/input/reference_audio.wav \
  --outfile /app/output/result_720p.mp4 \
  --out_height 720 \
  --quality Fast"

# 1080p高画質（7秒処理、口パク保証）
docker run --gpus all --rm --privileged \
  -v /usr/lib/wsl:/usr/lib/wsl -e LD_LIBRARY_PATH=/usr/lib/wsl/lib \
  -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output \
  -v $(pwd)/checkpoints:/app/checkpoints -v $(pwd):/app/host \
  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  wav2lip-optimized:v1 bash -c "
  mkdir -p temp && cp /app/host/inference.py /app/inference.py 
  python /app/inference.py \
  --checkpoint_path /app/checkpoints/wav2lip_gan.pth \
  --face /app/input/target_video.mp4 \
  --audio /app/input/reference_audio.wav \
  --outfile /app/output/result_1080p.mp4 \
  --out_height 1080 \
  --quality Fast"
```

## Development and Testing Commands

### System Verification
```bash
# Test GPU access in all containers
nvidia-smi
docker run --gpus all --rm --privileged -v /usr/lib/wsl:/usr/lib/wsl -e LD_LIBRARY_PATH=/usr/lib/wsl/lib gpt-sovits:v4 nvidia-smi
docker run --gpus all --rm --privileged -v /usr/lib/wsl:/usr/lib/wsl -e LD_LIBRARY_PATH=/usr/lib/wsl/lib llama-cpp-python:cuda nvidia-smi

# Test individual projects
cd Gptsovits && docker run --gpus all --rm --privileged -v /usr/lib/wsl:/usr/lib/wsl -e LD_LIBRARY_PATH=/usr/lib/wsl/lib -v $(pwd)/scripts:/app/scripts gpt-sovits:v4 python /app/scripts/test_voice_clone_warmup.py
cd llamacpp && docker run --rm --network host llama-cpp-python:cuda python /app/scripts/test_api.py
cd wav2lip && docker run --gpus all --rm --privileged -v /usr/lib/wsl:/usr/lib/wsl -e LD_LIBRARY_PATH=/usr/lib/wsl/lib wav2lip-yolo:v1 python /app/scripts/test_system.py
```

### Performance Benchmarking
```bash
# LlamaCPP performance test
cd llamacpp && docker run --gpus all --rm --privileged -v /usr/lib/wsl:/usr/lib/wsl -e LD_LIBRARY_PATH=/usr/lib/wsl/lib -v $(pwd):/app llama-cpp-python:cuda python /app/test_performance.py

# GPT-SoVITS different length tests
cd Gptsovits && docker run --gpus all --rm --privileged -v /usr/lib/wsl:/usr/lib/wsl -e LD_LIBRARY_PATH=/usr/lib/wsl/lib -v $(pwd):/app gpt-sovits:v4 python /app/scripts/test_different_lengths.py
```

## System Architecture

### Multi-Project Integration Pattern
Each project follows a consistent architecture:
- **Multi-stage Docker builds** with model downloading stages
- **FastAPI servers** for production deployment
- **CLI interfaces** for development and testing
- **NVIDIA CUDA optimization** for RTX 3050 specifically
- **WSL2 GPU integration** with specialized volume mounting

### Project-Specific Architectures

#### GPT-SoVITS
- **Core Engine**: `fastapi_voice_server.py` with one-time initialization
- **Model Management**: Monkey patching system for custom .ckpt loading
- **Optimization**: TensorCore + FP16 + torch.compile integration
- **Models**: Standard v2 (general) + Japanese hscene-e17 (emotional)

#### LlamaCPP  
- **LLM Engine**: `src/llm_engine.py` with YAML configuration management
- **API Server**: `src/api_server.py` with streaming and character presets
- **Optimization**: FP16 + Low VRAM mode for 4GB efficiency
- **Model**: Quantized Q4_K_S GGUF format (4.26GB)

#### Wav2Lip ✅ WORKING
- **Core Engine**: `inference.py` with GUI-disabled headless operation
- **Headless Mode**: Modified cv2.imshow/cv2.waitKey disabled for Docker compatibility
- **Processing Pipeline**: Face detection (2s) + Lip-sync generation (4s) + Video encoding
- **Output**: Successfully generates 240KB MP4 files from 63-frame input

### Performance Targets (RTX 3050)
- **GPT-SoVITS**: 3s per request (after 20s init), 4-6GB VRAM
- **LlamaCPP**: 1.03s response time (direct CLI), 3.86GB VRAM, 20-30 tokens/sec
- **Wav2Lip**: ✅ CONFIRMED: Quality Fast only for lip sync
  - 基本設定: 5秒 (元解像度)
  - 720p Fast: 5秒 (高画質、口パク保証)
  - 1080p Fast: 7秒 (最高画質、口パク保証)
  - ❌ Improved/Enhanced: 口パク消失

## Configuration Management

### Model File Requirements
Critical files that must be downloaded separately:
```bash
# GPT-SoVITS Japanese emotional model (148MB)
Gptsovits/models/v4/GPT-SoVITS/gpt-sovits-ja-h/hscene-e17.ckpt

# LlamaCPP quantized model (4.26GB)  
llamacpp/models/Berghof-NSFW-7B.i1-Q4_K_S.gguf

# Copy from Windows Downloads if available:
cp /mnt/c/Users/adama/Downloads/hscene-e17.ckpt Gptsovits/models/v4/GPT-SoVITS/gpt-sovits-ja-h/
cp /mnt/c/Users/adama/Downloads/Berghof-NSFW-7B.i1-Q4_K_S.gguf llamacpp/models/
```

### Key Configuration Files
- `llamacpp/config/model_config.yaml`: LLM parameters, GPU settings, character personalities
- `wav2lip/config/default_config.yaml`: Quality settings, YOLO detection parameters
- `setup.sh`: Environment detection and automated model downloads
- Each project's `docker-compose.yml`: Production deployment with health checks

### Port Allocation Strategy
- **GPT-SoVITS FastAPI**: 8000
- **LlamaCPP API**: 8001 (avoids conflicts)
- **Wav2Lip Server**: 8002 (if implemented)

## Critical Development Rules

### Docker and GPU
- **BuildKit cache**: ALWAYS use `DOCKER_BUILDKIT=1`, NEVER `--no-cache`
- **GPU inference**: REQUIRED for all AI operations, CPU fallback disabled
- **WSL2 GPU access**: Must include privileged mode and library mounting
- **NVIDIA Container Toolkit**: Required for GPU Docker access

### Model and Data Management
- **WSL2 filesystem**: All operations in Linux filesystem, avoid `/mnt/c/` (too slow)
- **Model downloads**: Can take 30-60 minutes initially
- **Absolute paths**: Required for all Docker volume mounts
- **Permission management**: Scripts need execute permissions after clone

### Development Workflow
- **Staged testing**: Verify each component individually before integration
- **Log monitoring**: All projects write detailed logs to `/logs/` directories
- **Error isolation**: Use project-specific containers to isolate issues
- **Performance monitoring**: Use `nvidia-smi` and `watch` for GPU utilization

## Common Issues and Solutions

### Wav2Lip Quality Settings Critical Issue (SOLVED) ✅
- **問題**: 高画質化すると口パクが消失し、元動画がそのまま再生される
- **原因**: `--quality Improved` や `Enhanced` が口パク機能を破壊
- **解決策**: 必ず `--quality Fast` を使用
```bash
# ✅ 口パクする設定
--quality Fast

# ❌ 口パクしない設定（使用禁止）
--quality Improved
--quality Enhanced
```
- **検証結果**:
  - 基本設定: 5秒処理、口パク正常
  - 720p Fast: 5秒処理、口パク正常
  - 1080p Fast: 7秒処理、口パク正常
  - 720p Improved: 6秒処理、口パク消失
  - 1080p Improved: 1分14秒処理、口パク消失

### Wav2Lip Qt GUI Errors (SOLVED) ✅
- **問題**: `qt.qpa.plugin: Could not find the Qt platform plugin "xcb"`
- **原因**: inference.pyがheadless環境でcv2.imshow()とcv2.waitKey()を呼び出し
- **解決策**: inference.pyでGUI機能を無効化
```python
# 修正箇所：inference.py 733-746行目
if not g_colab:
    # Display the frame (DISABLED for headless operation)
    # cv2.imshow コマンドをコメントアウト
    # cv2.waitKey コマンドをコメントアウト
    pass  # GUI機能を無効化
```
- **実行方法**: 修正済みinference.pyを使用してDocker実行
- **結果**: 正常に口パク動画生成完了（240KB出力ファイル確認済み）

### GPU Access Problems
```bash
# Install NVIDIA Container Toolkit (if missing)
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt update && sudo apt install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### Model File Issues
```bash
# Verify correct file sizes
ls -lh Gptsovits/models/v4/GPT-SoVITS/gpt-sovits-ja-h/hscene-e17.ckpt  # Should be 148MB
ls -lh llamacpp/models/Berghof-NSFW-7B.i1-Q4_K_S.gguf  # Should be 4.26GB
ls -lh wav2lip/models/wav2lip/wav2lip_gan.pth  # Should be downloaded by setup script
```

### Port Conflicts
```bash
# Check port usage
sudo lsof -i :8000 :8001 :8002

# Use alternative port mapping
docker run ... -p 8003:8000 ...  # Map to different host port
```

### Performance Issues
```bash
# Monitor GPU usage during operations
watch -n 1 nvidia-smi

# Check container resource usage
docker stats

# Verify WSL2 GPU integration
cat /proc/version  # Should contain "microsoft" for WSL2
```

## System Credentials
- sudo password: `suarez321`
- All operations require WSL2 with NVIDIA GPU support
- Projects designed for RTX 3050 but scale to other NVIDIA GPUs