# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GPT-SoVITS is a Japanese-specialized voice cloning system running on WSL2 Ubuntu with Docker GPU acceleration. It generates emotional and natural-sounding Japanese speech using both standard and specialized models.

## Essential Commands

### Model Management
```bash
# Download all required models (run first)
./scripts/download_models.sh
```

### Docker Build
```bash
# Always use BuildKit with cache optimization
DOCKER_BUILDKIT=1 docker build -t gpt-sovits:v4 .

# NEVER use --no-cache unless explicitly required
```

### Production Server (Recommended - 9x faster)
```bash
# Start FastAPI server with GPU acceleration
docker run --gpus all -d -p 8000:8000 --privileged --name gpt-sovits-api \
  -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output \
  -v $(pwd)/models/v4/GPT-SoVITS/gpt-sovits-ja-h:/app/GPT_SoVITS/pretrained_models/gpt-sovits-ja-h \
  -v /usr/lib/wsl:/usr/lib/wsl -e LD_LIBRARY_PATH=/usr/lib/wsl/lib \
  gpt-sovits:v4 bash -c "pip install fastapi uvicorn python-multipart && python /app/scripts/fastapi_voice_server.py"

# API Usage: POST to localhost:8000/generate_audio
```

### CLI Scripts (Development/Testing)
```bash
# Standard v2 model
docker run --gpus all --rm -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output gpt-sovits:v4 python /app/scripts/test_voice_clone.py

# Japanese specialized model (hscene-e17.ckpt)
docker run --gpus all --rm -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output gpt-sovits:v4 python /app/scripts/test_voice_clone_ja_complete.py

# Performance optimized variants
docker run --gpus all --rm -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output gpt-sovits:v4 python /app/scripts/test_voice_clone_torch_compile.py
docker run --gpus all --rm -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output gpt-sovits:v4 python /app/scripts/test_voice_clone_warmup.py
```

## Architecture Overview

### Core Components
- **FastAPI Server** (`fastapi_voice_server.py`): Production REST API with persistent model loading
- **Voice Cloning Scripts** (12 variants): CLI interfaces for different optimization strategies
- **Model System**: Multi-model support (v2, v4, Japanese specialized) with monkey patching
- **Docker Multi-stage Build**: Model download → Python environment → optimized runtime

### Model Integration Patterns
- **Monkey Patching**: Dynamic model loading via `load_sovits_new()` and `change_sovits_weights()`
- **Format Conversion**: .ckpt to dict conversion for custom models
- **Multi-model Support**: Standard v2, v4, and Japanese specialized (hscene-e17.ckpt)

### Performance Architecture
- **Production**: FastAPI server (25s init, 3s/request, 20 req/min)
- **Development**: CLI scripts (25s init per execution, 27s/request)
- **Optimizations**: FP16, Torch.compile, warmup, ONNX variants

## System Requirements

### Platform Constraints
- **WSL2 Ubuntu 22.04** (required)
- **NVIDIA RTX 3050+** with 8GB+ VRAM
- **Docker** (native, NOT Docker Desktop)
- **NVIDIA Container Toolkit** installed

### CUDA Configuration
- Container: CUDA 12.1
- Host: Compatible with CUDA 12.6
- WSL2-specific mount: `/usr/lib/wsl:/usr/lib/wsl`

## Development Rules

### Docker Usage
- Always use `DOCKER_BUILDKIT=1` for builds
- Never use `--no-cache` unless explicitly required for debugging
- GPU acceleration mandatory (`--gpus all`)
- Use privileged mode for WSL2 compatibility

### Model Management
- Models stored in `models/v4/GPT-SoVITS/`
- Reference audio in `input/` directory (.wav format)
- Generated output in `output/` directory
- Japanese specialized model: `gpt-sovits-ja-h/hscene-e17.ckpt` (650-hour trained)

### Performance Optimization
- Use FastAPI server for production workloads
- CLI scripts for development/testing only
- Monitor GPU memory usage and audio quality metrics
- Apply FP16 + Torch.compile for optimal performance

### Code Patterns
- Monkey patching for dynamic model loading
- Multi-stage Docker builds mandatory
- Comprehensive logging and error handling
- Audio quality validation (RMS values, duration, silence detection)

## Directory Structure

```
scripts/              # 12 Python execution scripts
├── fastapi_voice_server.py    # Production API (preferred)
├── test_voice_clone.py        # Standard v2 model
├── test_voice_clone_ja_complete.py  # Japanese specialized
└── [9 optimization variants]
models/v4/GPT-SoVITS/ # Pre-trained models storage
input/                # Reference audio files (.wav)
output/               # Generated audio output
logs/                 # Execution logs
```

## Testing

No formal test framework - manual testing via 12 script variants.
Health checks via Docker container health monitoring and GPU availability validation.