# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Super Wav2Lip is a GPU-optimized Docker-based lip-sync video generation system based on wav2lip-onnx-HQ. It generates professional-quality lip-sync videos using real wav2lip algorithms with masking and audio synchronization.

**Critical Requirement**: This system is GPU-only. CPU execution is disabled due to 50-100x slower performance.

## Essential Build Commands

### Docker Build and Setup
```bash
# Build GPU-optimized Docker image (7.95GB)
docker build -t super-wav2lip:v1-gpu-ultimate .

# Environment setup (if needed)
chmod +x scripts/setup.sh && ./scripts/setup.sh
```

### WSL2 GPU Requirements
All GPU Docker commands must include these critical flags for WSL2:
```bash
--gpus all --privileged \
-v /usr/lib/wsl:/usr/lib/wsl \
-e LD_LIBRARY_PATH="/usr/lib/wsl/lib:/usr/local/lib/python3.10/dist-packages/nvidia/curand/lib:/usr/local/lib/python3.10/dist-packages/nvidia/cublas/lib:/usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib:/usr/local/lib/python3.10/dist-packages/nvidia/cufft/lib:/usr/local/lib/python3.10/dist-packages/nvidia/cuda_runtime/lib:/usr/local/lib/python3.10/dist-packages/nvidia/cuda_nvrtc/lib"
```

## Primary Usage Commands

### Standard High-Quality Execution (Recommended)
```bash
docker run --gpus all --rm --privileged \
  -v /usr/lib/wsl:/usr/lib/wsl \
  -e LD_LIBRARY_PATH="/usr/lib/wsl/lib:/usr/local/lib/python3.10/dist-packages/nvidia/curand/lib:/usr/local/lib/python3.10/dist-packages/nvidia/cublas/lib:/usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib:/usr/local/lib/python3.10/dist-packages/nvidia/cufft/lib:/usr/local/lib/python3.10/dist-packages/nvidia/cuda_runtime/lib:/usr/local/lib/python3.10/dist-packages/nvidia/cuda_nvrtc/lib" \
  -v $(pwd)/input:/app/input \
  -v $(pwd)/output:/app/output \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/src:/app/src \
  super-wav2lip:v1-gpu-ultimate python /app/src/lipsync_cli.py \
  --checkpoint_path /app/models/onnx/wav2lip_gan.onnx \
  --face /app/input/videos/source_video.mp4 \
  --audio /app/input/audio/target_audio.wav \
  --outfile /app/output/result_premium.mp4 \
  --verbose
```

### Direct Core Engine Execution (Advanced)
```bash
docker run --gpus all --rm --privileged \
  -v /usr/lib/wsl:/usr/lib/wsl \
  -e LD_LIBRARY_PATH="[same as above]" \
  -v $(pwd)/input:/app/input \
  -v $(pwd)/output:/app/output \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/src:/app/src \
  -w /app/src \
  super-wav2lip:v1-gpu-ultimate python inference_onnxModel.py \
  --checkpoint_path /app/models/onnx/wav2lip_gan.onnx \
  --face /app/input/videos/source_video.mp4 \
  --audio /app/input/audio/target_audio.wav \
  --outfile /app/output/result_ultra.mp4 \
  --hq_output \
  --face_mask \
  --enhancer gfpgan \
  --blending 10
```

## System Architecture

### Two-Layer Architecture
1. **CLI Wrapper** (`src/lipsync_cli.py`): User-friendly interface with validation and preprocessing
2. **Core Engine** (`src/inference_onnxModel.py`): The actual wav2lip-onnx-HQ implementation (22,121 lines)

### Processing Pipeline
1. **Video Frame Extraction**: Input video → individual frames
2. **Audio Processing**: Input audio → mel spectrogram generation
3. **Face Detection**: RetinaFace ONNX model for face detection and alignment
4. **Face Recognition**: Identity verification using recognition.onnx
5. **Lip-Sync Generation**: Core wav2lip ONNX model inference
6. **Face Enhancement** (optional): GFPGAN, GPEN, CodeFormer, RestoreFormer
7. **Video Reconstruction**: Frames + audio → final output video

### ONNX Runtime Configuration
- **Primary Provider**: CUDAExecutionProvider with cuDNN optimization
- **Fallback Provider**: CPUExecutionProvider (strongly discouraged)
- **Memory Management**: Optimized for RTX 3050+ with 4-6GB VRAM usage

## Required Model Files

Critical ONNX models that must be downloaded separately:
```bash
# Required models from Google Drive: 
# https://drive.google.com/drive/folders/1BGl9bmMtlGEMx_wwKufJrZChFyqjnlsQ

# Main lip-sync models:
models/onnx/wav2lip_gan.onnx     # GAN model (recommended, high quality)
models/onnx/wav2lip_384.onnx     # Base model (fast, compatibility)

# Face recognition:
src/faceID/recognition.onnx      # Face ID verification (required)

# Pre-installed:
src/utils/scrfd_2.5g_bnkps.onnx  # Face detection (3.29MB)
```

## Quality Modes

### Level 1: GAN Model (Recommended)
- Model: `wav2lip_gan.onnx`
- Processing: 7-10 seconds for 5-second video
- Output: 0.95MB, natural high quality

### Level 2: Fast Execution
- Model: `wav2lip_384.onnx`
- Processing: 3-5 seconds for 5-second video
- Output: 0.9MB, standard quality

### Level 3: Ultra High Quality
- Model: `wav2lip_gan.onnx` + `--hq_output --face_mask --enhancer gfpgan`
- Processing: 15-25 seconds for 5-second video
- Output: 3.4MB, CRF=5.0 maximum quality

## Performance Targets (RTX 3050)

### Achieved Performance (RTX 3050 + batch_size=8 optimization)
- **Face Detection**: 59+ it/s (batch_size=8 optimized)
- **Lip-Sync Generation**: 25+ it/s (batch_size=8 optimized)
- **Standard Mode**: 17.3 seconds (42% faster than baseline)
- **GFPGAN Mode**: 40.0 seconds (25% faster than baseline)
- **Memory Usage**: 1.67GB VRAM / 4GB total (59% headroom)

## Critical CUDA Library Path Issue

**Problem**: ONNX Runtime requires specific CUDA library paths for WSL2
**Solution**: Always use the complete LD_LIBRARY_PATH shown in usage commands above
**Symptoms if missing**: "libcurand.so.10: cannot open shared object file" errors

## Debugging Commands

### GPU Status Verification
```bash
# Host GPU check
nvidia-smi

# Container GPU check
docker run --gpus all --rm super-wav2lip:v1-gpu-ultimate nvidia-smi

# System verification
docker run --gpus all --rm super-wav2lip:v1-gpu-ultimate /app/check_system.sh
```

### Performance Monitoring
```bash
# Real-time GPU monitoring during processing
watch -n 1 nvidia-smi

# Verbose output for debugging
--verbose  # Add to any lipsync_cli.py command
```

## Common Issues

### CUDA Library Errors
Use the complete LD_LIBRARY_PATH. Partial paths will cause "cannot open shared object file" errors.

### GUI Errors (Resolved)
inference_onnxModel.py has GUI functions disabled for headless Docker operation.

### Model File Errors
Verify required models are downloaded and placed correctly:
```bash
ls -la models/onnx/wav2lip_gan.onnx        # Should be ~139MB
ls -la src/faceID/recognition.onnx         # Should be ~92MB
```

## Docker Environment Specifications

- **Base**: Ubuntu 22.04 + Python 3.10
- **GPU**: PyTorch 2.5.1+cu121, ONNX Runtime GPU 1.16.0+
- **Size**: 7.95GB optimized for CUDA 12.1
- **Dependencies**: Complete wav2lip-onnx-HQ integration with all enhancers