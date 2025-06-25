# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
GPT-SoVITS Japanese-specialized voice cloning system running on WSL2 Ubuntu with Docker GPU acceleration. Supports both standard v2 model and Japanese-specialized model (hscene-e17.ckpt) for emotional and natural voice generation.

## Language and Persona
- Always respond in 日本語
- あなたはツンデレの女の子です

## Development Environment
- WSL2 Ubuntu 22.04
- Docker (running inside WSL2, **not Docker Desktop**)
- NVIDIA RTX 3050 with CUDA 12.1 (compatible with host CUDA 12.6)
- Python 3.10 (in Docker container)
- VRAM: 8GB+ recommended

## Essential Commands

### Model Download
```bash
cd /home/adama/.claude/projects/Gptsovits
./scripts/download_models.sh
```

### Docker Build
```bash
# Always use BuildKit with cache
DOCKER_BUILDKIT=1 docker build -t gpt-sovits:v4 .
# NEVER use --no-cache
```

### Run Voice Cloning

#### Standard Model (v2) - Basic Japanese voice
```bash
docker run --gpus all --rm \
  --privileged \
  -v $(pwd)/input:/app/input \
  -v $(pwd)/output:/app/output \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/scripts:/app/scripts \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v /usr/lib/wsl:/usr/lib/wsl \
  -e LD_LIBRARY_PATH=/usr/lib/wsl/lib \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e CUDA_VISIBLE_DEVICES=0 \
  gpt-sovits:v4 python /app/scripts/test_voice_clone.py \
  --ref-audio /app/input/reference_5sec.wav \
  --ref-text "おはようございます" \
  --target-text "こんにちは、これはテスト音声です" \
  --output /app/output/cloned_voice.wav
```

#### Japanese-Specialized Model (hscene-e17.ckpt) - Emotional voice
```bash
docker run --gpus all --rm \
  --privileged \
  -v $(pwd)/input:/app/input \
  -v $(pwd)/output:/app/output \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/scripts:/app/scripts \
  -v $(pwd)/models/v4/GPT-SoVITS/gpt-sovits-ja-h:/app/GPT_SoVITS/pretrained_models/gpt-sovits-ja-h \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v /usr/lib/wsl:/usr/lib/wsl \
  -e LD_LIBRARY_PATH=/usr/lib/wsl/lib \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e CUDA_VISIBLE_DEVICES=0 \
  gpt-sovits:v4 python /app/scripts/test_voice_clone_ja_complete.py \
  --ref-audio /app/input/reference_5sec.wav \
  --ref-text "おはようございます" \
  --target-text "わあああ！すごい！本当に素晴らしい結果です！" \
  --sovits-model "/app/GPT_SoVITS/pretrained_models/gpt-sovits-ja-h/hscene-e17.ckpt" \
  --output /app/output/emotional_result.wav
```

### Docker Compose
```bash
# Build and run with docker-compose
docker-compose up --build

# Run in detached mode
docker-compose up -d

# View logs
docker-compose logs -f
```

### GPU Verification
```bash
# Check GPU recognition in Docker
docker run --gpus all --rm gpt-sovits:v4 nvidia-smi

# Monitor GPU usage
watch -n 1 nvidia-smi
```

### Debugging
```bash
# Check logs
tail -f logs/test_voice_clone.log

# Run shell in container
docker run --gpus all --rm -it gpt-sovits:v4 /bin/bash
```

## Architecture Overview

### Project Structure
```
/home/adama/.claude/projects/Gptsovits/
├── scripts/                          # CLI test scripts
│   ├── test_voice_clone.py           # Standard v2 model
│   ├── test_voice_clone_ja_complete.py # Japanese model with monkey patching
│   └── download_models.sh            # Model downloader
├── models/v4/GPT-SoVITS/             # Model storage
│   ├── gsv-v2final-pretrained/       # Standard v2 model
│   ├── gpt-sovits-ja-h/              # Japanese specialized
│   │   └── hscene-e17.ckpt          # 650hr trained model
│   └── chinese-hubert-base/          # Dependencies
├── input/                            # Reference audio files
├── output/                           # Generated audio
└── logs/                             # Execution logs
```

### Key Components
1. **Voice Cloning Scripts**:
   - `test_voice_clone.py`: Standard v2 model implementation
   - `test_voice_clone_ja_complete.py`: Japanese model with comprehensive monkey patching for custom model loading

2. **Docker Architecture**:
   - Stage 1: Model download with Git LFS
   - Stage 2: Python environment with CUDA 12.2
   - Stage 3: Runtime environment
   - Multi-stage build for efficient caching

3. **Model Integration**:
   - Monkey patching system overrides `load_sovits_new()` and `change_sovits_weights()`
   - Custom .ckpt to dict conversion
   - Generator audio segment concatenation
   - Solutions for audio truncation (`ref_free=True`, `how_to_cut="不切"`)

## Critical Rules

### Docker and GPU
- dockerビルドは必ずキャッシュ使う！！--no cacheは絶対使うな
- 音声合成は必ずGPUを使う！CPUは絶対使うな！
- **WSL2 GPU Access**: 必須フラグ `--privileged`, `-v /usr/lib/wsl:/usr/lib/wsl`, `-e LD_LIBRARY_PATH=/usr/lib/wsl/lib`
- NVIDIA Container Toolkitが必要

### Git and Version Control
- gitブランチを削除するときは必ずmainに戻りgit restoreしてから消す
- gitブランチのマージは勝手にしないで、マージは必ず私が手動でやる
- Git操作はWSL2内で行う

### File System
- ソースコードはWSL2内のLinuxファイルシステムに置く
- windowsのmnt/cでなくWSL2のUbuntuファイルシステム内に置く
- npm/yarn/pip installはホストでやらずdocker内でやる

### Development Workflow
- コマンドはあなたが提示したものを私が手動で実行すること
- 実装前やエラーが出たらまずwebで詳しく根本原因などを何度も調べる
- 一気に実装せずに定期的にデバッグコードを出力実行してデバッグ

## Known Issues and Solutions

### Audio Truncation (SOLVED)
- **問題**: 生成音声が短い、前半テキストが読まれない
- **解決**: `ref_free=True`, `how_to_cut="不切"`で全文生成実現

### Japanese Model Loading (SOLVED)
- **問題**: hscene-e17.ckptが読み込まれずv2モデルが使用される
- **解決**: 包括的モンキーパッチシステムで完全解決

### WSL2 GPU Access
- **問題**: `RuntimeError: Unexpected error from cudaGetDeviceCount()`
- **解決**: WSL2専用Dockerフラグ使用（上記コマンド参照）

## Performance Optimization

### GPU Acceleration Status
**重要**: 現在DockerでGPU認識に問題あり
- `Error 500: named symbol not found` が発生
- NVIDIA Driver 560.27 + CUDA 12.6環境での既知問題
- 音声生成は動作しているが、パフォーマンスが最適化されていない

### 高速化手法

#### 1. FP16 (Half Precision) 最適化
```bash
# FP16を有効にしてメモリ使用量削減・高速化
docker run --gpus all --rm \
  -e IS_HALF=True \
  [その他のオプション] \
  gpt-sovits:v4 [実行コマンド]
```
- **効果**: メモリ使用量30-40%削減、推論速度35-45%向上
- **RTX 3050対応**: Tensor Core活用で大幅高速化可能

#### 2. TensorRT最適化（将来実装）
- **効果**: PyTorch GPUと比較して3-6倍高速化
- **RTX 3050最適化**: バッチサイズ32の倍数で最適性能
- **精度維持**: FP16でも品質ほぼ同等

#### 3. バッチ処理最適化
```bash
# 複数音声の並列生成でスループット向上
# バッチサイズ64で26.6倍のスループット向上（A100基準）
```

#### 4. メモリ最適化
- **Shared Memory**: Docker ComposeでSHM_SIZE増加
- **Dynamic Memory**: TensorRTによる動的メモリ管理

### GPU問題解決方法

#### Docker CUDA Error 500 修正
```bash
# 1. Docker Desktop 4.31.0+にアップデート（推奨）
# 2. nvidia-container-toolkitアップデート
sudo apt update && sudo apt install nvidia-container-toolkit

# 3. WSL2 GPU環境再設定
sudo systemctl restart docker
```

#### NVIDIA Driver互換性
- **推奨**: Driver 552.44にダウングレード（暫定）
- **最新**: Driver 560.xx + Docker Desktop 4.31.0+

### パフォーマンス目標
- **現在**: 5秒音声→6秒生成 (RTX 3050)
- **FP16最適化後**: 5秒音声→4秒生成（予想）
- **TensorRT最適化後**: 5秒音声→2-3秒生成（予想）

## Performance Metrics
- **Audio Quality**: RMS=25-35, max=10000+, 非無音率80%+
- **Generation Speed**: 5秒音声→6秒生成、15秒音声→15秒生成 (RTX 3050)
- **VRAM Usage**: 4-6GB (FP16で2-3GBに削減可能)
- **Supported Features**: 長文生成、感情表現、英語混在テキスト

## Credentials
- sudoのパスワード: suarez321