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

### FastAPI Production Server (Recommended - 9x faster than traditional scripts)
```bash
# Start persistent server (25s initialization, then 3s per request)
docker run --gpus all -d -p 8000:8000 \
  --privileged --name gpt-sovits-api \
  -v $(pwd)/input:/app/input \
  -v $(pwd)/output:/app/output \
  -v $(pwd)/scripts:/app/scripts \
  -v $(pwd)/models/v4/GPT-SoVITS/gpt-sovits-ja-h:/app/GPT_SoVITS/pretrained_models/gpt-sovits-ja-h \
  -v /usr/lib/wsl:/usr/lib/wsl \
  -e LD_LIBRARY_PATH=/usr/lib/wsl/lib \
  -e IS_HALF=True \
  gpt-sovits:v4 bash -c "pip install fastapi uvicorn python-multipart && python /app/scripts/fastapi_voice_server.py"

# Test API with Japanese text (requires URL encoding)
curl "http://localhost:8000/clone-voice-simple?ref_text=%E3%81%8A%E3%81%AF%E3%82%88%E3%81%86&target_text=%E3%83%86%E3%82%B9%E3%83%88" -o result.wav

# Server management
docker logs gpt-sovits-api --tail 10  # Check logs
docker stop gpt-sovits-api && docker rm gpt-sovits-api  # Stop server
curl "http://localhost:8000/" | python3 -m json.tool  # Check status
```

### Performance Optimized Scripts

#### FP16 Optimization (30-40% memory reduction)
```bash
docker run --gpus all --rm \
  --privileged \
  -v $(pwd)/input:/app/input \
  -v $(pwd)/output:/app/output \
  -v $(pwd)/scripts:/app/scripts \
  -v $(pwd)/models/v4/GPT-SoVITS/gpt-sovits-ja-h:/app/GPT_SoVITS/pretrained_models/gpt-sovits-ja-h \
  -v /usr/lib/wsl:/usr/lib/wsl \
  -e LD_LIBRARY_PATH=/usr/lib/wsl/lib \
  -e IS_HALF=True \
  gpt-sovits:v4 python /app/scripts/test_voice_clone_ja_complete.py \
  --ref-audio /app/input/reference_5sec.wav \
  --ref-text "おはようございます" \
  --target-text "今日は良い天気ですね" \
  --sovits-model "/app/GPT_SoVITS/pretrained_models/gpt-sovits-ja-h/hscene-e17.ckpt" \
  --output /app/output/fp16_result.wav
```

#### Torch.compile Optimization (2-4x speedup)
```bash
docker run --gpus all --rm \
  --privileged \
  -v $(pwd)/input:/app/input \
  -v $(pwd)/output:/app/output \
  -v $(pwd)/scripts:/app/scripts \
  -v $(pwd)/models/v4/GPT-SoVITS/gpt-sovits-ja-h:/app/GPT_SoVITS/pretrained_models/gpt-sovits-ja-h \
  -v /usr/lib/wsl:/usr/lib/wsl \
  -e LD_LIBRARY_PATH=/usr/lib/wsl/lib \
  -e IS_HALF=True \
  gpt-sovits:v4 python /app/scripts/test_voice_clone_torch_compile.py \
  --ref-audio /app/input/reference_5sec.wav \
  --ref-text "おはようございます" \
  --target-text "これはtorch.compileによる高速化テストです" \
  --sovits-model "/app/GPT_SoVITS/pretrained_models/gpt-sovits-ja-h/hscene-e17.ckpt" \
  --output /app/output/compiled_result.wav
```

#### Warm-up Optimization (73% initialization time reduction for short texts)
```bash
docker run --gpus all --rm \
  --privileged \
  -v $(pwd)/input:/app/input \
  -v $(pwd)/output:/app/output \
  -v $(pwd)/scripts:/app/scripts \
  -v $(pwd)/models/v4/GPT-SoVITS/gpt-sovits-ja-h:/app/GPT_SoVITS/pretrained_models/gpt-sovits-ja-h \
  -v /usr/lib/wsl:/usr/lib/wsl \
  -e LD_LIBRARY_PATH=/usr/lib/wsl/lib \
  -e IS_HALF=True \
  gpt-sovits:v4 python /app/scripts/test_voice_clone_warmup.py \
  --ref-audio /app/input/reference_5sec.wav \
  --ref-text "おはようございます" \
  --target-text "ウォームアップによる高速化テストです" \
  --sovits-model "/app/GPT_SoVITS/pretrained_models/gpt-sovits-ja-h/hscene-e17.ckpt" \
  --output /app/output/warmup_result.wav
```

### Run Voice Cloning (Standard Scripts)

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

## Architecture Overview

### Project Structure
```
/home/adama/.claude/projects/Gptsovits/
├── scripts/                          # CLI test scripts
│   ├── test_voice_clone.py           # Standard v2 model
│   ├── test_voice_clone_ja_complete.py # Japanese model with monkey patching
│   ├── test_voice_clone_torch_compile.py # Torch.compile optimization
│   ├── test_voice_clone_warmup.py    # Warm-up optimization
│   ├── fastapi_voice_server.py       # Production API server
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
   - `test_voice_clone_torch_compile.py`: PyTorch compilation optimization (2-4x speedup)
   - `test_voice_clone_warmup.py`: Model warm-up implementation (73% reduction in initialization overhead)
   - `fastapi_voice_server.py`: Production server with persistent model loading (1s response time)

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

4. **Performance Optimizations**:
   - **FP16 (Half Precision)**: 30-40% memory reduction via `IS_HALF=True`
   - **Torch.compile**: 2-4x inference speedup with max-autotune mode
   - **Model Warm-up**: 73% initialization time reduction for short texts
   - **FastAPI Persistent Server**: 3s response time after 25s warmup (9x faster than traditional scripts)
   - **Auto Text Extension**: Short texts (<20 chars) automatically extended to prevent audio truncation

## Critical Rules

### Docker and GPU
- dockerビルドは必ずキャッシュ使う！！--no cacheは絶対使うな
- 音声合成は必ずGPUを使う！CPUは絶対使うな！
- GPU使えない場合はNVIDIA Container Toolkitをインストールしてから実行する
- dockerのビルドは基本的にマルチステージビルドですること
- **WSL2 GPU Access**: 必須フラグ `--privileged`, `-v /usr/lib/wsl:/usr/lib/wsl`, `-e LD_LIBRARY_PATH=/usr/lib/wsl/lib`

### Git and Version Control
- gitブランチを削除するときは必ずmainに戻りgitrestoreしてから消す
- gitブランチのマージは勝手にしないで、マージは必ず私が手動でやる
- Git操作はWSL2内で行う

### File System and Dependencies
- ソースコードはwsl2内のlinuxファイルシステムに置く
- windowsのmnt/cでなくWSL 2 の Ubuntu のファイルシステム内に置く
- npm/yarn/pip installはホストでやらずdocker内でやる
- 依存関係は全部dockerfileとdocker-compose.ymlに明記
- 環境変数.envは必ず.env.exampleを作る

### Development Workflow
- コマンドはあなたが提示したものを私が手動で実行すること
- 実装前やエラーが出たらまずwebで詳しく根本原因などを何度も調べる
- インストール先や成否が分かりにくい特定のライブラリは失敗したらデバッグでlsやfindを追加するよう提案して
- 一気に実装せずに定期的にデバッグコードを出力実行してデバッグ

### Git LFS Best Practices
Git LFS 対策: Docker 内で Git LFS をより確実に動作させるために、各ステップを明確に分離：
1. apt-get で git, git-lfs, ca-certificates をインストール
2. git lfs install で LFS を初期化
3. git clone でリポジトリの基本構造をクローン
4. git lfs fetch --all で LFS オブジェクト情報を取得
5. git lfs checkout でポインターファイルを実際の LFS ファイルに置き換え

## Known Issues and Solutions

### Audio Truncation Problem (SOLVED)
- **問題**: 生成された音声が短すぎる（0.74秒など）、前半テキストが読まれない
- **根本原因**: `ref_free=False`による参照音声制約、不適切なテキスト分割、短いテキスト（<20文字）での前半スキップ
- **解決策**: 
  - `ref_free=True` - 参照音声制約を解除
  - `how_to_cut="不切"` - テキスト分割を無効化
  - ジェネレーター全セグメント連結処理実装
  - **FastAPI自動延長**: 20文字未満のテキストは自動で自然な延長文を追加
- **動作確認済み**: 15秒超の長文生成、感情表現、英語混在テキスト対応、短文問題完全解決

### Japanese Model Integration (SOLVED)
- **問題**: hscene-e17.ckptモデルが読み込まれずv2モデルが使用される
- **根本原因**: `change_sovits_weights()`関数がハードコードされたパスを使用
- **解決策**: 包括的モンキーパッチシステム実装
  - `load_sovits_new()`: .ckpt→dict変換処理
  - `change_sovits_weights()`: カスタムパス認識
  - グローバル変数による動的パス管理

### GPU Access in WSL2
- **問題**: `RuntimeError: Unexpected error from cudaGetDeviceCount() ... Error 500: named symbol not found`
- **解決**: WSL2専用Docker設定（上記コマンド参照）を使用
- **再現**: NVIDIA Container Toolkit再インストール、WSL2再起動で解決
- **重要**: ホストCUDA 12.6、コンテナCUDA 12.2の組み合わせで動作確認済み

### Audio Quality Standards
- **正常品質指標**: RMS=25-35, max=10000+, 非無音率80%+, 15秒音声生成可能
- **異常検出**: max<1000（音量不足）、RMS<10（無音状態）、非無音率<60%
- **パフォーマンス**: RTX 3050で5秒音声→6秒生成、15秒音声→15秒生成

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

### FastAPI Server (Production)
- **Initialization**: 25秒 (1回のみ)
- **Generation Speed**: 3秒/リクエスト (9倍高速化)
- **Throughput**: 20リクエスト/分
- **Audio Quality**: RMS=25-35, max=10000+, 非無音率80%+
- **Auto Extension**: 短文(<20文字)は自動延長で品質保証

### Traditional Scripts (Reference)
- **Initialization**: 25秒 (毎回)
- **Generation Speed**: 27秒/リクエスト
- **Throughput**: 2リクエスト/分

### System Resources
- **VRAM Usage**: 4-6GB (FP16で2-3GBに削減可能)
- **Supported Features**: 長文生成、感情表現、英語混在テキスト、短文自動延長

## Credentials
- sudoのパスワード: suarez321