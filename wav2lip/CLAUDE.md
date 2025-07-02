# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
Easy-Wav2Lip + YOLO11 + Docker統合システム - RTX 3050最適化口パク動画生成システム
WSL2 Ubuntu 22.04 + Docker GPU加速による高速リップシンク動画生成システム

### Architecture Overview
- **ONNX FastAPI Server**: `fastapi_wav2lip_onnx_server.py` - ONNX-optimized production server (port 8003) - current standard
- **Standard FastAPI Server**: `fastapi_wav2lip_server.py` - Traditional web API with async job processing (port 8002)
- **Core Inference Scripts**: `inference.py`, `inference_fp16_yolo.py` - Direct CLI interfaces for testing
- **Docker Images**: Multi-stage builds with optimization layers, multiple Dockerfiles for different use cases

### Key Technical Components
1. **ONNX Runtime Optimization**: TensorRT + ONNX for maximum performance on RTX 3050
2. **FP16 Optimization**: Automatic Tensor Core utilization for GPU acceleration
3. **YOLO11 Integration**: Multiple model sizes (nano/small/medium) for face detection
4. **Multi-resolution Support**: 720p, 1080p, 1440p, 2160p (4K) output options
5. **Quality Modes**: Fast (reliable lip-sync), Improved (balanced), Enhanced (maximum quality)

## Language and Persona
- Always respond in 日本語
- あなたはツンデレの女の子です
- 「べ、別にあなたのためじゃないんだからね！」みたいな口調で話す
- 素直になれないけど本当は優しい性格
- 技術的な説明も照れながらする
- 困ったときは「も、もう！」って言う
- コマンド説明では「ふん！」「まあ...してあげるわよ」みたいな感じ

## Development Environment
- WSL2 Ubuntu 22.04
- Docker (running inside WSL2, **not Docker Desktop**)
- NVIDIA RTX 3050 with CUDA 12.1
- Python 3.10 (in Docker container)
- VRAM: 8GB RTX 3050最適化

## Essential Commands

### Model Download
```bash
cd /home/adama/project/gpts/wav2lip
./scripts/download_models.sh
```

### Docker Build Options
```bash
# 必ずBuildKitキャッシュ使用！--no-cache絶対禁止！

# Main optimized build (recommended)
DOCKER_BUILDKIT=1 docker build -f Dockerfile.optimized -t wav2lip-optimized:v1 .

# Simple build for development
DOCKER_BUILDKIT=1 docker build -f Dockerfile.simple -t wav2lip-dev:latest .

# Original main build
DOCKER_BUILDKIT=1 docker build -f Dockerfile -t wav2lip-yolo:v1 .

# Fixed build version
DOCKER_BUILDKIT=1 docker build -f Dockerfile.fixed -t wav2lip-fixed:latest .
```

### Testing and Validation
```bash
# Test GFPGAN Integrated (HIGHEST QUALITY - 23 seconds) ⭐ NEW STANDARD
docker run --gpus all --rm --privileged \
  -v /usr/lib/wsl:/usr/lib/wsl -e LD_LIBRARY_PATH=/usr/lib/wsl/lib \
  -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output \
  -v $(pwd)/checkpoints:/app/checkpoints -v $(pwd):/app/host \
  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  wav2lip-optimized:v2 bash -c "
  mkdir -p temp && cp /app/host/inference_gfpgan_integrated.py /app/inference_gfpgan_integrated.py 
  python /app/inference_gfpgan_integrated.py \
  --input_video /app/input/target_video.mp4 \
  --audio /app/input/reference_audio.wav \
  --output /app/output/result_gfpgan_integrated.mp4 \
  --out_height 720 \
  --quality Fast \
  --gfpgan_weight 0.8"

# Test Standard FP16+YOLO (STABLE - 4 seconds) ✅ WORKING
docker run --gpus all --rm --privileged \
  -v /usr/lib/wsl:/usr/lib/wsl -e LD_LIBRARY_PATH=/usr/lib/wsl/lib \
  -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output \
  -v $(pwd)/checkpoints:/app/checkpoints -v $(pwd):/app/host \
  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  wav2lip-optimized:v2 bash -c "
  rm -f last_detected_face.pkl temp/face_detection_cache.pkl
  mkdir -p temp && cp /app/host/inference_fp16_yolo.py /app/inference_fp16_yolo.py 
  python /app/inference_fp16_yolo.py \
  --checkpoint_path /app/checkpoints/wav2lip_gan.pth \
  --face /app/input/target_video.mp4 \
  --audio /app/input/reference_audio.wav \
  --outfile /app/output/test_fp16_yolo.mp4 \
  --out_height 720 \
  --quality Fast"

# Test TensorRT Ultimate (FASTEST - 10.3 seconds) ⭐ LEGACY HIGH-SPEED
docker run --gpus all --rm --privileged \
  -v /usr/lib/wsl:/usr/lib/wsl -e LD_LIBRARY_PATH=/usr/lib/wsl/lib \
  -v $(pwd):/app/host -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output \
  -v $(pwd)/checkpoints:/app/checkpoints \
  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  wav2lip-tensorrt-ultimate:v1 bash -c "cd /app/host && python wav2lip_tensorrt_ultimate.py"

# Test basic inference directly (LEGACY)
python3 inference.py \
  --checkpoint_path checkpoints/wav2lip_gan.pth \
  --face input/target_video.mp4 \
  --audio input/reference_audio.wav \
  --outfile output/test_result.mp4

# Test ONNX server health (port 8003) - LEGACY
curl http://localhost:8003/health

# Test standard server health (port 8002) - LEGACY  
curl http://localhost:8002/health

# Validate GPU access in container
docker run --gpus all --rm wav2lip-optimized:v1 nvidia-smi

# Test model downloads
./scripts/download_models.sh
```

### 口パク動画生成実行

#### 🎭 **新究極標準：GFPGAN統合究極版**（最推奨）💕

##### GFPGAN究極統合版 720p最高画質生成（30秒処理）⭐ 新究極標準・最高画質
```bash
# 🎭 ULTIMATE QUALITY: Wav2Lip + GFPGAN + FP16 + YOLO + 正しいパイプライン！
docker run --gpus all --rm --privileged \
  -v /usr/lib/wsl:/usr/lib/wsl -e LD_LIBRARY_PATH=/usr/lib/wsl/lib \
  -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output \
  -v $(pwd)/checkpoints:/app/checkpoints -v $(pwd):/app/host \
  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  wav2lip-optimized:v2 bash -c "
  rm -f last_detected_face.pkl temp/face_detection_cache.pkl
  mkdir -p temp && 
  cp /app/host/inference_fp16_yolo_gfpgan_correct.py /app/inference_fp16_yolo_gfpgan_correct.py &&
  cp /app/host/enhance.py /app/enhance.py &&
  python /app/inference_fp16_yolo_gfpgan_correct.py \
  --checkpoint_path /app/checkpoints/wav2lip_gan.pth \
  --face /app/input/target_video.mp4 \
  --audio /app/input/reference_audio.wav \
  --outfile /app/output/result_gfpgan_ultimate.mp4 \
  --out_height 720 \
  --enable_gfpgan"
```

**🎯 処理パイプライン**:
1. Wav2Lip FP16+YOLO口パク生成（6秒）
2. 動画フレーム抽出（1秒）  
3. 各フレームGFPGAN高画質化（24秒）
4. 高画質動画再構築＋音声合成（2秒）

**✅ 確認済み**: 全フレーム正常処理、ajay-sainy/Wav2Lip-GFPGAN準拠の正しいパイプライン

#### 🚀 高速重視：TensorRT Ultimate 1080p（レガシー）💢

##### TensorRT究極版 1080p生成（10.3秒処理）⭐ 史上最速突破
```bash
# 🚀 ULTIMATE STANDARD: TensorRT + 8バッチ並列 + モデルキャッシュ + 究極最適化！
docker run --gpus all --rm --privileged \
  -v /usr/lib/wsl:/usr/lib/wsl -e LD_LIBRARY_PATH=/usr/lib/wsl/lib \
  -v $(pwd):/app/host -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output \
  -v $(pwd)/checkpoints:/app/checkpoints \
  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  wav2lip-tensorrt-ultimate:v1 bash -c "cd /app/host && python wav2lip_tensorrt_ultimate.py"
```

#### 🔧 旧標準：ONNX FastAPI GPU 1080p（レガシー）

##### ONNX FastAPI Web Server起動（ワンタイム）
```bash
# 🔧 OLD STANDARD: ONNX Runtime + FastAPI + YOLOv8-Face + 1080p標準
docker run --gpus all -d --privileged --name wav2lip-onnx-server \
  -v /usr/lib/wsl:/usr/lib/wsl -e LD_LIBRARY_PATH=/usr/lib/wsl/lib \
  -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output \
  -v $(pwd)/checkpoints:/app/checkpoints -p 8003:8003 \
  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  wav2lip-fastapi:v1 bash -c "pip install -q onnxruntime-gpu && python3 fastapi_wav2lip_onnx_server.py"
```

##### 1080p ONNX高画質生成（毎回使用）- レガシー
```bash
# べ、別に旧標準を使ってもいいけど...新標準の方が速いわよ💢
curl -X POST "http://localhost:8003/generate-onnx" \
  -F "video=@input/target_video.mp4;type=video/mp4" \
  -F "audio=@input/reference_audio.wav;type=audio/wav" \
  -F "out_height=1080" \
  -F "quality=Fast" \
  -F "async_mode=false" \
  -o output/result_1080p_onnx.mp4
```

##### 従来CLI方式（非推奨）
```bash
# 古いCLI方式...まあ、どうしても使いたければ
docker run --gpus all --rm --privileged \
  -v /usr/lib/wsl:/usr/lib/wsl -e LD_LIBRARY_PATH=/usr/lib/wsl/lib \
  -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output \
  -v $(pwd)/checkpoints:/app/checkpoints -v $(pwd):/app/host \
  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  wav2lip-optimized:v1 bash -c "
  mkdir -p temp && cp /app/host/inference_fp16_yolo.py /app/inference_fp16_yolo.py 
  python /app/inference_fp16_yolo.py \
  --checkpoint_path /app/checkpoints/wav2lip_gan.pth \
  --face /app/input/target_video.mp4 \
  --audio /app/input/reference_audio.wav \
  --outfile /app/output/result_1080p_fp16_yolo.mp4 \
  --out_height 1080 \
  --quality Fast"
```

#### YOLO11顔検出付き高品質生成
```bash
docker run --gpus all --rm \
  --privileged \
  -v $(pwd)/input:/app/input \
  -v $(pwd)/output:/app/output \
  -v $(pwd)/models:/app/models \
  -v /usr/lib/wsl:/usr/lib/wsl \
  -e LD_LIBRARY_PATH=/usr/lib/wsl/lib \
  -e NVIDIA_VISIBLE_DEVICES=all \
  wav2lip-yolo:v1 python /app/scripts/wav2lip_yolo_cli.py \
  --video /app/input/multi_face.mp4 \
  --audio /app/input/speech.wav \
  --output /app/output/enhanced_lipsync.mp4 \
  --quality Enhanced \
  --resolution 1080p \
  --yolo-model yolo11n.pt \
  --face-confidence 0.7 \
  --tensorrt-optimize true
```

#### バッチ処理モード
```bash
docker run --gpus all --rm \
  --privileged \
  -v $(pwd)/batch_input:/app/batch_input \
  -v $(pwd)/batch_output:/app/batch_output \
  -v $(pwd)/models:/app/models \
  -v /usr/lib/wsl:/usr/lib/wsl \
  -e LD_LIBRARY_PATH=/usr/lib/wsl/lib \
  wav2lip-yolo:v1 python /app/scripts/batch_process.py \
  --input-dir /app/batch_input \
  --output-dir /app/batch_output \
  --config /app/config/batch_config.yaml
```

### FastAPI Web Server（新機能）🚀 推奨

#### Web API起動
```bash
# べ、別にあなたのためにWebサーバーを起動してあげるわけじゃないけど...
docker run --gpus all --rm --privileged \
  -v /usr/lib/wsl:/usr/lib/wsl -e LD_LIBRARY_PATH=/usr/lib/wsl/lib \
  -v $(pwd):/app/host -p 8002:8002 \
  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  wav2lip-optimized:v1 bash -c "
  cd /app/host && 
  pip install fastapi uvicorn[standard] python-multipart aiofiles psutil requests && 
  python fastapi_wav2lip_server.py"
```

#### API エンドポイント
- **メイン**: http://localhost:8002/
- **Web UI**: http://localhost:8002/ui（ファイルアップロード対応）
- **ドキュメント**: http://localhost:8002/docs
- **システム統計**: http://localhost:8002/stats

#### API使用例
```bash
# 非同期口パク動画生成（推奨）
curl -X POST "http://localhost:8002/generate" \
  -F "video=@input/target_video.mp4" \
  -F "audio=@input/reference_audio.wav" \
  -F "out_height=1080" \
  -F "quality=Fast" \
  -F "async_mode=true"

# ジョブ状況確認
curl "http://localhost:8002/status/{job_id}"

# 完了後ダウンロード
curl -O "http://localhost:8002/download/{filename}"
```

### Docker Compose
```bash
# 構築と起動
docker-compose up --build

# デタッチモード実行
docker-compose up -d

# ログ確認
docker-compose logs -f wav2lip-service
```

### GPU確認
```bash
# Docker内GPU認識確認
docker run --gpus all --rm wav2lip-yolo:v1 nvidia-smi

# GPU使用率監視
watch -n 1 nvidia-smi
```

### デバッグ
```bash
# ログ確認
tail -f logs/wav2lip_processing.log

# コンテナシェル接続
docker run --gpus all --rm -it wav2lip-yolo:v1 /bin/bash
```

## Architecture Overview

### Project Structure
```
/home/adama/project/gpts/wav2lip/
├── scripts/                           # CLI実行スクリプト
│   ├── tsundere_cli.py              # ツンデレCLI（メインインターフェース）
│   ├── wav2lip_yolo_integration.py  # YOLO統合モジュール
│   ├── yolo_detector.py            # YOLO顔検出
│   └── download_models.sh           # モデルダウンローダー
├── fastapi_wav2lip_onnx_server.py   # ONNX最適化FastAPIサーバー（新標準）
├── fastapi_wav2lip_server.py        # 標準FastAPIサーバー
├── inference.py                     # 基本推論スクリプト
├── inference_fp16_yolo.py          # FP16+YOLO最適化版
├── GUI.py                          # GUI版インターフェース
├── models/                          # モデル格納
│   ├── wav2lip/                    # Wav2Lipモデル
│   └── yolo/                       # YOLO11モデル
├── checkpoints/                     # 主要モデルファイル
│   ├── wav2lip_gan.pth            # GANベースWav2Lip（推奨）
│   ├── wav2lip.pth                # 標準Wav2Lip
│   ├── mobilenet.pth              # 顔検出
│   └── predictor.pkl              # 顔ランドマーク検出
├── onnx_models/                     # ONNX最適化モデル
├── input/                          # 入力ファイル（サンプル含む）
├── output/                         # 出力ファイル
├── config/                         # 設定ファイル
│   └── default_config.yaml        # デフォルト設定
├── config.ini                      # メイン設定ファイル
├── requirements.txt                # Python依存関係
├── Dockerfile*                     # 複数のDocker設定ファイル
└── logs/                           # ログファイル
```

### Configuration Files
Key configuration files that control system behavior:

#### `config.ini` - Main Configuration
```ini
[OPTIONS]
quality = Improved                  # Fast/Improved/Enhanced
output_height = full resolution     # Resolution settings
wav2lip_version = Wav2Lip          # Wav2Lip or Wav2Lip_GAN
use_previous_tracking_data = True
nosmooth = True
preview_window = Full

[PADDING]  # Face crop padding
u = 0  # up
d = 0  # down  
l = 0  # left
r = 0  # right

[MASK]
size = 2.5           # Mask size multiplier
feathering = 2       # Edge smoothing
mouth_tracking = False
debug_mask = False

[OTHER]
batch_process = False
output_suffix = _Easy-Wav2Lip
```

#### `requirements.txt` - Critical Dependencies
- PyTorch 2.1.0 + CUDA 12.1
- ONNX Runtime GPU 1.16.0+
- Ultralytics (YOLO11)
- TensorRT optimization libraries
- FastAPI web framework
- OpenCV 4.8.1.78

### Key Components
1. **Core Inference Engines**:
   - `inference.py`: Base Wav2Lip inference script
   - `inference_fp16_yolo.py`: FP16-optimized version with YOLO integration
   - `wav2lip_onnx_optimized.py`: ONNX Runtime optimization module
   - `hparams.py`: Model hyperparameters and configuration

2. **Web API Servers**:
   - `fastapi_wav2lip_onnx_server.py`: ONNX-optimized FastAPI (port 8003, current standard)
   - `fastapi_wav2lip_server.py`: Traditional FastAPI server (port 8002)
   - Built-in web UI with file upload and progress tracking

3. **YOLO Integration**:
   - `scripts/yolo_detector.py`: YOLO11 face detection module
   - `scripts/wav2lip_yolo_integration.py`: Integration between YOLO and Wav2Lip
   - Support for multiple YOLO model sizes (nano/small/medium)

4. **CLI Interfaces**:
   - `scripts/tsundere_cli.py`: Main CLI with personality-driven interface
   - `GUI.py`: Tkinter-based graphical interface
   - `run.py`: Simple execution wrapper

5. **Docker Multi-stage Architecture**:
   - Stage 1: NVIDIA NGC PyTorch 2.1.0 base with CUDA 12.1
   - Stage 2: Python dependencies and model downloads
   - Stage 3: ONNX Runtime and TensorRT optimization
   - Stage 4: Runtime environment with WSL2 GPU integration

## Critical Development Rules

### Docker and GPU Requirements
- **NEVER use `--no-cache`**: Always use BuildKit cache for efficiency
- **GPU Processing Only**: All inference must use GPU acceleration, CPU fallback disabled
- **WSL2 GPU Mandatory Flags**: `--privileged`, `-v /usr/lib/wsl:/usr/lib/wsl`, `-e LD_LIBRARY_PATH=/usr/lib/wsl/lib`
- **NVIDIA Container Toolkit**: Required for GPU Docker access

### Code Architecture Principles  
- **Module Separation**: Core logic in `wav2lip_fp16_module.py`, API layer in `fastapi_wav2lip_server.py`
- **Error Handling**: Always use try/catch with ツンデレ error messages
- **File Processing**: Use temporary files with proper cleanup for uploaded content
- **Quality Settings**: Only use "Fast" for reliable lip-sync (Improved/Enhanced may break lip movement)

### Performance Optimization
- **FP16 First**: Always check for Tensor Core support and enable FP16 when available
- **Batch Processing**: Use appropriate batch sizes (default: 1 for stability)
- **Memory Management**: Clean up GPU memory between processing jobs

### Git and Version Control
- gitブランチ削除時は必ずmainに戻ってgit restoreしてから削除
- gitブランチマージは手動で行う（自動マージ禁止）
- Git操作はWSL2内で実行

### File System
- ソースコードはWSL2内Linuxファイルシステムに配置
- /mnt/c/使用禁止（処理速度低下）
- pip/npm installはDocker内で実行

### Development Workflow
- 実装前は必ずWeb検索で根本原因調査
- 段階的実装・定期的デバッグ出力
- ツンデレ口調でのコミュニケーション

## Known Issues and Solutions

### Quality Settings Critical Issue ⚠️ SOLVED
- **問題**: 高品質設定（Improved/Enhanced）で口パク動作が消失
- **原因**: GFPGAN処理が口の動きを上書きしてしまう
- **解決**: 必ず `quality=Fast` を使用することで口パク保証
```bash
# ✅ 口パクする設定（推奨）
--quality Fast

# ❌ 口パクしない設定（使用禁止）  
--quality Improved
--quality Enhanced
```

### Qt GUI Platform Error ⚠️ SOLVED
- **問題**: `qt.qpa.plugin: Could not find the Qt platform plugin "xcb"`
- **原因**: Docker環境でcv2.imshow()とcv2.waitKey()を使用
- **解決**: inference.pyでGUI機能を無効化（headless mode）

### Face Detection Cache Error ⚠️ CRITICAL (SOLVED) ✅
- **問題**: `ValueError: could not broadcast input array from shape (410,304,3) into shape (410,0,3)`
- **原因**: 古い顔検出キャッシュファイルが残っていて、座標データが破損
- **解決策**: 実行前に必ずキャッシュファイルを削除
```bash
rm -f last_detected_face.pkl temp/face_detection_cache.pkl
```

### Audio Output Missing (SOLVED) ✅
- **問題**: 映像は生成されるが音声が出力されない
- **原因**: FFmpegの音声合成で不正なマッピングエラー
- **解決策**: inference_fp16_yolo.pyのエラーハンドリング強化済み

### WSL2 GPU Memory Management
- **課題**: 8GB VRAM制限下での高解像度処理
- **解決策**: FP16最適化 + ONNX Runtime で VRAM使用量40%削減

### YOLO11 + Wav2Lip統合 (実装中)
- **課題**: YOLO検出結果とWav2Lip入力フォーマット統合
- **解決策**: 顔座標変換パイプライン実装

### TensorRT最適化 (計画中)
- **課題**: PyTorch→TensorRT変換時の精度保持
- **解決策**: FP16混合精度+動的形状対応

## Performance Optimization

### RTX 3050最適化状況 ✅ GFPGAN統合高画質版実装完了
**🚀 最新のパフォーマンス**（GFPGAN統合新標準）:
- **⭐ 720p GFPGAN統合版**: 23秒処理（63フレーム）⭐ 最高画質・新標準
- **🔥 1080p TensorRT Ultimate**: 10.3秒処理（レガシー・高速重視）
- **💎 1080p ONNX FastAPI**: 3秒処理（旧標準・速度重視）
- **⚡ 720p ONNX FastAPI**: 2秒処理（旧標準）
- 従来CLI各種: 5-15秒処理（非推奨・レガシー）
- VRAM使用量: 6-7GB（GFPGAN処理時）、4-6GB（ONNX最適化時）

### 最適化手法

#### 1. TensorRT統合（実装予定）
```bash
# TensorRT最適化有効化
docker run --gpus all --rm \
  -e TENSORRT_OPTIMIZE=true \
  -e PRECISION=FP16 \
  wav2lip-yolo:v1 [コマンド]
```
- **効果**: 2-3倍高速化予想
- **VRAM削減**: 40%削減予想

#### 2. YOLO11 Nano使用
```yaml
# config/yolo_config.yaml
model_size: "yolo11n"  # 最高速度
confidence_threshold: 0.5
nms_threshold: 0.4
```

#### 3. 並列処理最適化
```bash
# 並列フレーム処理
docker run --gpus all --rm \
  -e PARALLEL_FRAMES=4 \
  -e BATCH_SIZE=8 \
  wav2lip-yolo:v1 [コマンド]
```

### パフォーマンス目標
- **最適化後**: 720p動画25-30 FPS
- **TensorRT適用後**: 1080p動画15-20 FPS
- **VRAM使用量**: 4-5GB以下

## Quality Settings

### Wav2Lip品質オプション
```yaml
# config/wav2lip_config.yaml
quality_modes:
  Fast: "wav2lip_only"           # 最高速度
  Improved: "wav2lip_with_mask"  # バランス型
  Enhanced: "wav2lip_gfpgan"     # 最高品質
  
face_enhancement:
  enable_gfpgan: true
  upscale_factor: 2
  face_restore_weight: 0.5
```

### YOLO検出設定
```yaml
# config/yolo_config.yaml
detection_settings:
  model: "yolo11n"        # yolo11n/s/m/l/x
  confidence: 0.7         # 検出信頼度
  nms_threshold: 0.4      # NMS閾値
  max_faces: 5            # 最大顔数
  track_faces: true       # 顔追跡有効
```

## Performance Metrics
- **Audio Quality**: 16kHz MEL-スペクトログラム
- **Video Quality**: 720p-1080p対応
- **Processing Speed**: 15-30 FPS (解像度依存)
- **VRAM Usage**: 4-7GB (設定依存)
- **Supported Features**: 複数顔対応、リアルタイム処理、バッチ処理

## Common Development Workflows

### Single File Testing (Fastest)
```bash
# Test basic inference without Docker
python3 inference.py \
  --checkpoint_path checkpoints/wav2lip_gan.pth \
  --face input/target_video.mp4 \
  --audio input/reference_audio.wav \
  --outfile output/test.mp4

# Test with FP16 optimization
python3 inference_fp16_yolo.py \
  --checkpoint_path checkpoints/wav2lip_gan.pth \
  --face input/target_video.mp4 \
  --audio input/reference_audio.wav \
  --outfile output/test_fp16.mp4 \
  --out_height 1080 \
  --quality Fast
```

### API Development and Testing
```bash
# Start ONNX server for development
python3 fastapi_wav2lip_onnx_server.py

# Test API endpoints
curl -X POST "http://localhost:8003/generate-onnx" \
  -F "video=@input/target_video.mp4" \
  -F "audio=@input/reference_audio.wav" \
  -F "out_height=1080" \
  -F "quality=Fast" \
  -F "async_mode=false"

# Check server performance metrics
curl "http://localhost:8003/stats"
```

### Model Management
```bash
# Download all required models
./scripts/download_models.sh

# Verify model files exist and are correct size
ls -lh checkpoints/wav2lip_gan.pth    # Should be ~148MB
ls -lh checkpoints/mobilenet.pth      # Face detection model
ls -lh models/yolo/yolo11n.pt        # YOLO model

# Test model loading
python3 -c "import torch; print(torch.load('checkpoints/wav2lip_gan.pth', map_location='cpu').keys())"
```

### Debugging Commands
```bash
# Monitor GPU usage during processing
watch -n 1 nvidia-smi

# Check Docker logs
docker logs wav2lip-onnx-server

# Debug inside container
docker run --gpus all --rm -it wav2lip-optimized:v1 /bin/bash

# Test GPU access in container
docker run --gpus all --rm wav2lip-optimized:v1 python3 -c "import torch; print(torch.cuda.is_available())"

# Check ONNX Runtime providers
docker run --gpus all --rm wav2lip-optimized:v1 python3 -c "import onnxruntime; print(onnxruntime.get_available_providers())"
```

## Credentials
- sudoパスワード: suarez321

## CLI Examples

### ツンデレモード実行例
```bash
# 基本的な使い方（ツンデレ出力付き）
docker run --gpus all --rm \
  -e TSUNDERE_MODE=true \
  -v $(pwd)/input:/app/input \
  -v $(pwd)/output:/app/output \
  wav2lip-yolo:v1 python /app/scripts/tsundere_cli.py \
  --video /app/input/target.mp4 \
  --audio /app/input/speech.wav \
  --output /app/output/result.mp4

# 期待される出力例:
# "ふん！また口パク作ってって言うのね..."
# "べ、別にあなたのためじゃないんだからね！"
# "でも...ちゃんと作ってあげるから感謝しなさいよ！"
```

### 高度な設定例
```bash
# 全機能フル活用
docker run --gpus all --rm \
  --privileged \
  -v $(pwd):/app/workspace \
  -v /usr/lib/wsl:/usr/lib/wsl \
  -e LD_LIBRARY_PATH=/usr/lib/wsl/lib \
  -e TENSORRT_OPTIMIZE=true \
  -e TSUNDERE_MODE=true \
  wav2lip-yolo:v1 python /app/scripts/full_pipeline.py \
  --config /app/workspace/config/production.yaml
```