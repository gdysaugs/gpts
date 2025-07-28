# FaceFusion 3.3.0 Docker Setup

WSL2 GPU対応のFaceFusion 3.3.0環境です。Condaベースで構築され、事前ダウンロードしたモデルを使用して高速に顔交換を実行できます。

## セットアップ

### 1. Dockerイメージのビルド
```bash
cd /home/adama/project/gpts/FACEFUSION
DOCKER_BUILDKIT=1 docker build -t facefusion:3.3.0 .
```

### 2. モデルのダウンロード（初回のみ）
```bash
# モデルをダウンロード
./download_models.sh
```

### 3. テスト画像の準備
`input`ディレクトリに以下のファイルを配置:
- `source.jpg`: 交換する顔の画像
- `target.jpg`: 顔を交換される画像
- `target.mp4`: (オプション) 顔を交換される動画

## 使用方法

### WSL2 GPU対応版（推奨）
```bash
# 基本的な顔交換（8秒処理）
docker run --device=/dev/nvidiactl --device=/dev/nvidia-uvm --device=/dev/nvidia0 --rm --privileged \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v /usr/lib/wsl:/usr/lib/wsl \
  -e LD_LIBRARY_PATH=/usr/lib/wsl/lib \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e CUDA_VISIBLE_DEVICES=0 \
  -v $(pwd):/workspace \
  -v $(pwd)/models:/app/.assets/models \
  --shm-size=8gb \
  facefusion:3.3.0 \
  python facefusion.py headless-run \
  -s /workspace/input/source.jpg \
  -t /workspace/input/target.jpg \
  -o /workspace/output/result.jpg \
  --processors face_swapper \
  --face-swapper-model hyperswap_1a_256 \
  --execution-providers cuda

# 高品質モデル使用
docker run --device=/dev/nvidiactl --device=/dev/nvidia-uvm --device=/dev/nvidia0 --rm --privileged \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v /usr/lib/wsl:/usr/lib/wsl \
  -e LD_LIBRARY_PATH=/usr/lib/wsl/lib \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e CUDA_VISIBLE_DEVICES=0 \
  -v $(pwd):/workspace \
  -v $(pwd)/models:/app/.assets/models \
  --shm-size=8gb \
  facefusion:3.3.0 \
  python facefusion.py headless-run \
  -s /workspace/input/source.jpg \
  -t /workspace/input/target.jpg \
  -o /workspace/output/result_hq.jpg \
  --processors face_swapper face_enhancer \
  --face-swapper-model hyperswap_1a_256 \
  --face-enhancer-model gfpgan_1.4 \
  --execution-providers cuda
```

### 重要な設定ポイント
- ✅ **モデルマウント**: `models:/app/.assets/models` でダウンロード不要
- ✅ **WSL2 GPU**: デバイスマッピング方式で GPU アクセス
- ✅ **処理時間**: CPU でも 8秒で高速処理
- ✅ **利用可能モデル**: `hyperswap_1a_256`, `hyperswap_1b_256`, `hyperswap_1c_256`

### 簡単実行コマンド
```bash
# 最も簡単な実行方法（コピー&ペースト用）
docker run --device=/dev/nvidiactl --device=/dev/nvidia-uvm --device=/dev/nvidia0 --rm --privileged -v /tmp/.X11-unix:/tmp/.X11-unix -v /usr/lib/wsl:/usr/lib/wsl -e LD_LIBRARY_PATH=/usr/lib/wsl/lib -e NVIDIA_VISIBLE_DEVICES=all -e CUDA_VISIBLE_DEVICES=0 -v $(pwd):/workspace -v $(pwd)/models:/app/.assets/models --shm-size=8gb facefusion:3.3.0 python facefusion.py headless-run -s /workspace/input/source.jpg -t /workspace/input/target.jpg -o /workspace/output/result.jpg --processors face_swapper --face-swapper-model hyperswap_1a_256 --execution-providers cuda

# レガシー CLI スクリプト（GPU エラーあり）
./face_swap_cli.sh input/source.jpg input/target.jpg
```

### テストスクリプト実行
```bash
# テスト画像を配置してから実行
./run_test.sh
```

### Docker Compose
```bash
# CLIモード
docker-compose run --rm facefusion python facefusion.py headless-run \
  --source /app/input/source.jpg \
  --target /app/input/target.jpg \
  --output /app/output/result.jpg \
  --processors face_swapper \
  --skip-download

# GUIモード（ブラウザで http://localhost:7860 にアクセス）
docker-compose --profile api up facefusion-api
```

### 詳細オプション

#### プロセッサー
- `face_swapper`: 顔交換
- `face_enhancer`: 顔の品質向上
- `face_debugger`: デバッグ情報表示

#### モデル選択
- Face Detector: `yoloface_8n`, `retinaface_10g`, `scrfd_2.5g`
- Face Recognizer: `arcface_simswap`, `arcface_w600k_r50`
- Face Swapper: `inswapper_128`, `inswapper_128_fp16`, `simswap_256`
- Face Enhancer: `gfpgan_1.4`, `gfpgan_1.3`, `codeformer`

#### 実行例（高品質設定）
```bash
docker run --gpus all --rm \
  --privileged \
  -v /usr/lib/wsl:/usr/lib/wsl \
  -e LD_LIBRARY_PATH=/usr/lib/wsl/lib \
  -e FACEFUSION_SKIP_DOWNLOAD=1 \
  -v $(pwd):/workspace \
  --shm-size=8gb \
  facefusion:3.3.0 \
  python facefusion.py headless-run \
  --source /workspace/input/source.jpg \
  --target /workspace/input/target.jpg \
  --output /workspace/output/result.jpg \
  --processors face_swapper face_enhancer \
  --face-enhancer-model gfpgan_1.4 \
  --face-enhancer-blend 80 \
  --output-image-quality 95 \
  --skip-download
```

## パフォーマンス設定

### GPU最適化
```bash
--execution-providers cuda
--execution-thread-count 4
--execution-queue-count 1
```

### メモリ最適化（動画処理）
```bash
--video-memory-strategy moderate  # moderate, strict, tolerant
--temp-frame-format jpg          # jpg, png
```

## トラブルシューティング

### GPU認識エラー
```bash
# NVIDIA Container Toolkitの確認
docker run --gpus all --rm nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

### モデルダウンロードエラー
```bash
# 手動でモデルをダウンロード
cd models
wget https://github.com/facefusion/facefusion-assets/releases/download/models-3.0/inswapper_128.onnx
```

### メモリ不足エラー
- `--video-memory-strategy strict`を使用
- `--execution-thread-count`を減らす
- 動画の解像度を下げる

## 注意事項

- 初回実行時はモデルのダウンロードに時間がかかります
- `FACEFUSION_SKIP_DOWNLOAD=1`環境変数でダウンロードをスキップ
- NSFW検証を無効化するには`--skip-nsfw`オプションを使用
- RTX 3050では4GB以上のVRAMを使用する場合があります