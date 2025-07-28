#!/bin/bash
"""
Super Wav2Lip最適化版デプロイメントスクリプト
事前ロード型FastAPIサーバーで初回遅延を解消
"""

set -e

echo "🚀 Super Wav2Lip 最適化版デプロイ開始..."

# 現在のディレクトリ確認
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "📁 Working directory: $SCRIPT_DIR"

# 必要なファイルの存在確認
required_files=(
    "src/fastapi_wav2lip_server_optimized.py"
    "src/wav2lip_inference_core.py"
    "docker-compose.yml"
    "gradio_wav2lip_ui.py"
)

for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        echo "❌ エラー: 必要なファイルが見つかりません: $file"
        exit 1
    fi
done

echo "✅ 必要なファイルを確認しました"

# バックアップ作成
echo "💾 現在の設定をバックアップ中..."
backup_dir="backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$backup_dir"

if [ -f docker-compose.yml ]; then
    cp docker-compose.yml "$backup_dir/"
fi
if [ -f src/fastapi_wav2lip_server.py ]; then
    cp src/fastapi_wav2lip_server.py "$backup_dir/"
fi

echo "✅ バックアップ完了: $backup_dir"

# Docker Composeの更新（最適化版サーバーを使用）
echo "🔧 Docker Compose設定を最適化版に更新中..."

# 最適化版サーバーを使用するように設定
cat > docker-compose-optimized.yml << 'EOF'
# Super Wav2Lip Complete System - Frontend + Backend (OPTIMIZED)
# 最適化版: 事前ロード型FastAPIで初回遅延を解消

version: '3.8'

services:
  # GPT-SoVITS Voice Cloning API
  gpt-sovits-api:
    image: gpt-sovits:v4
    container_name: gpt-sovits-api
    restart: unless-stopped
    runtime: runc
    
    privileged: true
    
    environment:
      - LD_LIBRARY_PATH=/usr/lib/wsl/lib
      - PYTHONUNBUFFERED=1
    
    volumes:
      - /usr/lib/wsl:/usr/lib/wsl
      - /home/adama/project/gpts/Gptsovits/input:/app/input
      - /home/adama/project/gpts/Gptsovits/output:/app/output
      - /home/adama/project/gpts/Gptsovits/scripts:/app/scripts
      - /home/adama/project/gpts/Gptsovits/models/v4/GPT-SoVITS/gpt-sovits-ja-h:/app/GPT_SoVITS/pretrained_models/gpt-sovits-ja-h
    
    ports:
      - "8000:8000"
    
    networks:
      - wav2lip-network
    
    command: >
      bash -c "
      pip install fastapi uvicorn python-multipart &&
      python /app/scripts/fastapi_voice_server.py
      "
    
    healthcheck:
      test: ["CMD", "python", "-c", "import requests; requests.get('http://localhost:8000/').raise_for_status()"]
      interval: 10s
      timeout: 5s
      retries: 5
      start_period: 60s

  # FastAPI Backend - Super Wav2Lip Processing Engine (OPTIMIZED)
  super-wav2lip-optimized:
    image: super-wav2lip:v1-gpu-ultimate
    container_name: super-wav2lip-optimized
    restart: unless-stopped
    runtime: runc
    
    privileged: true
    
    environment:
      - LD_LIBRARY_PATH=/usr/lib/wsl/lib:/usr/local/lib/python3.10/dist-packages/nvidia/curand/lib:/usr/local/lib/python3.10/dist-packages/nvidia/cublas/lib:/usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib:/usr/local/lib/python3.10/dist-packages/nvidia/cufft/lib:/usr/local/lib/python3.10/dist-packages/nvidia/cuda_runtime/lib:/usr/local/lib/python3.10/dist-packages/nvidia/cuda_nvrtc/lib
      - PYTHONPATH=/app/src
      - PYTHONUNBUFFERED=1
      - QT_QPA_PLATFORM=offscreen
      - OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS=0
      - MPLBACKEND=Agg
    
    volumes:
      - /usr/lib/wsl:/usr/lib/wsl
      - ./input:/app/input
      - ./output:/app/output
      - ./models:/app/models
      - ./src:/app/src
      - ./temp:/app/temp
      - ./gradio_wav2lip_ui.py:/app/gradio_wav2lip_ui.py
      - ./manifest.json:/app/manifest.json
      - ./wav2lip_gan.onnx:/app/original_source/checkpoints/wav2lip_gan.onnx
      - ./wav2lip.onnx:/app/original_source/checkpoints/wav2lip.onnx
    
    working_dir: /app/src
    
    ports:
      - "8002:8002"
    
    networks:
      - wav2lip-network
    
    command: >
      bash -c "
      pip install fastapi uvicorn python-multipart &&
      echo '🚀 Starting OPTIMIZED Wav2Lip Server with Model Preloading...' &&
      python /app/src/fastapi_wav2lip_server_optimized.py
      "
    
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8002/health"]
      interval: 15s
      timeout: 10s
      retries: 5
      start_period: 120s  # 最適化版は初期化に少し時間がかかる
    
    # リソース制限
    mem_limit: 8g
    memswap_limit: 8g
    shm_size: 2g

  # Gradio Frontend - Web UI
  gradio-frontend:
    image: super-wav2lip:v1-gpu-ultimate
    container_name: super-wav2lip-ui
    restart: unless-stopped
    
    environment:
      - FASTAPI_URL=http://super-wav2lip-optimized:8002
      - SOVITS_API=http://gpt-sovits-api:8000
      - WAV2LIP_API=http://super-wav2lip-optimized:8002
      - PYTHONUNBUFFERED=1
    
    volumes:
      - ./gradio_wav2lip_ui.py:/app/gradio_wav2lip_ui.py
      - ./manifest.json:/app/manifest.json
      - ./temp:/app/temp
      - ./input:/app/input
      - ./output:/app/output
    
    ports:
      - "7860:7860"
    
    networks:
      - wav2lip-network
    
    depends_on:
      gpt-sovits-api:
        condition: service_healthy
      super-wav2lip-optimized:
        condition: service_healthy
    
    command: >
      bash -c "
      pip install gradio requests opencv-python Pillow numpy &&
      python /app/gradio_wav2lip_ui.py
      "
    
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:7860/"]
      interval: 15s
      timeout: 5s
      retries: 5
      start_period: 90s

networks:
  wav2lip-network:
    driver: bridge

volumes:
  wav2lip-models:
  wav2lip-output:
  wav2lip-temp:
EOF

echo "✅ 最適化版Docker Compose設定を作成しました"

# 現在実行中のコンテナを停止
echo "🛑 既存のコンテナを停止中..."
docker-compose down 2>/dev/null || true

# 最適化版をデプロイ
echo "🚀 最適化版システムを起動中..."
docker-compose -f docker-compose-optimized.yml up -d

# 起動状況を確認
echo "⏳ システム起動を待機中..."
sleep 10

echo "📊 コンテナ状況確認:"
docker-compose -f docker-compose-optimized.yml ps

echo ""
echo "🔍 ヘルスチェック実行中..."

# GPT-SoVITSのヘルスチェック
echo "  🎤 GPT-SoVITS API (port 8000)..."
for i in {1..6}; do
    if curl -s http://localhost:8000/ > /dev/null 2>&1; then
        echo "  ✅ GPT-SoVITS API is healthy"
        break
    else
        echo "  ⏳ Waiting for GPT-SoVITS API... ($i/6)"
        sleep 10
    fi
done

# Wav2Lip最適化版のヘルスチェック
echo "  🎬 Wav2Lip Optimized API (port 8002)..."
for i in {1..12}; do
    if curl -s http://localhost:8002/health > /dev/null 2>&1; then
        echo "  ✅ Wav2Lip Optimized API is healthy"
        break
    else
        echo "  ⏳ Waiting for Wav2Lip Optimized API... ($i/12)"
        sleep 10
    fi
done

# Gradio UIのヘルスチェック
echo "  🌐 Gradio UI (port 7860)..."
for i in {1..6}; do
    if curl -s http://localhost:7860/ > /dev/null 2>&1; then
        echo "  ✅ Gradio UI is healthy"
        break
    else
        echo "  ⏳ Waiting for Gradio UI... ($i/6)"
        sleep 10
    fi
done

echo ""
echo "🎉 === Super Wav2Lip 最適化版デプロイ完了! ==="
echo ""
echo "📍 アクセス情報:"
echo "  🌐 メインUI:          http://localhost:7860"
echo "  🎤 GPT-SoVITS API:    http://localhost:8000"
echo "  🎬 Wav2Lip API:       http://localhost:8002"
echo ""
echo "📊 最適化内容:"
echo "  ✅ 事前ロード型モデルシステム"
echo "  ✅ 初回リクエストから高速レスポンス"
echo "  ✅ subprocessを排除したin-memory処理"
echo "  ✅ GPU排他制御による安定動作"
echo "  ✅ 段階的ウォームアップ推論"
echo ""
echo "⚡ 期待される性能向上:"
echo "  🚀 初回生成時間: 30秒 → 10秒（-66%短縮）"
echo "  🚀 2回目以降: 即座にレスポンス開始"
echo "  📈 メモリ効率: 常駐モデルによる安定性向上"
echo ""
echo "📝 ログ確認コマンド:"
echo "  docker-compose -f docker-compose-optimized.yml logs -f"
echo ""
echo "🔄 元の設定に戻すには:"
echo "  docker-compose -f docker-compose-optimized.yml down"
echo "  docker-compose up -d"
echo ""

# 使用方法を表示
cat << 'EOF'
🎯 使用方法:
1. ブラウザで http://localhost:7860 にアクセス
2. テキスト入力、動画ファイル、参照音声をアップロード
3. 「🚀 口パク動画生成開始」をクリック
4. 初回は10秒程度、2回目以降はさらに高速で完了！

⚠️  注意: 初回アクセス時にモデルの最終初期化が行われるため、
   最初のリクエストのみ通常より1-2秒多くかかる場合があります。
EOF