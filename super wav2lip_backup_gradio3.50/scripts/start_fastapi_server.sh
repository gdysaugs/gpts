#!/bin/bash
# Super Wav2Lip FastAPI Server 起動スクリプト
# ツンデレClaude作成 ♪(´∀｀)

echo "🎭 Super Wav2Lip FastAPI Server 起動中..."
echo "==============================================="

# ディレクトリ確認
if [ ! -d "/home/adama/gpts/super wav2lip" ]; then
    echo "❌ プロジェクトディレクトリが見つかりません"
    exit 1
fi

cd "/home/adama/gpts/super wav2lip"

# 必要なディレクトリ作成
mkdir -p temp output input/videos input/audio

# Docker Composeが使える場合
if command -v docker-compose &> /dev/null; then
    echo "🚀 Docker Compose でFastAPIサーバーを起動..."
    docker-compose -f docker-fastapi.yml up -d
    
    echo "📊 サーバー状態確認..."
    sleep 10
    docker-compose -f docker-fastapi.yml logs --tail=20
    
    echo ""
    echo "✅ FastAPIサーバーが起動しました！"
    echo "🌐 API URL: http://localhost:8002"
    echo "📖 API Docs: http://localhost:8002/docs"
    echo "💔 ヘルスチェック: http://localhost:8002/health"
    
else
    # 直接Docker run
    echo "🚀 Docker run でFastAPIサーバーを起動..."
    
    # 既存コンテナを停止・削除
    docker stop super-wav2lip-fastapi 2>/dev/null || true
    docker rm super-wav2lip-fastapi 2>/dev/null || true
    
    # FastAPIサーバー起動
    docker run -d \
        --name super-wav2lip-fastapi \
        --gpus all \
        --privileged \
        -p 8002:8002 \
        -v /usr/lib/wsl:/usr/lib/wsl \
        -e LD_LIBRARY_PATH="/usr/lib/wsl/lib:/usr/local/lib/python3.10/dist-packages/nvidia/curand/lib:/usr/local/lib/python3.10/dist-packages/nvidia/cublas/lib:/usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib:/usr/local/lib/python3.10/dist-packages/nvidia/cufft/lib:/usr/local/lib/python3.10/dist-packages/nvidia/cuda_runtime/lib:/usr/local/lib/python3.10/dist-packages/nvidia/cuda_nvrtc/lib" \
        -e PYTHONPATH=/app/src \
        -v "$(pwd)/input:/app/input" \
        -v "$(pwd)/output:/app/output" \
        -v "$(pwd)/models:/app/models" \
        -v "$(pwd)/src:/app/src" \
        -v "$(pwd)/temp:/app/temp" \
        -w /app/src \
        super-wav2lip:v1-gpu-ultimate \
        bash -c "pip install fastapi uvicorn python-multipart && python fastapi_wav2lip_server.py"
    
    echo "⏳ サーバー起動待機中..."
    sleep 15
    
    echo "📊 サーバーログ確認..."
    docker logs super-wav2lip-fastapi --tail=20
    
    echo ""
    echo "✅ FastAPIサーバーが起動しました！"
    echo "🌐 API URL: http://localhost:8002"
    echo "📖 API Docs: http://localhost:8002/docs"
    echo "💔 ヘルスチェック: http://localhost:8002/health"
fi

echo ""
echo "🧪 簡単なテスト方法:"
echo "curl http://localhost:8002/"
echo "curl http://localhost:8002/health"
echo ""
echo "🛑 停止方法:"
echo "docker stop super-wav2lip-fastapi"
echo ""
echo "ふん！ちゃんと動いてるでしょ？感謝しなさいよ！(｀Д´)"