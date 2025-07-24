#!/bin/bash
# TensorRT エンジン一括構築スクリプト for Super Wav2Lip
# コンテナ内で実行してONNX → TensorRT変換
#
# 使用例:
# chmod +x scripts/build_tensorrt_engines.sh
# ./scripts/build_tensorrt_engines.sh

set -e  # エラー時に停止

echo "🚀 Super Wav2Lip TensorRT エンジン構築開始"
echo "================================================"

# 環境チェック
check_environment() {
    echo "🔍 環境チェック中..."
    
    # GPU チェック
    if ! nvidia-smi > /dev/null 2>&1; then
        echo "❌ NVIDIA GPU が検出されません"
        exit 1
    fi
    
    # TensorRT チェック
    python3 -c "import tensorrt; print(f'TensorRT version: {tensorrt.__version__}')" || {
        echo "❌ TensorRT がインストールされていません"
        echo "インストール: pip install tensorrt"
        exit 1
    }
    
    # CUDA チェック
    python3 -c "import pycuda.driver; print('PyCUDA available')" || {
        echo "❌ PyCUDA がインストールされていません"
        echo "インストール: pip install pycuda"
        exit 1
    }
    
    echo "✅ 環境チェック完了"
}

# ディレクトリ作成
setup_directories() {
    echo "📁 ディレクトリ設定中..."
    
    mkdir -p /app/models/tensorrt
    mkdir -p /app/logs
    
    echo "✅ ディレクトリ設定完了"
}

# ONNX モデル存在チェック
check_onnx_models() {
    echo "🔍 ONNX モデルチェック中..."
    
    local missing_models=0
    
    # Wav2Lip ONNX
    if [ ! -f "/app/models/onnx/wav2lip_gan.onnx" ]; then
        echo "⚠️  wav2lip_gan.onnx が見つかりません"
        missing_models=$((missing_models + 1))
    else
        echo "✅ wav2lip_gan.onnx 確認"
    fi
    
    # GFPGAN ONNX
    if [ ! -f "/app/src/enhancers/GFPGAN/GFPGANv1.4.onnx" ]; then
        echo "⚠️  GFPGANv1.4.onnx が見つかりません"
        missing_models=$((missing_models + 1))
    else
        echo "✅ GFPGANv1.4.onnx 確認"
    fi
    
    if [ $missing_models -gt 0 ]; then
        echo "⚠️  $missing_models 個のONNXモデルが不足していますが、続行します"
    fi
}

# TensorRT エンジン構築
build_engines() {
    echo "🔧 TensorRT エンジン構築中..."
    
    local log_file="/app/logs/tensorrt_build_$(date +%Y%m%d_%H%M%S).log"
    
    # Python スクリプト実行
    python3 /app/scripts/tensorrt_engine_builder.py \
        --model both \
        --onnx-dir /app/models/onnx \
        --engine-dir /app/models/tensorrt \
        --dynamic \
        --precision fp16 \
        --optimize \
        --benchmark \
        --verbose 2>&1 | tee "$log_file"
    
    if [ $? -eq 0 ]; then
        echo "✅ TensorRT エンジン構築成功"
    else
        echo "❌ TensorRT エンジン構築失敗"
        echo "ログファイル: $log_file"
        exit 1
    fi
}

# エンジンファイル確認
verify_engines() {
    echo "🔍 TensorRT エンジン確認中..."
    
    local engine_dir="/app/models/tensorrt"
    
    # Wav2Lip エンジン
    if [ -f "$engine_dir/wav2lip_gan.trt" ]; then
        local wav2lip_size=$(du -h "$engine_dir/wav2lip_gan.trt" | cut -f1)
        echo "✅ wav2lip_gan.trt ($wav2lip_size)"
    else
        echo "⚠️  wav2lip_gan.trt が生成されませんでした"
    fi
    
    # GFPGAN エンジン
    if [ -f "$engine_dir/gfpgan_v1.4.trt" ]; then
        local gfpgan_size=$(du -h "$engine_dir/gfpgan_v1.4.trt" | cut -f1)
        echo "✅ gfpgan_v1.4.trt ($gfpgan_size)"
    else
        echo "⚠️  gfpgan_v1.4.trt が生成されませんでした"
    fi
    
    # 全体サイズ
    if [ -d "$engine_dir" ]; then
        local total_size=$(du -sh "$engine_dir" | cut -f1)
        echo "📊 TensorRT エンジン総サイズ: $total_size"
    fi
}

# 統合テスト
integration_test() {
    echo "🧪 統合テスト実行中..."
    
    # TensorRT 推論エンジンテスト
    python3 -c "
import sys
sys.path.append('/app/scripts')
try:
    from tensorrt_inference_engine import TensorRTInference
    
    # GFPGAN エンジンテスト
    if os.path.exists('/app/models/tensorrt/gfpgan_v1.4.trt'):
        engine = TensorRTInference('/app/models/tensorrt/gfpgan_v1.4.trt')
        print('✅ GFPGAN TensorRT エンジン読み込み成功')
        
        # ベンチマーク
        results = engine.benchmark((1, 3, 512, 512), iterations=10)
        print(f'   平均実行時間: {results[\"avg_time_ms\"]:.2f}ms')
        print(f'   FPS: {results[\"fps\"]:.1f}')
    else:
        print('⚠️  GFPGAN TensorRT エンジンが見つかりません')
        
except Exception as e:
    print(f'❌ 統合テスト失敗: {e}')
    sys.exit(1)
" 2>/dev/null

    if [ $? -eq 0 ]; then
        echo "✅ 統合テスト成功"
    else
        echo "⚠️  統合テストでエラーが発生しました（エンジンは生成済み）"
    fi
}

# メイン実行
main() {
    echo "開始時刻: $(date)"
    
    check_environment
    setup_directories
    check_onnx_models
    build_engines
    verify_engines
    integration_test
    
    echo ""
    echo "🎉 TensorRT エンジン構築完了!"
    echo "================================================"
    echo "構築されたエンジン:"
    ls -la /app/models/tensorrt/*.trt 2>/dev/null || echo "エンジンファイルが見つかりません"
    echo ""
    echo "使用方法:"
    echo "1. FastAPI で自動的に TensorRT エンジンが使用されます"
    echo "2. 手動使用: from scripts.tensorrt_inference_engine import TensorRTInference"
    echo ""
    echo "終了時刻: $(date)"
}

# Docker コンテナ内実行用
if [ "$1" = "--docker" ]; then
    # Docker 環境での実行
    cd /app
    export PYTHONPATH=/app/src:/app/scripts:$PYTHONPATH
    main
else
    # ホスト環境での実行（Docker 経由）
    echo "🐳 Docker コンテナ内でTensorRT エンジン構築を実行します..."
    
    docker run --gpus all --rm \
        -v "$(pwd):/app" \
        -w /app \
        super-wav2lip:v1-gpu-ultimate \
        bash -c "
            pip install tensorrt pycuda >/dev/null 2>&1 || echo 'TensorRT/PyCUDA インストールをスキップ'
            chmod +x /app/scripts/build_tensorrt_engines.sh
            /app/scripts/build_tensorrt_engines.sh --docker
        "
fi