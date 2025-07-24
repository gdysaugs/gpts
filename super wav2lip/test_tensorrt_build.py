#!/usr/bin/env python3
"""
TensorRT エンジン構築テストスクリプト
コンテナ内でTensorRT変換をテスト実行

使用例:
python test_tensorrt_build.py
"""

import os
import subprocess
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_tensorrt_availability():
    """TensorRT利用可能性チェック"""
    try:
        import tensorrt as trt
        print(f"✅ TensorRT available: {trt.__version__}")
        return True
    except ImportError:
        print("❌ TensorRT not available")
        return False

def test_pycuda_availability():
    """PyCUDA利用可能性チェック"""
    try:
        import pycuda.driver as cuda
        import pycuda.autoinit
        print("✅ PyCUDA available")
        return True
    except ImportError:
        print("❌ PyCUDA not available")
        return False

def run_tensorrt_build():
    """TensorRT エンジン構築実行"""
    print("🚀 TensorRT エンジン構築開始...")
    
    # Dockerコンテナ内で実行
    cmd = [
        "docker", "run", "--gpus", "all", "--rm",
        "-v", f"{os.getcwd()}:/app",
        "-w", "/app",
        "super-wav2lip:v1-gpu-ultimate",
        "bash", "-c", 
        """
        echo "📦 TensorRT/PyCUDA インストール中..."
        pip install tensorrt pycuda --quiet || echo "インストールスキップ"
        
        echo "🔧 TensorRT エンジン構築実行中..."
        python3 scripts/tensorrt_engine_builder.py \
            --model both \
            --dynamic \
            --precision fp16 \
            --benchmark \
            --verbose
        """
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ TensorRT エンジン構築成功")
        print("標準出力:")
        print(result.stdout[-1000:])  # 最後の1000文字
        return True
    except subprocess.CalledProcessError as e:
        print("❌ TensorRT エンジン構築失敗")
        print("エラー出力:")
        print(e.stderr[-1000:])  # 最後の1000文字
        return False

def verify_engines():
    """構築されたエンジンの確認"""
    engine_dir = Path("models/tensorrt")
    
    if not engine_dir.exists():
        print("❌ TensorRT エンジンディレクトリが存在しません")
        return False
        
    engines = list(engine_dir.glob("*.trt"))
    
    if not engines:
        print("❌ TensorRT エンジンファイルが見つかりません")
        return False
        
    print(f"✅ 構築されたエンジン ({len(engines)}個):")
    for engine in engines:
        size_mb = engine.stat().st_size / (1024 * 1024)
        print(f"   {engine.name}: {size_mb:.1f}MB")
        
    return True

def main():
    print("🧪 TensorRT エンジン構築テスト")
    print("=" * 50)
    
    # 前提条件チェック
    print("1. 前提条件チェック")
    if not test_tensorrt_availability():
        print("   TensorRTをインストールしてください: pip install tensorrt")
        
    if not test_pycuda_availability():
        print("   PyCUDAをインストールしてください: pip install pycuda")
        
    # エンジン構築実行
    print("\n2. TensorRT エンジン構築")
    success = run_tensorrt_build()
    
    # 結果確認
    print("\n3. 結果確認")
    if success:
        verify_engines()
    
    print(f"\n{'✅ テスト完了' if success else '❌ テスト失敗'}")

if __name__ == "__main__":
    main()