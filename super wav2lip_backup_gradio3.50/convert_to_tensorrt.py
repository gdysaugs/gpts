#!/usr/bin/env python3
"""
ONNX to TensorRT Engine Converter
ONNXモデルをTensorRTエンジンに変換して高速化
"""

import os
import sys
import time
from pathlib import Path

def convert_with_onnxruntime():
    """ONNXRuntimeのTensorRTプロバイダーでエンジン生成"""
    try:
        import onnxruntime as ort
        import numpy as np
        
        print("🔧 ONNXRuntime TensorRT Provider による変換開始...")
        
        # モデルパス
        onnx_path = "/app/models/onnx/wav2lip_gan.onnx"
        
        if not Path(onnx_path).exists():
            print(f"❌ ONNXモデルが見つかりません: {onnx_path}")
            return False
        
        print(f"📁 入力: {onnx_path}")
        print(f"📁 サイズ: {Path(onnx_path).stat().st_size / 1024 / 1024:.2f} MB")
        
        # TensorRTプロバイダー設定
        providers = [
            ('TensorrtExecutionProvider', {
                'device_id': 0,
                'trt_max_workspace_size': 4 * 1024 * 1024 * 1024,  # 4GB
                'trt_fp16_enable': True,
                'trt_engine_cache_enable': True,
                'trt_engine_cache_path': '/app/models/onnx/trt_cache',
                'trt_int8_enable': False,
                'trt_max_partition_iterations': 1000,
                'trt_min_subgraph_size': 1
            }),
            'CUDAExecutionProvider',
            'CPUExecutionProvider'
        ]
        
        # キャッシュディレクトリ作成
        os.makedirs('/app/models/onnx/trt_cache', exist_ok=True)
        
        # セッション作成（この時点でTensorRTエンジンが生成される）
        print("🔨 TensorRTエンジン生成中...")
        start_time = time.time()
        
        session = ort.InferenceSession(onnx_path, providers=providers)
        
        # 入力情報取得
        input_name = session.get_inputs()[0].name
        input_shape = session.get_inputs()[0].shape
        print(f"📋 入力: {input_name}, shape: {input_shape}")
        
        # ダミー入力でウォームアップ（エンジン最適化）
        print("🔥 ウォームアップ実行中...")
        if input_shape[0] == 'batch_size' or input_shape[0] is None:
            dummy_shape = [1] + list(input_shape[1:])
        else:
            dummy_shape = input_shape
            
        # Noneを適切な値に置換
        for i, dim in enumerate(dummy_shape):
            if dim is None or (isinstance(dim, str)):
                if i == 0:
                    dummy_shape[i] = 1  # バッチサイズ
                elif i == 1:
                    dummy_shape[i] = 80  # 音声特徴量次元（推定）
                else:
                    dummy_shape[i] = 16  # その他の次元
        
        dummy_input = np.random.randn(*dummy_shape).astype(np.float32)
        
        # 複数回実行してエンジン最適化
        for i in range(3):
            _ = session.run(None, {input_name: dummy_input})
            print(f"  ウォームアップ {i+1}/3 完了")
        
        elapsed = time.time() - start_time
        print(f"✅ TensorRTエンジン生成完了: {elapsed:.2f}秒")
        
        # キャッシュファイル確認
        cache_files = list(Path('/app/models/onnx/trt_cache').glob('*'))
        if cache_files:
            total_size = sum(f.stat().st_size for f in cache_files)
            print(f"📁 TensorRTキャッシュ: {len(cache_files)}ファイル, {total_size/1024/1024:.2f}MB")
            for f in cache_files:
                print(f"  - {f.name}: {f.stat().st_size/1024/1024:.2f}MB")
        
        return True
        
    except Exception as e:
        print(f"❌ 変換エラー: {e}")
        return False

def main():
    print("🚀 TensorRT変換スクリプト開始")
    print("=" * 50)
    
    # 環境確認
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        print(f"📋 利用可能プロバイダー: {providers}")
        
        if 'TensorrtExecutionProvider' not in providers:
            print("❌ TensorrtExecutionProvider が利用できません")
            return
            
    except ImportError:
        print("❌ ONNXRuntime がインストールされていません")
        return
    
    # 変換実行
    success = convert_with_onnxruntime()
    
    if success:
        print("\n🎉 TensorRT変換完了！")
        print("📈 次回の推論から高速化されます")
    else:
        print("\n❌ TensorRT変換失敗")

if __name__ == "__main__":
    main()