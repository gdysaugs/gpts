#!/usr/bin/env python3
"""
TensorRTエンジン直接実行テスト
76MB のwav2lip_gan.trt を直接使用して性能測定
"""

import time
import numpy as np
from pathlib import Path

def test_tensorrt_performance():
    """TensorRTエンジンの性能テスト"""
    
    print("🚀 TensorRT直接実行テスト開始")
    print("=" * 50)
    
    # TensorRTエンジンファイル確認
    engine_path = "/app/models/tensorrt/wav2lip_gan.trt"
    if not Path(engine_path).exists():
        print(f"❌ TensorRTエンジンが見つかりません: {engine_path}")
        return False
        
    print(f"✅ TensorRTエンジン確認: {Path(engine_path).stat().st_size / 1024 / 1024:.1f}MB")
    
    try:
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit
        print("✅ TensorRT/PyCUDA インポート成功")
    except ImportError as e:
        print(f"❌ TensorRT/PyCUDA インポート失敗: {e}")
        return False
    
    # TensorRTエンジン読み込み
    try:
        trt_logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(trt_logger)
        
        with open(engine_path, 'rb') as f:
            engine = runtime.deserialize_cuda_engine(f.read())
            
        context = engine.create_execution_context()
        print("✅ TensorRTエンジン読み込み成功")
        
        # エンジン情報表示
        print(f"   入力数: {engine.num_bindings}")
        for i in range(engine.num_bindings):
            name = engine.get_binding_name(i)
            shape = engine.get_binding_shape(i)
            print(f"   Binding {i}: {name}, shape: {shape}")
            
    except Exception as e:
        print(f"❌ TensorRTエンジン読み込み失敗: {e}")
        return False
    
    # 性能ベンチマーク（ダミーデータ）
    try:
        print("\n🔥 性能ベンチマーク開始...")
        
        # ダミー入力データ生成（wav2lip形状）
        batch_size = 1
        # 一般的なwav2lipの入力形状を推定
        mel_shape = (batch_size, 1, 80, 16)  # メルスペクトログラム
        video_shape = (batch_size, 6, 96, 96)  # ビデオフレーム
        
        dummy_mel = np.random.randn(*mel_shape).astype(np.float32)
        dummy_video = np.random.randn(*video_shape).astype(np.float32)
        
        # GPU メモリ確保
        mel_size = np.prod(mel_shape) * np.dtype(np.float32).itemsize
        video_size = np.prod(video_shape) * np.dtype(np.float32).itemsize
        output_size = batch_size * 3 * 96 * 96 * np.dtype(np.float32).itemsize
        
        d_mel = cuda.mem_alloc(mel_size)
        d_video = cuda.mem_alloc(video_size)
        d_output = cuda.mem_alloc(output_size)
        
        # データ転送
        cuda.memcpy_htod(d_mel, dummy_mel)
        cuda.memcpy_htod(d_video, dummy_video)
        
        # ウォームアップ
        for _ in range(5):
            context.execute_v2([int(d_mel), int(d_video), int(d_output)])
        cuda.Context.synchronize()
        
        # 性能測定
        iterations = 100
        start_time = time.time()
        
        for _ in range(iterations):
            context.execute_v2([int(d_mel), int(d_video), int(d_output)])
        cuda.Context.synchronize()
        
        end_time = time.time()
        
        # 結果計算
        total_time = end_time - start_time
        avg_time_ms = (total_time / iterations) * 1000
        fps = 1000 / avg_time_ms
        
        print(f"✅ TensorRT性能結果:")
        print(f"   平均推論時間: {avg_time_ms:.2f}ms")
        print(f"   推論FPS: {fps:.1f}")
        print(f"   総実行時間: {total_time:.2f}s ({iterations}回)")
        
        # メモリ解放
        cuda.mem_free(d_mel)
        cuda.mem_free(d_video)
        cuda.mem_free(d_output)
        
        return True
        
    except Exception as e:
        print(f"❌ 性能ベンチマーク失敗: {e}")
        return False

if __name__ == "__main__":
    success = test_tensorrt_performance()
    if success:
        print("\n🎉 TensorRT直接実行テスト完了！")
    else:
        print("\n❌ TensorRT直接実行テスト失敗")