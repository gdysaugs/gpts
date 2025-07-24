#!/usr/bin/env python3
"""
Dynamic Shape対応TensorRT性能テスト
"""

import time
import numpy as np
import onnxruntime as ort

def test_tensorrt_dynamic():
    """Dynamic TensorRT推論テスト"""
    
    print("🚀 Dynamic TensorRT 性能テスト開始")
    
    # TensorRTプロバイダー設定（動的形状対応）
    providers = [
        ('TensorrtExecutionProvider', {
            'device_id': 0,
            'trt_max_workspace_size': 4 * 1024 * 1024 * 1024,  # 4GB
            'trt_fp16_enable': True,
            'trt_engine_cache_enable': True,
            'trt_engine_cache_path': '/app/models/onnx/trt_dynamic_cache',
            'trt_builder_optimization_level': 5,
            'trt_auxiliary_streams': 2,
            # Dynamic shape profiles
            'trt_profile_min_shapes': 'mel_spectrogram:1x1x80x16|video_frames:1x6x96x96',
            'trt_profile_opt_shapes': 'mel_spectrogram:1x1x80x16|video_frames:1x6x96x96',
            'trt_profile_max_shapes': 'mel_spectrogram:4x1x80x64|video_frames:4x6x384x384'
        }),
        'CUDAExecutionProvider'
    ]
    
    try:
        # セッション作成
        print("📊 TensorRTセッション作成中...")
        session = ort.InferenceSession(
            '/app/models/onnx/wav2lip_gan.onnx',
            providers=providers
        )
        print(f"✅ 使用プロバイダー: {session.get_providers()}")
        
        # テスト入力
        batch_size = 1
        mel_input = np.random.randn(batch_size, 1, 80, 16).astype(np.float32)
        video_input = np.random.randn(batch_size, 6, 96, 96).astype(np.float32)
        
        # ウォームアップ
        print("\n🔥 ウォームアップ中...")
        for _ in range(5):
            _ = session.run(None, {
                'mel_spectrogram': mel_input,
                'video_frames': video_input
            })
        
        # 性能測定
        print("\n⚡ 性能測定開始...")
        iterations = 100
        
        start = time.time()
        for _ in range(iterations):
            output = session.run(None, {
                'mel_spectrogram': mel_input,
                'video_frames': video_input
            })
        end = time.time()
        
        avg_ms = (end - start) / iterations * 1000
        fps = 1000 / avg_ms
        
        print(f"\n📈 結果:")
        print(f"   平均推論時間: {avg_ms:.2f}ms")
        print(f"   推論FPS: {fps:.1f}")
        print(f"   理論スループット: {fps * 96 * 96 / 1000000:.2f} Mpixels/sec")
        
        # 異なる解像度でテスト
        print("\n📊 解像度スケーリングテスト:")
        resolutions = [(96, 96), (192, 192), (256, 256)]
        
        for h, w in resolutions:
            video_test = np.random.randn(1, 6, h, w).astype(np.float32)
            
            start = time.time()
            for _ in range(10):
                try:
                    _ = session.run(None, {
                        'mel_spectrogram': mel_input,
                        'video_frames': video_test
                    })
                except Exception as e:
                    print(f"   {h}x{w}: ❌ エラー - {str(e)[:50]}...")
                    break
            else:
                elapsed = time.time() - start
                avg_ms = elapsed / 10 * 1000
                print(f"   {h}x{w}: {avg_ms:.1f}ms/frame ({1000/avg_ms:.1f} FPS)")
        
    except Exception as e:
        print(f"❌ エラー: {e}")
        
        # CUDAプロバイダーでフォールバック
        print("\n📊 CUDAプロバイダーでの性能:")
        session = ort.InferenceSession(
            '/app/models/onnx/wav2lip_gan.onnx',
            providers=['CUDAExecutionProvider']
        )
        
        start = time.time()
        for _ in range(100):
            _ = session.run(None, {
                'mel_spectrogram': mel_input,
                'video_frames': video_input
            })
        end = time.time()
        
        avg_ms = (end - start) / 100 * 1000
        fps = 1000 / avg_ms
        print(f"   平均推論時間: {avg_ms:.2f}ms")
        print(f"   推論FPS: {fps:.1f}")

if __name__ == "__main__":
    test_tensorrt_dynamic()