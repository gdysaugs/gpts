#!/usr/bin/env python3
"""
🚀 ツンデレONNX GFPGANテストスクリプト（シンプル版）
べ、別にあなたのためにONNX GFPGAN高速化してあげるわけじゃないけど...💢
"""

import cv2
import numpy as np
import onnxruntime as ort
import torch
import os
import time
from enhance import upscale, load_sr

# ONNX Providers設定
providers = [
    ('CUDAExecutionProvider', {
        'device_id': 0,
        'arena_extend_strategy': 'kNextPowerOfTwo',
        'gpu_mem_limit': 6 * 1024 * 1024 * 1024,  # 6GB制限
    }),
    'CPUExecutionProvider'
]

def preprocess_frame_gfpgan(frame):
    """GFPGAN ONNX用前処理"""
    # (512, 512)にリサイズ
    frame_resized = cv2.resize(frame, (512, 512))
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    frame_normalized = frame_rgb.astype(np.float32) / 255.0
    frame_tensor = np.transpose(frame_normalized, (2, 0, 1))  # HWC → CHW
    return np.expand_dims(frame_tensor, axis=0)  # バッチ次元追加

def postprocess_gfpgan(output, original_shape):
    """GFPGAN出力後処理"""
    # 最初の出力（メイン結果）を使用
    if isinstance(output, list) and len(output) > 0:
        main_output = output[0]
    else:
        main_output = output
    
    # CHW → HWC
    output_frame = np.transpose(main_output[0], (1, 2, 0))
    # 正規化解除
    output_frame = (output_frame * 255.0).clip(0, 255).astype(np.uint8)
    # RGB → BGR
    output_frame = cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR)
    # 元のサイズにリサイズ
    output_frame = cv2.resize(output_frame, (original_shape[1], original_shape[0]))
    return output_frame

def test_onnx_gfpgan():
    """ONNX GFPGAN性能テスト"""
    print("🎭 ツンデレONNX GFPGANテスト開始💢")
    
    # テスト画像読み込み
    test_image_path = "input/target_video.mp4"
    if not os.path.exists(test_image_path):
        print("❌ テスト動画が見つからない")
        return
    
    # 動画から1フレーム抽出
    cap = cv2.VideoCapture(test_image_path)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("❌ フレーム読み込みエラー")
        return
    
    print(f"フレームサイズ: {frame.shape}")
    
    # 利用可能なONNXモデルをテスト
    onnx_models = [
        "onnx_models/gfpgan_512x512_type_fixed.onnx",
        "onnx_models/gfpgan_512x512_working.onnx",
        "onnx_models/gfpgan_512x512_opset11.onnx"
    ]
    
    for model_path in onnx_models:
        if not os.path.exists(model_path):
            continue
            
        print(f"\n🚀 テスト中: {model_path}")
        
        try:
            # ONNX Runtime セッション初期化
            session = ort.InferenceSession(model_path, providers=providers)
            print(f"✅ モデル読み込み成功")
            
            # 入力名確認
            input_name = session.get_inputs()[0].name
            print(f"入力名: {input_name}")
            
            # 前処理
            input_tensor = preprocess_frame_gfpgan(frame)
            print(f"入力テンソル形状: {input_tensor.shape}")
            
            # 推論実行（時間測定）
            start_time = time.time()
            outputs = session.run(None, {input_name: input_tensor})
            inference_time = time.time() - start_time
            
            print(f"✅ 推論成功！ 処理時間: {inference_time:.3f}秒")
            print(f"出力数: {len(outputs)}")
            for i, output in enumerate(outputs):
                print(f"出力{i}形状: {output.shape}")
            
            # 後処理
            enhanced_frame = postprocess_gfpgan(outputs, frame.shape[:2])
            
            # 結果保存
            output_path = f"output/onnx_test_{os.path.basename(model_path).replace('.onnx', '')}.png"
            cv2.imwrite(output_path, enhanced_frame)
            print(f"✅ 結果保存: {output_path}")
            
            print("🚀 ONNX GFPGAN成功よ！💕")
            return True  # 最初に成功したモデルで終了
            
        except Exception as e:
            print(f"❌ エラー: {e}")
            continue
    
    # 全ONNX失敗時はPyTorch版で比較
    print("\n💢 全ONNXモデル失敗...PyTorch版で比較テストよ")
    try:
        run_params = load_sr()
        start_time = time.time()
        enhanced_frame = upscale(frame, run_params)
        pytorch_time = time.time() - start_time
        
        print(f"PyTorch版処理時間: {pytorch_time:.3f}秒")
        cv2.imwrite("output/pytorch_gfpgan_test.png", enhanced_frame)
        print("PyTorch版結果保存: output/pytorch_gfpgan_test.png")
        
    except Exception as e:
        print(f"PyTorch版もエラー: {e}")
    
    return False

if __name__ == "__main__":
    test_onnx_gfpgan()