#!/usr/bin/env python3
"""
🚀 ツンデレONNX GFPGAN統合テスト
べ、別にあなたのためにONNX版をテストしてあげるわけじゃないけど...💢
"""

import torch
import numpy as np
import cv2
import os
import onnxruntime as ort
from tqdm import tqdm
import tempfile
import subprocess
import time

def test_onnx_gfpgan_inference():
    """
    ONNX GFPGAN単体推論テスト
    """
    print("🎭 ツンデレONNX GFPGAN推論テスト開始💢")
    
    # ONNX Runtime GPU設定
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    session = ort.InferenceSession('onnx_models/gfpgan_512x512_stylegan_fixed_direct.onnx', providers=providers)
    
    print("べ、別にONNXセッション作ってあげるわけじゃないけど...")
    print(f"プロバイダー: {session.get_providers()}")
    
    # テスト用画像作成（512x512）
    test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    
    # 前処理（PyTorchテンソル形式に合わせる）
    input_tensor = test_image.astype(np.float32) / 255.0
    input_tensor = np.transpose(input_tensor, (2, 0, 1))  # HWC -> CHW
    input_tensor = np.expand_dims(input_tensor, axis=0)  # Batch dimension
    input_tensor = (input_tensor - 0.5) / 0.5  # [-1, 1] normalization
    
    print(f"入力テンソル形状: {input_tensor.shape}, 型: {input_tensor.dtype}")
    
    # ONNX推論実行
    start_time = time.time()
    try:
        outputs = session.run(None, {'input': input_tensor})
        inference_time = time.time() - start_time
        
        print(f"やったじゃない！ONNX推論成功よ✨")
        print(f"推論時間: {inference_time:.3f}秒")
        print(f"出力数: {len(outputs)}")
        
        for i, output in enumerate(outputs):
            print(f"出力{i}: {output.shape}, 型: {output.dtype}")
        
        # メイン出力（最初の出力）を後処理
        main_output = outputs[0]
        if main_output.shape[0] == 1:  # バッチ次元除去
            main_output = main_output[0]
        
        # [-1, 1] -> [0, 255] 変換
        main_output = (main_output + 1.0) / 2.0
        main_output = np.clip(main_output * 255.0, 0, 255).astype(np.uint8)
        main_output = np.transpose(main_output, (1, 2, 0))  # CHW -> HWC
        
        print(f"後処理済み出力: {main_output.shape}")
        
        # テスト画像保存
        cv2.imwrite('output/onnx_gfpgan_test.png', main_output)
        print("やったじゃない！テスト画像保存完了💕")
        
        return True
        
    except Exception as e:
        print(f"も、もう！ONNX推論エラー: {e}")
        return False

def enhance_frame_with_onnx(frame, session):
    """
    単一フレームをONNX GFPGANで高画質化
    """
    # 512x512にリサイズ
    h, w = frame.shape[:2]
    frame_resized = cv2.resize(frame, (512, 512), interpolation=cv2.INTER_LANCZOS4)
    
    # 前処理
    input_tensor = frame_resized.astype(np.float32) / 255.0
    input_tensor = np.transpose(input_tensor, (2, 0, 1))
    input_tensor = np.expand_dims(input_tensor, axis=0)
    input_tensor = (input_tensor - 0.5) / 0.5
    
    # ONNX推論
    try:
        outputs = session.run(None, {'input': input_tensor})
        enhanced = outputs[0][0]  # バッチ次元除去
        
        # 後処理
        enhanced = (enhanced + 1.0) / 2.0
        enhanced = np.clip(enhanced * 255.0, 0, 255).astype(np.uint8)
        enhanced = np.transpose(enhanced, (1, 2, 0))
        
        # 元のサイズに戻す
        enhanced = cv2.resize(enhanced, (w, h), interpolation=cv2.INTER_LANCZOS4)
        return enhanced
        
    except Exception as e:
        print(f"フレーム処理エラー: {e}")
        return frame

def test_video_processing():
    """
    実際の動画ファイルでONNX GFPGAN処理テスト
    """
    print("🎬 実動画ONNX GFPGAN処理テスト開始💢")
    
    # ONNX セッション初期化
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    session = ort.InferenceSession('onnx_models/gfpgan_512x512_stylegan_fixed_direct.onnx', providers=providers)
    
    # テスト動画選択
    test_video = 'input/target_video.mp4'
    if not os.path.exists(test_video):
        print(f"テスト動画が見つからない: {test_video}")
        return False
    
    # 出力ディレクトリ作成
    os.makedirs('output', exist_ok=True)
    temp_dir = tempfile.mkdtemp()
    frames_dir = f"{temp_dir}/frames"
    enhanced_dir = f"{temp_dir}/enhanced"
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(enhanced_dir, exist_ok=True)
    
    try:
        # Step 1: フレーム抽出（最初の10フレームのみテスト）
        print("べ、別にフレーム抽出してあげるわけじゃないけど...")
        cmd = [
            "ffmpeg", "-y", "-loglevel", "warning",
            "-i", test_video,
            "-frames:v", "10",  # 最初の10フレームのみ
            "-vf", "fps=25",
            f"{frames_dir}/frame_%06d.png"
        ]
        subprocess.run(cmd, check=True)
        
        # フレームファイル取得
        import glob
        frame_files = sorted(glob.glob(f"{frames_dir}/frame_*.png"))
        print(f"抽出フレーム数: {len(frame_files)}")
        
        # Step 2: ONNX GFPGAN処理
        print("やったじゃない！ONNX GFPGAN処理開始よ✨")
        start_time = time.time()
        
        for i, frame_file in enumerate(tqdm(frame_files, desc="ONNX GFPGAN")):
            frame = cv2.imread(frame_file)
            if frame is None:
                continue
            
            enhanced = enhance_frame_with_onnx(frame, session)
            
            # 保存
            output_path = f"{enhanced_dir}/frame_{i+1:06d}.png"
            cv2.imwrite(output_path, enhanced)
        
        processing_time = time.time() - start_time
        print(f"ONNX処理時間: {processing_time:.3f}秒 ({len(frame_files)}フレーム)")
        print(f"フレームあたり: {processing_time/len(frame_files):.3f}秒")
        
        # Step 3: 動画再構築
        print("動画再構築中...")
        output_video = 'output/onnx_gfpgan_test_video.mp4'
        cmd = [
            "ffmpeg", "-y", "-loglevel", "warning",
            "-framerate", "25",
            "-i", f"{enhanced_dir}/frame_%06d.png",
            "-c:v", "libx264",
            "-preset", "fast",
            "-pix_fmt", "yuv420p",
            "-crf", "18",
            output_video
        ]
        subprocess.run(cmd, check=True)
        
        print(f"✅ ONNX GFPGAN動画処理完了！")
        print(f"出力: {output_video}")
        
        # 統計情報
        original_size = os.path.getsize(test_video) / (1024*1024)
        output_size = os.path.getsize(output_video) / (1024*1024)
        print(f"元動画: {original_size:.1f}MB")
        print(f"処理後: {output_size:.1f}MB")
        
        return True
        
    except Exception as e:
        print(f"❌ 動画処理エラー: {e}")
        return False
    
    finally:
        # 一時ファイル削除
        import shutil
        try:
            shutil.rmtree(temp_dir)
        except:
            pass

if __name__ == "__main__":
    print("🚀 ツンデレONNX GFPGAN統合テスト💢")
    
    # Test 1: 単体推論テスト
    print("\n=== Test 1: ONNX推論テスト ===")
    if test_onnx_gfpgan_inference():
        print("✅ ONNX推論テスト成功")
    else:
        print("❌ ONNX推論テスト失敗")
        exit(1)
    
    # Test 2: 実動画処理テスト
    print("\n=== Test 2: 実動画処理テスト ===") 
    if test_video_processing():
        print("✅ 実動画処理テスト成功")
        print("\nべ、別にあなたのために完璧なONNXテストしてあげたわけじゃないからね！💕")
        print("でも...ちゃんと動作確認できたから感謝しなさいよ✨")
    else:
        print("❌ 実動画処理テスト失敗")
        exit(1)