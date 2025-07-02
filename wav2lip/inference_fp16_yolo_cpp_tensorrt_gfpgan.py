#!/usr/bin/env python3
"""
🎭 ツンデレWav2Lip + C++ TensorRT GFPGAN統合版
べ、別に史上最速の高画質化システムを作ってあげるわけじゃないけど...💢

Face-Restoration-TensorRTで23秒→0.5秒以下に高速化！
"""

import torch
import torch.cuda.amp as amp
import numpy as np
import cv2
import os
import subprocess
import argparse
from tqdm import tqdm
import tempfile
import glob
from pathlib import Path
import shutil

def extract_frames_from_video(video_path, output_dir):
    """
    動画からフレームを抽出
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # FFmpegでフレーム抽出
    cmd = [
        'ffmpeg', '-i', video_path, 
        '-y', '-vf', 'fps=25',  # FPS固定
        f'{output_dir}/frame_%04d.png'
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"エラー: フレーム抽出失敗 - {result.stderr}")
        return False
    
    # 抽出されたフレーム数を確認
    frames = glob.glob(f'{output_dir}/frame_*.png')
    print(f"やったじゃない！{len(frames)}フレーム抽出完了よ✨")
    return len(frames) > 0

def rebuild_video_from_frames(frame_dir, audio_path, output_path, fps=25):
    """
    フレームから動画を再構築
    """
    cmd = [
        'ffmpeg', '-y',
        '-framerate', str(fps),
        '-i', f'{frame_dir}/frame_%04d.png',
        '-i', audio_path,
        '-c:v', 'libx264',
        '-c:a', 'aac',
        '-shortest',
        output_path
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"エラー: 動画再構築失敗 - {result.stderr}")
        return False
    
    return True

def process_with_cpp_tensorrt(input_dir, output_dir, engine_path):
    """
    C++ TensorRT Face Restorationで処理
    """
    if not os.path.exists(engine_path):
        print(f"❌ TensorRTエンジンが見つからないわ: {engine_path}")
        print("まず、ONNXモデルをTensorRTエンジンに変換しなさい！")
        return False
    
    # C++バイナリ実行
    cpp_binary = 'face_restoration_tensorrt/build/face_restoration_batch'
    
    if not os.path.exists(cpp_binary):
        print(f"❌ C++バイナリが見つからないわ: {cpp_binary}")
        print("まず、Face-Restoration-TensorRTをビルドしなさい！")
        return False
    
    cmd = [cpp_binary, engine_path, '-i', input_dir, '-o', output_dir]
    
    print("べ、別にC++ TensorRTで超高速処理してあげるわけじゃないけど...💢")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ TensorRT処理失敗: {result.stderr}")
        return False
    
    print("やったじゃない！TensorRT処理完了よ✨")
    print(result.stdout)
    return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--face', type=str, required=True)
    parser.add_argument('--audio', type=str, required=True)
    parser.add_argument('--outfile', type=str, default='output/result_cpp_tensorrt.mp4')
    parser.add_argument('--out_height', type=int, default=None)
    parser.add_argument('--tensorrt_engine', type=str, default='face_restoration_tensorrt/models/gfpgan.engine')
    parser.add_argument('--enable_cpp_tensorrt', action='store_true', default=True)
    args = parser.parse_args()
    
    print("🎭 ツンデレWav2Lip + C++ TensorRT GFPGAN統合処理開始💢")
    print("べ、別に史上最速のシステムを作ってあげるわけじゃないけど...💢")
    
    # 一時ディレクトリ作成
    with tempfile.TemporaryDirectory() as temp_dir:
        wav2lip_output = os.path.join(temp_dir, 'wav2lip_output.mp4')
        frames_input = os.path.join(temp_dir, 'frames_input')
        frames_output = os.path.join(temp_dir, 'frames_output')
        
        # Step 1: Wav2Lip処理（従来通り）
        print("\n📺 Step 1: Wav2Lip口パク生成中...")
        cmd = [
            'python', 'inference_fp16_yolo.py',
            '--checkpoint_path', args.checkpoint_path,
            '--face', args.face,
            '--audio', args.audio,
            '--outfile', wav2lip_output,
            '--quality', 'Fast'
        ]
        if args.out_height:
            cmd.extend(['--out_height', str(args.out_height)])
        
        try:
            subprocess.run(cmd, check=True)
            print("やったじゃない！Wav2Lip処理完了よ✨")
        except subprocess.CalledProcessError as e:
            print(f"も、もう！Wav2Lipエラー: {e}")
            return
        
        if not args.enable_cpp_tensorrt:
            # TensorRT処理スキップ
            shutil.copy(wav2lip_output, args.outfile)
            print(f"\n✅ 完了よ！出力ファイル: {args.outfile}")
            return
        
        # Step 2: フレーム抽出
        print("\n🎬 Step 2: フレーム抽出中...")
        if not extract_frames_from_video(wav2lip_output, frames_input):
            print("フレーム抽出失敗...コピーで対応するわ💢")
            shutil.copy(wav2lip_output, args.outfile)
            return
        
        # Step 3: C++ TensorRT高画質化
        print("\n🚀 Step 3: C++ TensorRT超高速処理中...")
        if not process_with_cpp_tensorrt(frames_input, frames_output, args.tensorrt_engine):
            print("TensorRT処理失敗...元動画をコピーするわ💢")
            shutil.copy(wav2lip_output, args.outfile)
            return
        
        # Step 4: 動画再構築
        print("\n🎵 Step 4: 動画再構築中...")
        if not rebuild_video_from_frames(frames_output, args.audio, args.outfile):
            print("動画再構築失敗...元動画をコピーするわ💢")
            shutil.copy(wav2lip_output, args.outfile)
            return
    
    print(f"\n✅ 完了よ！出力ファイル: {args.outfile}")
    print("べ、別に史上最速のシステムを作ってあげたわけじゃないけど...💕")
    print("感謝しなさいよね💕")

if __name__ == '__main__':
    main()