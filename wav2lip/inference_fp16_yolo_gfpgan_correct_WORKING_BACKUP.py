#!/usr/bin/env python3
"""
🎭 ツンデレWav2Lip + GFPGAN（正しい統合版）
べ、別にあなたのために正しい実装を作ってあげるわけじゃないけど...💢

ajay-sainy/Wav2Lip-GFPGAN の正しいパイプラインに従った実装：
1. Wav2Lipで口パク動画生成
2. フレーム抽出
3. 各フレームにGFPGAN適用
4. 動画再構築＋音声合成
"""

print("\rloading torch       ", end="")
import torch
import torch.cuda.amp as amp

print("\rloading numpy       ", end="")
import numpy as np

print("\rloading cv2         ", end="")
import cv2

print("\rloading os          ", end="")
import os

print("\rloading subprocess  ", end="")
import subprocess

print("\rloading argparse    ", end="")
import argparse

print("\rloading tqdm        ", end="")
from tqdm import tqdm

print("\rloading tempfile    ", end="")
import tempfile

print("\rloading glob        ", end="")
import glob

print("\rloading pathlib     ", end="")
from pathlib import Path

print("\rloading enhance     ", end="")
try:
    from enhance import upscale, load_sr
    GFPGAN_AVAILABLE = True
    print("GFPGAN OK!")
except ImportError:
    print("GFPGAN NOT FOUND!")
    GFPGAN_AVAILABLE = False

print("\rimports loaded!     ")

# 元のinference_fp16_yolo.pyをインポート
import sys
sys.path.append('/app/host')

def run_wav2lip(face_video, audio_file, output_path, checkpoint_path, out_height=720):
    """
    Step 1: Wav2Lipで口パク動画を生成
    """
    print("べ、別に急いで口パク動画を作ってあげるわけじゃないけど...💢")
    
    cmd = [
        "python", "/app/inference_fp16_yolo.py",
        "--checkpoint_path", checkpoint_path,
        "--face", face_video,
        "--audio", audio_file,
        "--outfile", output_path,
        "--out_height", str(out_height),
        "--quality", "Fast"  # 口パク保証のためFast必須
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"も、もう！Wav2Lipエラー: {result.stderr}")
        raise Exception(f"Wav2Lip failed: {result.stderr}")
    
    print("やったじゃない！口パク動画生成完了よ✨")
    return output_path

def extract_frames(video_path, frames_dir):
    """
    Step 2: 動画からフレームを抽出
    """
    print("べ、別にフレーム抽出してあげるわけじゃないけど...💕")
    
    os.makedirs(frames_dir, exist_ok=True)
    
    # FFmpegでフレーム抽出
    cmd = [
        "ffmpeg", "-y", "-loglevel", "warning",
        "-i", video_path,
        "-vf", "fps=25",  # 25fpsで抽出
        f"{frames_dir}/frame_%06d.png"
    ]
    
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise Exception("Frame extraction failed")
    
    frame_files = sorted(glob.glob(f"{frames_dir}/frame_*.png"))
    print(f"やったじゃない！{len(frame_files)}フレーム抽出完了よ✨")
    return frame_files

def enhance_frames_with_gfpgan(frame_files, output_dir):
    """
    Step 3: 各フレームにGFPGAN処理を適用
    """
    print("べ、別に顔画質向上してあげるわけじゃないけど...💢")
    
    if not GFPGAN_AVAILABLE:
        print("ふん！GFPGANが使えないから元のフレームをコピーするだけよ...")
        enhanced_dir = output_dir
        os.makedirs(enhanced_dir, exist_ok=True)
        for frame_file in frame_files:
            filename = os.path.basename(frame_file)
            subprocess.run(["cp", frame_file, f"{enhanced_dir}/{filename}"])
        return sorted(glob.glob(f"{enhanced_dir}/frame_*.png"))
    
    try:
        run_params = load_sr()
        print("やったじゃない！GFPGAN準備完了よ✨")
    except Exception as e:
        print(f"も、もう！GFPGANロードエラー: {e}")
        return frame_files
    
    enhanced_dir = output_dir
    os.makedirs(enhanced_dir, exist_ok=True)
    
    enhanced_files = []
    for i, frame_file in enumerate(tqdm(frame_files, desc="顔画質向上", ncols=80)):
        try:
            # フレーム読み込み
            frame = cv2.imread(frame_file)
            if frame is None:
                continue
            
            # GFPGAN処理
            enhanced_frame = upscale(frame, run_params)
            if enhanced_frame is None:
                enhanced_frame = frame
            
            # 保存
            filename = os.path.basename(frame_file)
            output_path = f"{enhanced_dir}/{filename}"
            cv2.imwrite(output_path, enhanced_frame)
            enhanced_files.append(output_path)
            
        except Exception as e:
            print(f"Frame {i} enhancement error: {e}")
            # エラー時は元フレームをコピー
            filename = os.path.basename(frame_file)
            subprocess.run(["cp", frame_file, f"{enhanced_dir}/{filename}"])
            enhanced_files.append(f"{enhanced_dir}/{filename}")
    
    print(f"GFPGAN処理完了！ {len(enhanced_files)} フレーム処理")
    return sorted(enhanced_files)

def reconstruct_video(enhanced_frames, output_video, fps=25):
    """
    Step 4: 処理済みフレームから動画を再構築
    """
    print("べ、別に動画を再構築してあげるわけじゃないけど...💕")
    
    if not enhanced_frames:
        raise Exception("No enhanced frames found")
    
    # フレームからMP4動画を生成
    frames_pattern = f"{os.path.dirname(enhanced_frames[0])}/frame_%06d.png"
    
    cmd = [
        "ffmpeg", "-y", "-loglevel", "warning",
        "-framerate", str(fps),
        "-i", frames_pattern,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",  # 高品質設定
        output_video
    ]
    
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise Exception("Video reconstruction failed")
    
    print("やったじゃない！動画再構築完了よ✨")
    return output_video

def add_audio(video_path, audio_path, output_path):
    """
    Step 5: 音声を合成
    """
    print("べ、別に音声合成してあげるわけじゃないけど...💢")
    
    cmd = [
        "ffmpeg", "-y", "-loglevel", "warning",
        "-i", video_path, "-i", audio_path,
        "-c:v", "copy", "-c:a", "aac",
        "-strict", "experimental",
        output_path
    ]
    
    result = subprocess.run(cmd)
    if result.returncode != 0:
        # フォールバック：音声なしコピー
        subprocess.run(["cp", video_path, output_path])
    
    print("完了よ！感謝しなさいよね💕")
    return output_path

def main():
    parser = argparse.ArgumentParser(description="Wav2Lip + GFPGAN Integration (Correct Pipeline)")
    parser.add_argument("--checkpoint_path", required=True, help="Wav2Lip checkpoint path")
    parser.add_argument("--face", required=True, help="Input video file")
    parser.add_argument("--audio", required=True, help="Input audio file")
    parser.add_argument("--outfile", default="output/result_gfpgan_correct.mp4", help="Output video file")
    parser.add_argument("--out_height", type=int, default=720, help="Output video height")
    parser.add_argument("--enable_gfpgan", action="store_true", default=True, help="Enable GFPGAN enhancement")
    parser.add_argument("--temp_dir", default="/tmp/wav2lip_gfpgan", help="Temporary directory")
    
    args = parser.parse_args()
    
    # 一時ディレクトリ設定
    temp_dir = args.temp_dir
    os.makedirs(temp_dir, exist_ok=True)
    
    wav2lip_output = f"{temp_dir}/wav2lip_result.mp4"
    frames_dir = f"{temp_dir}/frames"
    enhanced_dir = f"{temp_dir}/enhanced"
    final_video = f"{temp_dir}/final_no_audio.mp4"
    
    try:
        print("🎭 ツンデレWav2Lip + GFPGAN統合処理開始💢")
        
        # Step 1: Wav2Lipで口パク動画生成
        run_wav2lip(args.face, args.audio, wav2lip_output, args.checkpoint_path, args.out_height)
        
        # Step 2: フレーム抽出
        frame_files = extract_frames(wav2lip_output, frames_dir)
        
        # Step 3: GFPGAN処理（オプション）
        if args.enable_gfpgan and GFPGAN_AVAILABLE:
            enhanced_files = enhance_frames_with_gfpgan(frame_files, enhanced_dir)
        else:
            print("GFPGAN無効化：元フレームを使用")
            enhanced_files = frame_files
        
        # Step 4: 動画再構築
        reconstruct_video(enhanced_files, final_video)
        
        # Step 5: 音声合成
        add_audio(final_video, args.audio, args.outfile)
        
        print(f"\n✅ 完了よ！出力ファイル: {args.outfile}")
        
    except Exception as e:
        print(f"❌ エラー: {e}")
        return 1
    
    finally:
        # 一時ファイル削除
        try:
            import shutil
            shutil.rmtree(temp_dir)
            print("一時ファイル削除完了")
        except:
            pass
    
    return 0

if __name__ == "__main__":
    exit(main())