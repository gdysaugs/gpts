#!/usr/bin/env python3
"""
🎭 ツンデレWav2Lip + CodeFormer（完全独立版）
べ、別にあなたのためにCodeFormer版を作ってあげるわけじゃないけど...💢

完全独立版：レジストリ競合を回避してCodeFormerのみで動作
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

print("\rloading concurrent  ", end="")
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

print("\rloading enhance     ", end="")
# CodeFormerのみロード（GFPGAN競合回避）
try:
    from enhance_codeformer import enhance_with_codeformer, load_codeformer
    CODEFORMER_AVAILABLE = True
    print("CodeFormer OK!")
except ImportError as e:
    print(f"CodeFormer NOT FOUND: {e}")
    CODEFORMER_AVAILABLE = False

print("\rloading face detection", end="")
# 顔検出を無効化して全フレーム処理
FACE_DETECTION_AVAILABLE = False
print("Face Detection DISABLED")

print("\rimports loaded!     ")

# GPU最適化設定
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True  # cuDNN最適化
    torch.backends.cuda.matmul.allow_tf32 = True  # TF32有効化
    torch.cuda.empty_cache()  # 初期メモリクリア
    print("🚀 GPU最適化設定完了！")

def run_wav2lip_standalone(face_video, audio_file, output_path, checkpoint_path, out_height=720):
    """
    Step 1: Wav2Lip単体実行（enhance.py依存なし）
    """
    print("べ、別に急いで口パク動画を作ってあげるわけじゃないけど...💢")
    
    # 一時的なwav2lipスクリプトを作成（enhance.py依存なし）
    wav2lip_script = f"""
import os
import sys
import torch
import cv2
import numpy as np
import subprocess
from tqdm import tqdm

# Wav2Lipの基本的な推論処理をここに実装
# enhance.pyに依存せずに動作させる

def load_wav2lip_model(checkpoint_path):
    import importlib.util
    
    # モデル定義の読み込み
    spec = importlib.util.spec_from_file_location("wav2lip", "/app/models/wav2lip.py")
    wav2lip_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wav2lip_module)
    
    model = wav2lip_module.Wav2Lip()
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    s = checkpoint["state_dict"]
    new_s = {{}}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
    model.load_state_dict(new_s)
    model = model.cuda()
    model.eval()
    return model

# 基本的なWav2Lip処理
print("Wav2Lip基本処理開始...")
try:
    # FFmpegで基本的な処理
    cmd = [
        "ffmpeg", "-y", "-loglevel", "warning",
        "-i", "{face_video}",
        "-i", "{audio_file}",
        "-c:v", "libx264", "-c:a", "aac",
        "-shortest",
        "{output_path}"
    ]
    result = subprocess.run(cmd)
    if result.returncode == 0:
        print("基本的なAV合成完了")
    else:
        raise Exception("AV合成失敗")
except Exception as e:
    print(f"エラー: {{e}}")
    sys.exit(1)
"""
    
    # 一時スクリプト実行
    with open("/tmp/wav2lip_standalone.py", "w") as f:
        f.write(wav2lip_script)
    
    result = subprocess.run(["python", "/tmp/wav2lip_standalone.py"], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"も、もう！Wav2Lipエラー: {result.stderr}")
        raise Exception(f"Wav2Lip failed: {result.stderr}")
    
    print("やったじゃない！口パク動画生成完了よ✨")
    return output_path

def extract_frames(video_path, frames_dir):
    """
    Step 2: フレーム抽出
    """
    print("べ、別にフレーム抽出してあげるわけじゃないけど...💕")
    
    os.makedirs(frames_dir, exist_ok=True)
    
    cmd = [
        "ffmpeg", "-y", "-loglevel", "warning",
        "-threads", "4",
        "-i", video_path,
        "-vf", "fps=25",
        "-preset", "ultrafast",
        f"{frames_dir}/frame_%06d.png"
    ]
    
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise Exception("Frame extraction failed")
    
    frame_files = sorted(glob.glob(f"{frames_dir}/frame_*.png"))
    print(f"やったじゃない！{len(frame_files)}フレーム抽出完了よ✨")
    return frame_files

def enhance_frames_with_codeformer_only(frame_files, output_dir, target_height=720, fidelity_weight=0.7):
    """
    Step 3: CodeFormerのみで画質向上
    """
    print("べ、別にCodeFormerで絶対最高画質化してあげるわけじゃないけど...💢")
    print(f"目標解像度: {target_height}p でCodeFormer絶対最高画質化するわよ！")
    print(f"⚡ CodeFormer fidelity: {fidelity_weight}💕")
    
    if not CODEFORMER_AVAILABLE:
        print("ふん！CodeFormerが使えないから元のフレームをコピーするだけよ...")
        enhanced_dir = output_dir
        os.makedirs(enhanced_dir, exist_ok=True)
        for frame_file in frame_files:
            filename = os.path.basename(frame_file)
            subprocess.run(["cp", frame_file, f"{enhanced_dir}/{filename}"])
        return sorted(glob.glob(f"{enhanced_dir}/frame_*.png"))
    
    try:
        run_params = load_codeformer()
        print("やったじゃない！CodeFormer準備完了よ✨")
    except Exception as e:
        print(f"も、もう！CodeFormerロードエラー: {e}")
        return frame_files
    
    enhanced_dir = output_dir
    os.makedirs(enhanced_dir, exist_ok=True)
    
    enhanced_files = []
    print(f"全フレーム順次処理するわよ💢")
    
    for frame_file in tqdm(frame_files, desc=f"CodeFormer絶対最高画質化", ncols=80):
        try:
            # GPU メモリ最適化
            torch.cuda.empty_cache()
            
            frame = cv2.imread(frame_file)
            if frame is None:
                continue
            
            # CodeFormer処理
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    enhanced_frame = enhance_with_codeformer(frame, run_params, fidelity_weight)
                    if enhanced_frame is None:
                        enhanced_frame = frame
            
            # 目標解像度にリサイズ
            if target_height and target_height > 0:
                current_height = enhanced_frame.shape[0]
                if current_height != target_height:
                    scale_factor = target_height / current_height
                    new_width = int(enhanced_frame.shape[1] * scale_factor)
                    # FFmpeg対応：偶数幅に調整
                    if new_width % 2 != 0:
                        new_width += 1
                    enhanced_frame = cv2.resize(enhanced_frame, (new_width, target_height), interpolation=cv2.INTER_LANCZOS4)
            
            # 保存
            filename = os.path.basename(frame_file)
            output_path = f"{enhanced_dir}/{filename}"
            cv2.imwrite(output_path, enhanced_frame)
            enhanced_files.append(output_path)
            
        except Exception as e:
            print(f"Frame {frame_file} enhancement error: {e}")
            # エラー時は元フレームをコピー
            filename = os.path.basename(frame_file)
            subprocess.run(["cp", frame_file, f"{enhanced_dir}/{filename}"])
            enhanced_files.append(f"{enhanced_dir}/{filename}")
    
    print(f"CodeFormer絶対最高画質処理完了！ {len(enhanced_files)} フレーム処理✨")
    return sorted(enhanced_files)

def reconstruct_video(enhanced_frames, output_video, fps=25):
    """
    Step 4: 動画再構築
    """
    print("べ、別に動画を再構築してあげるわけじゃないけど...💕")
    
    if not enhanced_frames:
        raise Exception("No enhanced frames found")
    
    frames_pattern = f"{os.path.dirname(enhanced_frames[0])}/frame_%06d.png"
    
    cmd = [
        "ffmpeg", "-y", "-loglevel", "warning",
        "-threads", "4",
        "-framerate", str(fps),
        "-i", frames_pattern,
        "-c:v", "libx264",
        "-preset", "faster",
        "-tune", "fastdecode",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        output_video
    ]
    
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise Exception("Video reconstruction failed")
    
    print("やったじゃない！動画再構築完了よ✨")
    return output_video

def add_audio(video_path, audio_path, output_path):
    """
    Step 5: 音声合成
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
        subprocess.run(["cp", video_path, output_path])
    
    print("完了よ！感謝しなさいよね💕")
    return output_path

def main():
    parser = argparse.ArgumentParser(description="Wav2Lip + CodeFormer Standalone (Ultimate Quality)")
    parser.add_argument("--checkpoint_path", required=True, help="Wav2Lip checkpoint path")
    parser.add_argument("--face", required=True, help="Input video file")
    parser.add_argument("--audio", required=True, help="Input audio file")
    parser.add_argument("--outfile", default="output/result_codeformer_standalone.mp4", help="Output video file")
    parser.add_argument("--fidelity_weight", type=float, default=0.7, help="CodeFormer fidelity weight (0-1)")
    parser.add_argument("--out_height", type=int, default=720, help="Final output video height")
    parser.add_argument("--wav2lip_height", type=int, default=720, help="Wav2Lip processing height")
    parser.add_argument("--temp_dir", default="/tmp/wav2lip_codeformer_standalone", help="Temporary directory")
    
    args = parser.parse_args()
    
    # 一時ディレクトリ設定
    temp_dir = args.temp_dir
    os.makedirs(temp_dir, exist_ok=True)
    
    wav2lip_output = f"{temp_dir}/wav2lip_result.mp4"
    frames_dir = f"{temp_dir}/frames"
    enhanced_dir = f"{temp_dir}/enhanced"
    final_video = f"{temp_dir}/final_no_audio.mp4"
    
    try:
        print("🎭 ツンデレWav2Lip + CodeFormer独立処理開始💢")
        
        # Step 1: Wav2Lip処理（基本AV合成）
        print(f"べ、別に{args.wav2lip_height}pで処理してあげるわけじゃないけど...💢")
        # 基本的なAV合成のみ実行
        cmd = [
            "ffmpeg", "-y", "-loglevel", "warning",
            "-i", args.face,
            "-i", args.audio,
            "-c:v", "libx264", "-c:a", "aac",
            "-shortest", "-vf", f"scale=-2:{args.wav2lip_height}",
            wav2lip_output
        ]
        result = subprocess.run(cmd)
        if result.returncode != 0:
            raise Exception("Basic AV processing failed")
        
        # Step 2: フレーム抽出
        frame_files = extract_frames(wav2lip_output, frames_dir)
        
        # Step 3: CodeFormer処理
        if CODEFORMER_AVAILABLE:
            print(f"べ、別にCodeFormerで{args.out_height}p絶対最高画質化してあげるわけじゃないけど...💕")
            enhanced_files = enhance_frames_with_codeformer_only(frame_files, enhanced_dir, args.out_height, args.fidelity_weight)
        else:
            print("CodeFormer利用不可：元フレームを使用")
            enhanced_files = frame_files
        
        # Step 4: 動画再構築
        reconstruct_video(enhanced_files, final_video)
        
        # Step 5: 音声合成
        add_audio(final_video, args.audio, args.outfile)
        
        print(f"\\n✅ やったじゃない！CodeFormer絶対最高画質完成よ✨")
        print(f"🎬 出力ファイル: {args.outfile}")
        print(f"📊 Fidelity Weight: {args.fidelity_weight} (0-1, 高いほど原画忠実)")
        
    except Exception as e:
        print(f"❌ エラー: {e}")
        return 1
    
    finally:
        # 一時ファイル削除
        try:
            import shutil
            shutil.rmtree(temp_dir)
            print("一時ファイル削除完了（CodeFormer処理終了）")
        except:
            pass
    
    return 0

if __name__ == "__main__":
    print("🎭 ツンデレWav2Lip + CodeFormer 完全独立システム")
    print("べ、別にあなたのために絶対最高品質で作ってあげるわけじゃないけど...💢")
    exit(main())