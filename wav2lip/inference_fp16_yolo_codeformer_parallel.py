#!/usr/bin/env python3
"""
🎭 ツンデレWav2Lip + CodeFormer（超高速並列処理版）
べ、別にあなたのために並列処理で高速化してあげるわけじゃないけど...💢

マルチプロセス並列処理で高速化：
- フレーム抽出：並列化
- CodeFormer処理：バッチ並列処理
- 動画再構築：並列エンコード
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
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing as mp

print("\rloading enhance     ", end="")
# CodeFormerのみロード（GFPGAN競合回避）
try:
    from enhance_codeformer import enhance_with_codeformer, load_codeformer
    CODEFORMER_AVAILABLE = True
    print("CodeFormer OK!")
except ImportError as e:
    print(f"CodeFormer NOT FOUND: {e}")
    CODEFORMER_AVAILABLE = False

print("\rimports loaded!     ")

# GPU最適化設定
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.cuda.empty_cache()
    print("🚀 GPU最適化設定完了！")

# グローバル変数（プロセス間で共有）
CODEFORMER_MODEL = None

def init_worker():
    """ワーカープロセス初期化"""
    global CODEFORMER_MODEL
    if CODEFORMER_AVAILABLE:
        CODEFORMER_MODEL = load_codeformer()
        print(f"ワーカー {mp.current_process().name} 初期化完了")

def process_frame_worker(args):
    """単一フレーム処理（ワーカー用）"""
    frame_path, output_dir, target_height, fidelity_weight = args
    
    try:
        # フレーム読み込み
        frame = cv2.imread(frame_path)
        if frame is None:
            return None
        
        # CodeFormer処理
        if CODEFORMER_MODEL and CODEFORMER_MODEL['type'] == 'codeformer':
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    enhanced_frame = enhance_with_codeformer(frame, CODEFORMER_MODEL, fidelity_weight)
                    if enhanced_frame is None:
                        enhanced_frame = frame
        else:
            enhanced_frame = frame
        
        # リサイズ
        if target_height and target_height > 0:
            current_height = enhanced_frame.shape[0]
            if current_height != target_height:
                scale_factor = target_height / current_height
                new_width = int(enhanced_frame.shape[1] * scale_factor)
                if new_width % 2 != 0:
                    new_width += 1
                enhanced_frame = cv2.resize(enhanced_frame, (new_width, target_height), interpolation=cv2.INTER_LANCZOS4)
        
        # 保存
        filename = os.path.basename(frame_path)
        output_path = f"{output_dir}/{filename}"
        cv2.imwrite(output_path, enhanced_frame)
        
        return output_path
        
    except Exception as e:
        print(f"Frame {frame_path} error: {e}")
        # エラー時は元フレームをコピー
        filename = os.path.basename(frame_path)
        output_path = f"{output_dir}/{filename}"
        subprocess.run(["cp", frame_path, output_path])
        return output_path

def extract_frames_parallel(video_path, frames_dir, num_threads=4):
    """
    Step 2: 超高速並列フレーム抽出
    """
    print("べ、別に超高速でフレーム抽出してあげるわけじゃないけど...💕")
    
    os.makedirs(frames_dir, exist_ok=True)
    
    # 動画情報取得
    probe_cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-count_packets", "-show_entries",
        "stream=nb_read_packets,r_frame_rate",
        "-of", "csv=p=0",
        video_path
    ]
    result = subprocess.run(probe_cmd, capture_output=True, text=True)
    info = result.stdout.strip().split(',')
    total_frames = int(info[0]) if info[0].isdigit() else 1000  # デフォルト値
    
    # セグメント並列抽出
    segment_duration = max(1, total_frames // num_threads // 25)  # 25fps想定
    
    print(f"⚡ {num_threads}スレッドで並列抽出開始！")
    
    cmd = [
        "ffmpeg", "-y", "-loglevel", "warning",
        "-hwaccel", "auto",  # ハードウェアアクセラレーション
        "-threads", str(num_threads),
        "-i", video_path,
        "-vf", "fps=25",
        "-preset", "ultrafast",
        "-q:v", "2",  # 高品質JPEG
        f"{frames_dir}/frame_%06d.png"
    ]
    
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise Exception("Frame extraction failed")
    
    frame_files = sorted(glob.glob(f"{frames_dir}/frame_*.png"))
    print(f"やったじゃない！{len(frame_files)}フレーム超高速抽出完了よ✨")
    return frame_files

def enhance_frames_parallel(frame_files, output_dir, target_height=720, fidelity_weight=0.7, num_workers=None):
    """
    Step 3: マルチプロセス並列CodeFormer処理
    """
    print("べ、別に超並列でCodeFormer処理してあげるわけじゃないけど...💢")
    print(f"⚡ 並列CodeFormer処理開始（fidelity={fidelity_weight}）")
    
    if not CODEFORMER_AVAILABLE:
        print("CodeFormer利用不可：元フレームをコピー")
        os.makedirs(output_dir, exist_ok=True)
        for frame_file in frame_files:
            filename = os.path.basename(frame_file)
            subprocess.run(["cp", frame_file, f"{output_dir}/{filename}"])
        return sorted(glob.glob(f"{output_dir}/frame_*.png"))
    
    os.makedirs(output_dir, exist_ok=True)
    
    # ワーカー数決定（CPU数とGPUメモリを考慮）
    if num_workers is None:
        num_workers = min(mp.cpu_count() // 2, 4)  # 最大4プロセス
    
    print(f"⚡ {num_workers}プロセスで並列処理開始！")
    
    # 処理引数準備
    process_args = [
        (frame_file, output_dir, target_height, fidelity_weight)
        for frame_file in frame_files
    ]
    
    enhanced_files = []
    
    # マルチプロセス並列処理
    with ProcessPoolExecutor(max_workers=num_workers, initializer=init_worker) as executor:
        # 非同期で全タスクを投入
        futures = {executor.submit(process_frame_worker, args): args[0] for args in process_args}
        
        # プログレスバー付きで結果収集
        for future in tqdm(as_completed(futures), total=len(futures), desc="並列CodeFormer", ncols=80):
            result = future.result()
            if result:
                enhanced_files.append(result)
    
    print(f"超高速CodeFormer処理完了！ {len(enhanced_files)}フレーム処理✨")
    return sorted(enhanced_files)

def enhance_frames_batch_gpu(frame_files, output_dir, target_height=720, fidelity_weight=0.7, batch_size=8):
    """
    Step 3 Alternative: GPUバッチ処理版（単一GPU向け）
    """
    print("べ、別にGPUバッチ処理で爆速化してあげるわけじゃないけど...💢")
    print(f"⚡ バッチサイズ{batch_size}でGPU並列処理！")
    
    if not CODEFORMER_AVAILABLE:
        print("CodeFormer利用不可")
        return frame_files
    
    os.makedirs(output_dir, exist_ok=True)
    
    # CodeFormer初期化
    run_params = load_codeformer()
    if run_params['type'] != 'codeformer':
        print("CodeFormer初期化失敗")
        return frame_files
    
    enhanced_files = []
    
    # バッチ処理
    for batch_start in tqdm(range(0, len(frame_files), batch_size), desc="GPUバッチ処理", ncols=80):
        batch_end = min(batch_start + batch_size, len(frame_files))
        batch_files = frame_files[batch_start:batch_end]
        
        # GPUメモリ管理
        if batch_start % (batch_size * 10) == 0:
            torch.cuda.empty_cache()
        
        # バッチ読み込み
        batch_frames = []
        for frame_file in batch_files:
            frame = cv2.imread(frame_file)
            if frame is not None:
                batch_frames.append(frame)
        
        # バッチ処理
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                for i, frame in enumerate(batch_frames):
                    enhanced_frame = enhance_with_codeformer(frame, run_params, fidelity_weight)
                    if enhanced_frame is None:
                        enhanced_frame = frame
                    
                    # リサイズ&保存
                    if target_height and target_height > 0:
                        current_height = enhanced_frame.shape[0]
                        if current_height != target_height:
                            scale_factor = target_height / current_height
                            new_width = int(enhanced_frame.shape[1] * scale_factor)
                            if new_width % 2 != 0:
                                new_width += 1
                            enhanced_frame = cv2.resize(enhanced_frame, (new_width, target_height), interpolation=cv2.INTER_LANCZOS4)
                    
                    filename = os.path.basename(batch_files[i])
                    output_path = f"{output_dir}/{filename}"
                    cv2.imwrite(output_path, enhanced_frame)
                    enhanced_files.append(output_path)
    
    print(f"GPUバッチ処理完了！ {len(enhanced_files)}フレーム処理✨")
    return sorted(enhanced_files)

def reconstruct_video_parallel(enhanced_frames, output_video, fps=25):
    """
    Step 4: 並列動画再構築
    """
    print("べ、別に超高速で動画再構築してあげるわけじゃないけど...💕")
    
    if not enhanced_frames:
        raise Exception("No enhanced frames found")
    
    frames_pattern = f"{os.path.dirname(enhanced_frames[0])}/frame_%06d.png"
    
    # ハードウェアエンコーダ検出
    hw_encoders = []
    
    # NVIDIA GPU
    if subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True).stdout.find("h264_nvenc") != -1:
        hw_encoders.append("h264_nvenc")
    
    # Intel QSV
    if subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True).stdout.find("h264_qsv") != -1:
        hw_encoders.append("h264_qsv")
    
    # エンコーダ選択
    if hw_encoders:
        encoder = hw_encoders[0]
        print(f"⚡ ハードウェアエンコーダ使用: {encoder}")
        preset = "p4" if encoder == "h264_nvenc" else "fast"
    else:
        encoder = "libx264"
        preset = "ultrafast"
        print("ソフトウェアエンコーダ使用")
    
    cmd = [
        "ffmpeg", "-y", "-loglevel", "warning",
        "-threads", "0",  # 自動最適化
        "-framerate", str(fps),
        "-i", frames_pattern,
        "-c:v", encoder,
        "-preset", preset,
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        output_video
    ]
    
    if encoder == "h264_nvenc":
        cmd.extend(["-rc:v", "vbr", "-b:v", "5M"])
    
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise Exception("Video reconstruction failed")
    
    print("やったじゃない！超高速動画再構築完了よ✨")
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
    parser = argparse.ArgumentParser(description="Wav2Lip + CodeFormer Parallel (Ultra Fast)")
    parser.add_argument("--checkpoint_path", required=True, help="Wav2Lip checkpoint path")
    parser.add_argument("--face", required=True, help="Input video file")
    parser.add_argument("--audio", required=True, help="Input audio file")
    parser.add_argument("--outfile", default="output/result_codeformer_parallel.mp4", help="Output video file")
    parser.add_argument("--fidelity_weight", type=float, default=0.7, help="CodeFormer fidelity weight")
    parser.add_argument("--out_height", type=int, default=720, help="Output video height")
    parser.add_argument("--wav2lip_height", type=int, default=720, help="Wav2Lip processing height")
    parser.add_argument("--num_workers", type=int, default=None, help="Number of parallel workers")
    parser.add_argument("--batch_size", type=int, default=8, help="GPU batch size")
    parser.add_argument("--use_gpu_batch", action="store_true", help="Use GPU batch processing instead of multiprocess")
    parser.add_argument("--temp_dir", default="/tmp/wav2lip_parallel", help="Temporary directory")
    
    args = parser.parse_args()
    
    # 一時ディレクトリ
    temp_dir = args.temp_dir
    os.makedirs(temp_dir, exist_ok=True)
    
    wav2lip_output = f"{temp_dir}/wav2lip_result.mp4"
    frames_dir = f"{temp_dir}/frames"
    enhanced_dir = f"{temp_dir}/enhanced"
    final_video = f"{temp_dir}/final_no_audio.mp4"
    
    try:
        print("🎭 ツンデレWav2Lip + CodeFormer超並列処理開始💢")
        
        # Step 1: 基本的なAV合成
        print(f"べ、別に{args.wav2lip_height}pで処理してあげるわけじゃないけど...💢")
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
        
        # Step 2: 並列フレーム抽出
        frame_files = extract_frames_parallel(wav2lip_output, frames_dir)
        
        # Step 3: 並列CodeFormer処理
        if CODEFORMER_AVAILABLE:
            if args.use_gpu_batch:
                print("GPUバッチ処理モード選択")
                enhanced_files = enhance_frames_batch_gpu(
                    frame_files, enhanced_dir, args.out_height, 
                    args.fidelity_weight, args.batch_size
                )
            else:
                print("マルチプロセス並列処理モード選択")
                enhanced_files = enhance_frames_parallel(
                    frame_files, enhanced_dir, args.out_height,
                    args.fidelity_weight, args.num_workers
                )
        else:
            print("CodeFormer利用不可")
            enhanced_files = frame_files
        
        # Step 4: 並列動画再構築
        reconstruct_video_parallel(enhanced_files, final_video)
        
        # Step 5: 音声合成
        add_audio(final_video, args.audio, args.outfile)
        
        print(f"\n✅ 超高速CodeFormer処理完了よ✨")
        print(f"🎬 出力: {args.outfile}")
        print(f"⚡ 処理モード: {'GPUバッチ' if args.use_gpu_batch else 'マルチプロセス'}")
        
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
    print("🎭 ツンデレWav2Lip + CodeFormer 超並列システム")
    print("べ、別にあなたのために爆速化してあげるわけじゃないけど...💢")
    exit(main())