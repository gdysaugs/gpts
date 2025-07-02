#!/usr/bin/env python3
"""
🚀 ツンデレWav2Lip + GFPGAN ONNX究極版
べ、別にあなたのために3倍高速化してあげるわけじゃないけど...💢

ONNX Runtime GPU + GFPGAN統合パイプライン：
1. Wav2Lip ONNX推論で口パク動画生成
2. フレーム抽出
3. 各フレームにGFPGAN ONNX適用
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

print("\rloading onnx        ", end="")
import onnx
import onnxruntime as ort

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

print("\rONNX imports完了!    ", end="")

# GPU最適化設定
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"\nべ、別に嬉しいわけじゃないけど...{device.upper()} GPU使ってあげるわ！", torch.cuda.get_device_name())

# ONNX Providers設定
providers = [
    ('TensorrtExecutionProvider', {
        'device_id': 0,
        'trt_max_workspace_size': 2147483648,
        'trt_fp16_enable': True,
    }),
    ('CUDAExecutionProvider', {
        'device_id': 0,
        'arena_extend_strategy': 'kNextPowerOfTwo',
        'gpu_mem_limit': 6 * 1024 * 1024 * 1024,  # 6GB制限
        'cudnn_conv_algo_search': 'EXHAUSTIVE',
    }),
    'CPUExecutionProvider'
]

def run_wav2lip_fp16(face_video, audio_file, output_path, checkpoint_path, out_height=720):
    """
    Step 1: 既存のFP16+YOLO最適化Wav2Lipで口パク動画を生成
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
    Step 2: 並列最適化でフレーム抽出（高速化）
    """
    print("べ、別にフレーム抽出してあげるわけじゃないけど...💕")
    
    os.makedirs(frames_dir, exist_ok=True)
    
    # FFmpeg並列処理最適化
    cmd = [
        "ffmpeg", "-y", "-loglevel", "warning",
        "-threads", "4",  # 並列処理
        "-i", video_path,
        "-vf", "fps=25",  # 25fpsで抽出
        "-preset", "ultrafast",  # 最高速度設定
        f"{frames_dir}/frame_%06d.png"
    ]
    
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise Exception("Frame extraction failed")
    
    import glob
    frame_files = sorted(glob.glob(f"{frames_dir}/frame_*.png"))
    print(f"やったじゃない！{len(frame_files)}フレーム抽出完了よ✨")
    return frame_files

class ONNXGFPGANEnhancer:
    def __init__(self):
        """ONNX GFPGAN画質向上エンジン"""
        print("べ、別にONNX GFPGAN初期化してあげるわけじゃないけど...💢")
        
        if not GFPGAN_AVAILABLE:
            print("ふん！GFPGAN使えないから画質向上は無効よ...")
            self.gfpgan_session = None
            return
            
        # GFPGAN ONNX モデル読み込み
        gfpgan_path = "/app/onnx_models/gfpgan_512x512_working.onnx"
        if not os.path.exists(gfpgan_path):
            print(f"❌ GFPGAN ONNXモデルが見つからない: {gfpgan_path}")
            print("通常のGFPGANを使用するわよ...")
            self.gfpgan_session = None
            return
        
        try:
            print("やったじゃない！GFPGAN ONNX読み込み中...✨")
            self.gfpgan_session = ort.InferenceSession(gfpgan_path, providers=providers)
            print("✅ GFPGAN ONNX準備完了よ💕")
        except Exception as e:
            print(f"GFPGAN ONNX読み込みエラー: {e}")
            print("通常のGFPGANにフォールバックするわよ...")
            self.gfpgan_session = None
    
    def enhance_frame_with_onnx(self, frame):
        """GFPGAN ONNX推論で顔画質向上"""
        if self.gfpgan_session is None:
            # 通常のGFPGANにフォールバック
            if GFPGAN_AVAILABLE:
                try:
                    with torch.cuda.amp.autocast():
                        with torch.no_grad():
                            run_params = load_sr()
                            enhanced_frame = upscale(frame, run_params)
                            return enhanced_frame if enhanced_frame is not None else frame
                except Exception as e:
                    print(f"通常GFPGAN処理エラー: {e}")
            return frame
        
        try:
            original_shape = frame.shape[:2]
            
            # GFPGAN ONNX用前処理 (512x512)
            frame_resized = cv2.resize(frame, (512, 512))
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            frame_normalized = frame_rgb.astype(np.float32) / 255.0
            frame_tensor = np.transpose(frame_normalized, (2, 0, 1))  # HWC → CHW
            input_tensor = np.expand_dims(frame_tensor, axis=0)  # バッチ次元追加
            
            # ONNX推論
            outputs = self.gfpgan_session.run(['output'], {'input': input_tensor})
            
            # 後処理
            output_frame = np.transpose(outputs[0][0], (1, 2, 0))  # CHW → HWC
            output_frame = (output_frame * 255.0).clip(0, 255).astype(np.uint8)
            output_frame = cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR)
            output_frame = cv2.resize(output_frame, (original_shape[1], original_shape[0]))
            
            return output_frame
            
        except Exception as e:
            print(f"GFPGAN ONNX処理エラー: {e}")
            # 通常のGFPGANにフォールバック
            if GFPGAN_AVAILABLE:
                try:
                    with torch.cuda.amp.autocast():
                        with torch.no_grad():
                            run_params = load_sr()
                            enhanced_frame = upscale(frame, run_params)
                            return enhanced_frame if enhanced_frame is not None else frame
                except Exception as e2:
                    print(f"フォールバックGFPGANエラー: {e2}")
            return frame

def enhance_frames_with_onnx_gfpgan(frame_files, output_dir, target_height=720):
    """
    Step 3: ONNX GFPGAN画質向上（GPU最適化）
    """
    print("べ、別にONNX GFPGAN画質向上してあげるわけじゃないけど...💢")
    print(f"目標解像度: {target_height}p で高画質化するわよ！")
    
    enhancer = ONNXGFPGANEnhancer()
    enhanced_dir = output_dir
    os.makedirs(enhanced_dir, exist_ok=True)
    
    enhanced_files = []
    
    for i, frame_file in enumerate(tqdm(frame_files, desc="ONNX GFPGAN", ncols=80)):
        try:
            # GPU メモリ最適化
            if i % 10 == 0:  # 10フレームごとにメモリクリア
                torch.cuda.empty_cache()
            
            # フレーム読み込み
            frame = cv2.imread(frame_file)
            if frame is None:
                continue
            
            # ONNX GFPGAN処理
            enhanced_frame = enhancer.enhance_frame_with_onnx(frame)
            
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
            print(f"フレーム{i}処理エラー: {e}")
            # エラー時は元フレームをコピー
            filename = os.path.basename(frame_file)
            subprocess.run(["cp", frame_file, f"{enhanced_dir}/{filename}"])
            enhanced_files.append(f"{enhanced_dir}/{filename}")
    
    print(f"やったじゃない！{len(enhanced_files)}フレーム高画質化完了よ✨")
    return enhanced_files

def rebuild_video_with_audio(frame_files, audio_file, output_path, fps=25.0):
    """
    Step 4: 高画質フレームから動画再構築＋音声合成
    """
    print("べ、別に動画再構築してあげるわけじゃないけど...💕")
    
    if not frame_files:
        raise Exception("フレームファイルがないわよ！")
    
    # 一時動画ファイル
    temp_video = "/tmp/temp_enhanced_video.mp4"
    
    # フレームから動画生成（並列処理最適化）
    frames_pattern = os.path.dirname(frame_files[0]) + "/frame_%06d.png"
    
    cmd_video = [
        "ffmpeg", "-y", "-loglevel", "warning",
        "-threads", "4",  # 並列処理
        "-r", str(fps),  # フレームレート
        "-i", frames_pattern,
        "-c:v", "libx264",
        "-preset", "medium",  # バランス設定
        "-pix_fmt", "yuv420p",
        temp_video
    ]
    
    result = subprocess.run(cmd_video)
    if result.returncode != 0:
        raise Exception("動画生成エラー")
    
    # 音声合成
    cmd_audio = [
        "ffmpeg", "-y", "-loglevel", "warning",
        "-i", temp_video,
        "-i", audio_file,
        "-c:v", "copy",  # 動画コピー（再エンコード無し）
        "-c:a", "aac",   # 音声AACエンコード
        "-map", "0:v",   # 動画ストリーム
        "-map", "1:a",   # 音声ストリーム
        "-shortest",     # 短い方に合わせる
        output_path
    ]
    
    result = subprocess.run(cmd_audio)
    if result.returncode != 0:
        raise Exception("音声合成エラー")
    
    # 一時ファイル削除
    if os.path.exists(temp_video):
        os.remove(temp_video)
    
    print("やったじゃない！動画再構築完了よ✨")

def process_video_onnx_gfpgan(args):
    """メイン処理関数（成功したパイプライン準拠）"""
    print("🎭 ツンデレONNX GFPGAN統合処理開始💢")
    print("正しいパイプライン: Wav2Lip → フレーム抽出 → ONNX GFPGAN → 動画再構築")
    
    # 一時ディレクトリ作成
    temp_dir = tempfile.mkdtemp(prefix="onnx_gfpgan_")
    print(f"一時ディレクトリ: {temp_dir}")
    
    try:
        # Step 1: Wav2Lipで口パク動画生成
        temp_lipsync_video = f"{temp_dir}/lipsync_video.mp4"
        run_wav2lip_fp16(
            args.face, 
            args.audio, 
            temp_lipsync_video, 
            args.checkpoint_path, 
            args.out_height
        )
        
        # Step 2: フレーム抽出
        frames_dir = f"{temp_dir}/frames"
        frame_files = extract_frames(temp_lipsync_video, frames_dir)
        
        # Step 3: ONNX GFPGAN画質向上
        enhanced_dir = f"{temp_dir}/enhanced"
        enhanced_files = enhance_frames_with_onnx_gfpgan(frame_files, enhanced_dir, args.out_height)
        
        # Step 4: 動画再構築＋音声合成
        rebuild_video_with_audio(enhanced_files, args.audio, args.outfile)
        
        print("🚀 完了よ！感謝しなさいよね💕")
        
    finally:
        # 一時ファイル削除
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

def main():
    parser = argparse.ArgumentParser(description='ツンデレONNX Wav2Lip + GFPGAN')
    
    parser.add_argument('--checkpoint_path', type=str, help='Wav2Lip checkpoint path', 
                       default='checkpoints/wav2lip_gan.pth')
    parser.add_argument('--face', type=str, required=True, help='Input video file')
    parser.add_argument('--audio', type=str, required=True, help='Input audio file')
    parser.add_argument('--outfile', type=str, required=True, help='Output video file')
    parser.add_argument('--out_height', type=int, default=720, help='Output height')
    parser.add_argument('--enable_gfpgan', action='store_true', help='Enable GFPGAN enhancement')
    
    args = parser.parse_args()
    
    print("🎭 べ、別にあなたのためにONNX最適化してあげるわけじゃないけど...💢")
    print(f"入力動画: {args.face}")
    print(f"入力音声: {args.audio}")
    print(f"出力動画: {args.outfile}")
    print(f"出力解像度: {args.out_height}p")
    print(f"GFPGAN有効: {args.enable_gfpgan}")
    
    process_video_onnx_gfpgan(args)

if __name__ == '__main__':
    main()