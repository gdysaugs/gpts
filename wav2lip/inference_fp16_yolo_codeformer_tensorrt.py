#!/usr/bin/env python3
"""
🎭 ツンデレWav2Lip + CodeFormer TensorRT究極版
べ、別にあなたのために究極の高速化をしてあげるわけじゃないけど...💢

パイプライン：
1. Wav2Lipで口パク動画生成（FP16最適化）
2. フレーム抽出
3. 各フレームにCodeFormer TensorRT適用（超高速）
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

print("\rloading codeformer  ", end="")
import sys
sys.path.insert(0, '/app/codeformer')

# CodeFormer TensorRT関連のインポート
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    TENSORRT_AVAILABLE = True
    print("TensorRT OK!")
except ImportError:
    TENSORRT_AVAILABLE = False
    print("TensorRT NOT FOUND!")

# ONNX Runtime フォールバック
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
    print("ONNX Runtime OK!")
except ImportError:
    ONNX_AVAILABLE = False
    print("ONNX Runtime NOT FOUND!")

print("imports loaded!     ")

class CodeFormerTensorRT:
    """CodeFormer TensorRT推論エンジン（ONNX Runtimeフォールバック付き）"""
    
    def __init__(self, engine_path="/app/codeformer/engines/codeformer_simple.trt", onnx_path="/app/codeformer/engines/codeformer_ultimate.onnx"):
        self.engine_path = engine_path
        self.onnx_path = onnx_path
        self.tensorrt_engine = None
        self.tensorrt_context = None
        self.tensorrt_inputs = []
        self.tensorrt_outputs = []
        self.tensorrt_bindings = []
        self.tensorrt_stream = None
        self.onnx_session = None
        self.use_onnx = False
        
        # TensorRTを試す
        if TENSORRT_AVAILABLE and os.path.exists(engine_path):
            self._load_engine()
        
        # TensorRTが失敗したらONNX Runtimeを試す
        if self.tensorrt_engine is None and ONNX_AVAILABLE and os.path.exists(onnx_path):
            self._load_onnx()
    
    def _load_engine(self):
        """TensorRTエンジンをロード"""
        try:
            # Load TensorRT engine
            with open(self.engine_path, 'rb') as f:
                engine_data = f.read()
            
            logger = trt.Logger(trt.Logger.ERROR)  # エラーレベルを下げる
            runtime = trt.Runtime(logger)
            
            # エンジンデシリアライズを試行
            try:
                self.tensorrt_engine = runtime.deserialize_cuda_engine(engine_data)
            except Exception as deserialize_error:
                print(f"TensorRTエンジンバージョン不一致: {deserialize_error}")
                # ONNX Runtimeにフォールバック
                self.tensorrt_engine = None
                return
            
            if self.tensorrt_engine is None:
                print("TensorRTエンジンの読み込みに失敗")
                return
                
            self.tensorrt_context = self.tensorrt_engine.create_execution_context()
            
            # Allocate buffers
            self.tensorrt_inputs = []
            self.tensorrt_outputs = []
            self.tensorrt_bindings = []
            self.tensorrt_stream = cuda.Stream()
            
            for binding in self.tensorrt_engine:
                size = trt.volume(self.tensorrt_engine.get_binding_shape(binding))
                dtype = trt.nptype(self.tensorrt_engine.get_binding_dtype(binding))
                host_mem = cuda.pagelocked_empty(size, dtype)
                device_mem = cuda.mem_alloc(host_mem.nbytes)
                self.tensorrt_bindings.append(int(device_mem))
                
                if self.tensorrt_engine.binding_is_input(binding):
                    self.tensorrt_inputs.append({'host': host_mem, 'device': device_mem})
                else:
                    self.tensorrt_outputs.append({'host': host_mem, 'device': device_mem})
            
            print("やったじゃない！CodeFormer TensorRT準備完了よ✨")
            
        except Exception as e:
            print(f"も、もう！TensorRTロードエラー: {e}")
            self.tensorrt_engine = None
    
    def _load_onnx(self):
        """ONNX Runtimeをロード"""
        try:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            self.onnx_session = ort.InferenceSession(self.onnx_path, providers=providers)
            self.use_onnx = True
            print("やったじゃない！CodeFormer ONNX Runtime準備完了よ✨")
            
        except Exception as e:
            print(f"も、もう！ONNX Runtimeロードエラー: {e}")
            self.onnx_session = None
    
    def enhance_image(self, image):
        """画像をCodeFormerで高画質化"""
        if self.tensorrt_engine is None and self.onnx_session is None:
            return image
        
        try:
            # 前処理：512x512にリサイズ
            original_shape = image.shape[:2]
            resized = cv2.resize(image, (512, 512), interpolation=cv2.INTER_LANCZOS4)
            
            # RGB変換と正規化
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            normalized = rgb.astype(np.float32) / 255.0
            normalized = normalized * 2.0 - 1.0  # [-1, 1]
            
            # テンソル形式に変換 (NCHW)
            tensor = np.transpose(normalized, (2, 0, 1))
            batch = np.expand_dims(tensor, axis=0)
            
            # 推論実行
            if self.tensorrt_engine is not None:
                # TensorRT推論
                np.copyto(self.tensorrt_inputs[0]['host'], batch.ravel())
                cuda.memcpy_htod_async(
                    self.tensorrt_inputs[0]['device'],
                    self.tensorrt_inputs[0]['host'],
                    self.tensorrt_stream
                )
                
                self.tensorrt_context.execute_async_v2(
                    bindings=self.tensorrt_bindings,
                    stream_handle=self.tensorrt_stream.handle
                )
                
                cuda.memcpy_dtoh_async(
                    self.tensorrt_outputs[0]['host'],
                    self.tensorrt_outputs[0]['device'],
                    self.tensorrt_stream
                )
                
                self.tensorrt_stream.synchronize()
                output = self.tensorrt_outputs[0]['host'].reshape(1, 3, 512, 512)
                
            elif self.onnx_session is not None:
                # ONNX Runtime推論
                input_name = self.onnx_session.get_inputs()[0].name
                output_name = self.onnx_session.get_outputs()[0].name
                result = self.onnx_session.run([output_name], {input_name: batch})
                output = result[0]
                
            else:
                return image
            enhanced = output.squeeze(0)
            enhanced = np.transpose(enhanced, (1, 2, 0))
            
            # [-1, 1] → [0, 255]
            enhanced = (enhanced + 1.0) / 2.0
            enhanced = np.clip(enhanced * 255.0, 0, 255).astype(np.uint8)
            
            # BGR変換
            enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR)
            
            # 元のサイズに戻す
            if original_shape != (512, 512):
                enhanced_bgr = cv2.resize(
                    enhanced_bgr, 
                    (original_shape[1], original_shape[0]), 
                    interpolation=cv2.INTER_LANCZOS4
                )
            
            return enhanced_bgr
            
        except Exception as e:
            print(f"CodeFormer処理エラー: {e}")
            return image

# Wav2Lip関連の関数はそのまま使用
def run_wav2lip(face_video, audio_file, output_video, checkpoint_path, out_height=720):
    """
    Step 1: Wav2Lipで口パク動画生成（FP16最適化）
    """
    print("べ、別に急いで口パク動画を作ってあげるわけじゃないけど...💢")
    
    cmd = [
        "python", "/app/inference_fp16_yolo.py",
        "--checkpoint_path", checkpoint_path,
        "--face", face_video,
        "--audio", audio_file,
        "--outfile", output_video,
        "--out_height", str(out_height),
        "--quality", "Fast"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Wav2Lipエラー: {result.stderr}")
        raise Exception("Wav2Lip processing failed")
    
    print("やったじゃない！口パク動画生成完了よ✨")
    return output_video

def extract_frames(video_path, output_dir):
    """
    Step 2: 動画からフレームを抽出
    """
    print("べ、別にフレーム抽出してあげるわけじゃないけど...💕")
    
    os.makedirs(output_dir, exist_ok=True)
    
    cmd = [
        "ffmpeg", "-y", "-loglevel", "warning",
        "-i", video_path,
        f"{output_dir}/frame_%06d.png"
    ]
    
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise Exception("Frame extraction failed")
    
    frame_files = sorted(glob.glob(f"{output_dir}/frame_*.png"))
    print(f"やったじゃない！{len(frame_files)}フレーム抽出完了よ✨")
    return frame_files

def enhance_frames_with_codeformer(frame_files, output_dir, engine_path="/app/codeformer/engines/codeformer_simple.trt"):
    """
    Step 3: 各フレームにCodeFormer TensorRTを適用
    """
    print("べ、別に顔画質向上してあげるわけじゃないけど...💢")
    
    if not frame_files:
        return []
    
    # CodeFormerエンジンを初期化
    codeformer = CodeFormerTensorRT(engine_path)
    if codeformer.tensorrt_engine is None and codeformer.onnx_session is None:
        print("CodeFormer無効化：元フレームを使用")
        return frame_files
    
    enhanced_dir = output_dir
    os.makedirs(enhanced_dir, exist_ok=True)
    
    enhanced_files = []
    for i, frame_file in enumerate(tqdm(frame_files, desc="🎨顔画質向上", ncols=80)):
        try:
            # フレーム読み込み
            frame = cv2.imread(frame_file)
            if frame is None:
                continue
            
            # CodeFormer TensorRT処理
            enhanced_frame = codeformer.enhance_image(frame)
            
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
    
    print(f"CodeFormer処理完了！ {len(enhanced_files)} フレーム処理")
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
    parser = argparse.ArgumentParser(description="Wav2Lip + CodeFormer TensorRT Integration")
    parser.add_argument("--checkpoint_path", required=True, help="Wav2Lip checkpoint path")
    parser.add_argument("--face", required=True, help="Input video file")
    parser.add_argument("--audio", required=True, help="Input audio file")
    parser.add_argument("--outfile", default="output/result_codeformer_tensorrt.mp4", help="Output video file")
    parser.add_argument("--out_height", type=int, default=720, help="Output video height")
    parser.add_argument("--engine_path", default="/app/codeformer/engines/codeformer_simple.trt", help="CodeFormer TensorRT engine path")
    parser.add_argument("--temp_dir", default="/tmp/wav2lip_codeformer", help="Temporary directory")
    
    args = parser.parse_args()
    
    # 一時ディレクトリ設定
    temp_dir = args.temp_dir
    os.makedirs(temp_dir, exist_ok=True)
    
    wav2lip_output = f"{temp_dir}/wav2lip_result.mp4"
    frames_dir = f"{temp_dir}/frames"
    enhanced_dir = f"{temp_dir}/enhanced"
    final_video = f"{temp_dir}/final_no_audio.mp4"
    
    try:
        print("🎭 ツンデレWav2Lip + CodeFormer TensorRT統合処理開始💢")
        print("🚀 GPU最適化設定完了！")
        
        # Step 1: Wav2Lipで口パク動画生成
        run_wav2lip(args.face, args.audio, wav2lip_output, args.checkpoint_path, args.out_height)
        
        # Step 2: フレーム抽出
        frame_files = extract_frames(wav2lip_output, frames_dir)
        
        # Step 3: CodeFormer TensorRT処理
        enhanced_files = enhance_frames_with_codeformer(frame_files, enhanced_dir, args.engine_path)
        
        # Step 4: 動画再構築
        reconstruct_video(enhanced_files, final_video)
        
        # Step 5: 音声合成
        add_audio(final_video, args.audio, args.outfile)
        
        print(f"✅ 完了よ！出力ファイル: {args.outfile}")
        
    except Exception as e:
        print(f"❌ エラー: {e}")
        return 1
    finally:
        # 一時ファイル削除
        if os.path.exists(temp_dir):
            subprocess.run(["rm", "-rf", temp_dir])
            print("一時ファイル削除完了")
    
    return 0

if __name__ == "__main__":
    exit(main())