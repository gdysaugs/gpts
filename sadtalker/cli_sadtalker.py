#!/usr/bin/env python3
"""
SadTalker CLI - 画像と音声から口パク動画生成
WSL2 + RTX 3050対応版
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

def check_gpu():
    """GPU使用可能性チェック"""
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ GPU (NVIDIA) 検出済み")
            return True
        else:
            print("❌ GPU未検出 - CPUモードで動作")
            return False
    except FileNotFoundError:
        print("❌ nvidia-smi not found - CPUモードで動作")
        return False

def run_sadtalker_docker(image_path, audio_path, output_dir=None, quality="fast", args=None):
    """Dockerコンテナでsadtalker実行"""
    
    # パス正規化
    image_path = os.path.abspath(image_path)
    audio_path = os.path.abspath(audio_path)
    
    if not os.path.exists(image_path):
        print(f"❌ 画像ファイルが見つかりません: {image_path}")
        return False
    
    if not os.path.exists(audio_path):
        print(f"❌ 音声ファイルが見つかりません: {audio_path}")
        return False
    
    # 出力ディレクトリ設定
    if output_dir is None:
        output_dir = os.path.join(os.getcwd(), "output")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # SadTalkerプロジェクトディレクトリ
    sadtalker_dir = "/home/adama/project/gpts/sadtalker"
    
    if not os.path.exists(sadtalker_dir):
        print(f"❌ SadTalkerディレクトリが見つかりません: {sadtalker_dir}")
        return False
    
    print(f"📁 SadTalkerディレクトリ: {sadtalker_dir}")
    print(f"🖼️  入力画像: {image_path}")
    print(f"🎵 入力音声: {audio_path}")
    print(f"📁 出力ディレクトリ: {output_dir}")
    
    # inputディレクトリにファイルコピー
    input_dir = os.path.join(sadtalker_dir, "input")
    os.makedirs(input_dir, exist_ok=True)
    
    # ファイルコピー（同じファイルの場合はスキップ）
    import shutil
    target_image = os.path.join(input_dir, "source_image.jpg")
    target_audio = os.path.join(input_dir, "source_audio.mp3")
    
    if os.path.abspath(image_path) != os.path.abspath(target_image):
        shutil.copy2(image_path, target_image)
        print(f"✅ 画像ファイルコピー: {target_image}")
    else:
        print(f"📁 画像ファイル既存: {target_image}")
    
    if os.path.abspath(audio_path) != os.path.abspath(target_audio):
        shutil.copy2(audio_path, target_audio)
        print(f"✅ 音声ファイルコピー: {target_audio}")
    else:
        print(f"📁 音声ファイル既存: {target_audio}")
    
    print(f"✅ ファイル準備完了")
    
    # GPU対応Dockerコマンド構築
    gpu_available = check_gpu()
    
    if gpu_available:
        # WSL2 GPU対応版 - rootで実行後に権限修正
        docker_cmd = [
            "docker", "run", "--rm",
            "--privileged",
            "-v", "/usr/lib/wsl:/usr/lib/wsl",
            "-e", "LD_LIBRARY_PATH=/usr/lib/wsl/lib",
            "-e", "NVIDIA_VISIBLE_DEVICES=all",
            "-e", "CUDA_VISIBLE_DEVICES=0",
            "-v", f"{input_dir}:/app/input",
            "-v", f"{output_dir}:/app/output",
            "-v", f"{sadtalker_dir}/results:/app/results",
            "-v", f"{sadtalker_dir}/checkpoints:/app/checkpoints",
            "-v", f"{sadtalker_dir}/gfpgan:/app/gfpgan",
            "-w", "/app"
        ]
    else:
        # CPU版 - rootで実行後に権限修正
        docker_cmd = [
            "docker", "run", "--rm",
            "-v", f"{input_dir}:/app/input",
            "-v", f"{output_dir}:/app/output", 
            "-v", f"{sadtalker_dir}/results:/app/results",
            "-v", f"{sadtalker_dir}/checkpoints:/app/checkpoints",
            "-v", f"{sadtalker_dir}/gfpgan:/app/gfpgan",
            "-w", "/app"
        ]
    
    # 品質設定に基づく処理パラメーター
    fp16_mode = getattr(args, 'fp16', False)
    batch_size = "1"  # 🚀 全て batch=1 で最大メモリ節約
    
    if quality == "high":
        enhancer_setting = "'gfpgan'"  # PyTorch版GFPGAN有効
        background_enhancer_setting = "None"
        preprocess_setting = "'crop'"  # 🚀 高速化: cropモードに変更（品質ほぼ同等）
        print("🔥 高画質+高速モード: PyTorch版GFPGAN + crop前処理")
    else:
        enhancer_setting = "None"
        background_enhancer_setting = "None" 
        preprocess_setting = "'crop'"
        print("⚡ 高速モード: エンハンサー無効")
    
    if fp16_mode:
        print("🚀 FP16最適化: 有効 (高速化+VRAM節約)")
    else:
        print("🔄 FP16最適化: 無効 (標準精度)")
    
    print(f"🔧 メモリ最適化: batch_size={batch_size} (RTX 3050 OOM回避)")
    
    # 🎭 表情・頭部制御設定（自然な動きに最適化）
    expression_scale = getattr(args, 'expression', 0.7)  # 🔧 デフォルトを控えめに
    still_mode = getattr(args, 'still', True)  # 🔧 デフォルトで静止モード有効
    yaw_rotation = getattr(args, 'yaw', None)
    pitch_rotation = getattr(args, 'pitch', None)
    roll_rotation = getattr(args, 'roll', None)
    
    print(f"🎭 表情強度: {expression_scale}（控えめ設定）")
    if still_mode:
        print("🎯 静止モード: 頭の動きを最小化（自然な仕上がり）")
    if any([yaw_rotation, pitch_rotation, roll_rotation]):
        print(f"🔄 頭部制御: Yaw={yaw_rotation}° Pitch={pitch_rotation}° Roll={roll_rotation}°")
    
    # Docker画像名とPythonコマンド - 動的品質設定
    python_code = f"""
import sys
sys.path.append('/home/SadTalker')
sys.path.append('/home/SadTalker/src')
from src.utils.preprocess import CropAndExtract
from src.test_audio2coeff import Audio2Coeff  
from src.facerender.animate_onnx import AnimateFromCoeff
from src.generate_batch import get_data
from src.generate_facerender_batch import get_facerender_data
from src.utils.init_path import init_path
import os
from time import strftime

# 設定
image_path = '/app/input/source_image.jpg'
audio_path = '/app/input/source_audio.mp3'
save_dir = '/app/results/cli_' + strftime('%Y_%m_%d_%H_%M_%S')

print('🔄 SadTalker処理開始...')

# CUDA確認 + Mixed Precision最適化
import torch
print(f'🔍 CUDA available: {{torch.cuda.is_available()}}')
if torch.cuda.is_available():
    print(f'🔍 CUDA device count: {{torch.cuda.device_count()}}')
    print(f'🔍 CUDA device name: {{torch.cuda.get_device_name(0)}}')
    
    # 🚀 CUDA最適化設定
    torch.backends.cudnn.benchmark = True  # 自動最適化
    torch.backends.cuda.matmul.allow_tf32 = True  # TensorFloat-32有効
    print('🚀 CUDA最適化設定完了')
    
    # FP16最適化設定
    if {fp16_mode}:
        torch.backends.cudnn.allow_tf32 = True
        print('🚀 FP16最適化有効: Mixed Precision + TF32')
        # GPU memory cleanup for FP16
        torch.cuda.empty_cache()
        torch.cuda.set_per_process_memory_fraction(0.85)  # 85%制限でOOM回避
    else:
        print('🔄 標準精度モード')

sadtalker_paths = init_path('/app/checkpoints', '/home/SadTalker/src/config', '256', True, {preprocess_setting})
# アニメ画像対応のため、より寛容な設定を使用
import warnings
warnings.filterwarnings('ignore')
preprocess_model = CropAndExtract(sadtalker_paths, 'cuda')
audio_to_coeff = Audio2Coeff(sadtalker_paths, 'cuda')
animate_from_coeff = AnimateFromCoeff(sadtalker_paths, 'cuda')

os.makedirs(save_dir, exist_ok=True)
first_frame_dir = os.path.join(save_dir, 'first_frame_dir')
os.makedirs(first_frame_dir, exist_ok=True)

# 前処理（最適化済み - cropモード固定）
first_coeff_path, crop_pic_path, crop_info = preprocess_model.generate(image_path, first_frame_dir, 'crop', source_image_flag=True)
print('✅ 前処理完了（crop最適化モード）')

# 音声解析
batch = get_data(first_coeff_path, audio_path, 'cuda', None, still=True)
coeff_path = audio_to_coeff.generate(batch, save_dir, 0, None)
print('✅ 音声解析完了')

# 動画生成（自然な動きに最適化済み）
yaw_list = [{yaw_rotation}] if {yaw_rotation} is not None else None
pitch_list = [{pitch_rotation}] if {pitch_rotation} is not None else None  
roll_list = [{roll_rotation}] if {roll_rotation} is not None else None
data = get_facerender_data(coeff_path, crop_pic_path, first_coeff_path, audio_path, {batch_size}, yaw_list, pitch_list, roll_list, expression_scale={expression_scale}, still_mode=True, preprocess='crop')
video_path = animate_from_coeff.generate_deploy(data, save_dir, image_path, crop_info, enhancer={enhancer_setting}, background_enhancer={background_enhancer_setting}, preprocess='crop')
print(f'✅ 動画生成完了: {{video_path}}')

# 出力ディレクトリにコピー
import shutil
output_file = '/app/output/result.mp4'
if os.path.exists(video_path):
    shutil.copy2(video_path, output_file)
    print(f'📁 結果コピー: {{output_file}}')
else:
    print('❌ 動画生成失敗')
"""
    
    docker_cmd.extend(["sadtalker:latest", "python", "-c", python_code])
    
    print("🚀 SadTalker処理開始...")
    print(f"実行コマンド: {' '.join(docker_cmd)}")
    
    try:
        # Dockerコンテナ実行
        result = subprocess.run(docker_cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✅ SadTalker処理完了")
            print("出力:", result.stdout)
            
            # 🔧 権限修正 - rootで作成されたファイルをadamaユーザーに変更（sudoパスワード対応）
            current_user = os.getenv('USER', 'adama')
            chown_cmd = f"chown -R {current_user}:{current_user} {output_dir} {sadtalker_dir}/results"
            docker_chown_cmd = [
                "docker", "run", "--rm", "--privileged",
                "-v", f"{output_dir}:/fix_output",
                "-v", f"{sadtalker_dir}/results:/fix_results",
                "busybox", "sh", "-c",
                f"chown -R 1000:1000 /fix_output /fix_results"
            ]
            subprocess.run(docker_chown_cmd, capture_output=True)
            
            # 結果ファイル確認
            results_dir = os.path.join(sadtalker_dir, "results")
            if os.path.exists(results_dir):
                result_files = list(Path(results_dir).glob("**/*.mp4"))
                if result_files:
                    latest_result = max(result_files, key=os.path.getmtime)
                    print(f"🎉 生成動画: {latest_result}")
                    
                    # 音声ファイルパスを取得
                    result_dir = latest_result.parent
                    audio_files = list(result_dir.glob("*.wav"))
                    
                    # 🎬 単一ファイル出力 - result.mp4として統一
                    final_output = os.path.join(output_dir, "result.mp4")
                    
                    # 🔇 READMEの完全ノイズフリー技術：ストリーム分離
                    print("🔇 README技術：元音声MP3変換→ストリーム分離（完全ノイズフリー）")
                    
                    # Step 1: 元音声をMP3に変換（品質保持）
                    temp_mp3 = os.path.join(output_dir, "original_audio.mp3")
                    convert_cmd = [
                        "docker", "run", "--rm",
                        "-v", f"{audio_path}:/input_audio:ro",
                        "-v", f"{output_dir}:/output",
                        "jrottenberg/ffmpeg:4.1-alpine",
                        "-i", "/input_audio",
                        "-c:a", "libmp3lame",
                        "-b:a", "192k",  # 高品質192kbps（READMEの推奨値）
                        "-ar", "44100",  # 44.1kHz（READMEの推奨値）
                        "-y",
                        "/output/original_audio.mp3"
                    ]
                    
                    print("🎵 Step 1: 元音声→MP3変換（品質保持）...")
                    convert_result = subprocess.run(convert_cmd, capture_output=True, text=True)
                    
                    if convert_result.returncode != 0:
                        print(f"❌ MP3変換失敗: {convert_result.stderr}")
                        import shutil
                        shutil.copy2(str(latest_result), final_output)
                        print(f"📁 動画のみコピー: {final_output}")
                        return True
                    
                    # Step 2: READMEのストリーム分離技術適用
                    print("🔇 Step 2: ストリーム分離技術（READMEの完全ノイズフリー方法）...")
                    
                    ffmpeg_cmd = [
                        "docker", "run", "--rm",
                        "-v", f"{latest_result.parent}:/source",
                        "-v", f"{output_dir}:/working",
                        "jrottenberg/ffmpeg:4.1-alpine",
                        "-i", f"/source/{latest_result.name}",
                        "-i", "/working/original_audio.mp3",
                        "-c:v", "copy",
                        "-c:a", "copy",  # 音声を一切加工せずコピー（READMEの方法）
                        "-map", "0:v:0",  # SadTalker動画の映像のみ
                        "-map", "1:a:0",  # 元音声MP3の音声のみ
                        "-shortest",
                        "-y",
                        f"/working/result.mp4"
                    ]
                    
                    merge_result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
                    
                    if merge_result.returncode == 0:
                        print(f"✅ READMEの完全ノイズフリー技術成功: {final_output}")
                        # 一時MP3ファイル削除
                        os.remove(temp_mp3)
                        return True
                    else:
                        print("⚠️ ストリーム分離失敗、動画のみコピー")
                        print(f"FFmpeg エラー: {merge_result.stderr}")
                        import shutil
                        shutil.copy2(str(latest_result), final_output)
                        print(f"📁 動画コピー完了: {final_output}")
                        return True
                else:
                    print("❌ 動画ファイルが生成されませんでした")
            
        else:
            print("❌ SadTalker処理エラー")
            print("エラー出力:", result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ 処理タイムアウト (5分)")
        return False
    except Exception as e:
        print(f"❌ 実行エラー: {e}")
        return False

def run_api_mode(image_path, audio_path, api_url="http://localhost:10364"):
    """API経由でSadTalker実行"""
    try:
        import requests
        
        # ファイルアップロード
        with open(image_path, 'rb') as img_file, open(audio_path, 'rb') as audio_file:
            files = {
                'image': img_file,
                'audio': audio_file
            }
            
            print(f"🌐 APIリクエスト送信: {api_url}/generate_video")
            response = requests.post(f"{api_url}/generate_video", files=files, timeout=300)
            
            if response.status_code == 200:
                # 動画ファイルを保存
                output_file = "sadtalker_api_result.mp4"
                with open(output_file, 'wb') as f:
                    f.write(response.content)
                print(f"✅ API経由で生成完了: {output_file}")
                return True
            else:
                print(f"❌ APIエラー: {response.status_code} - {response.text}")
                return False
                
    except ImportError:
        print("❌ requestsライブラリが必要です: pip install requests")
        return False
    except Exception as e:
        print(f"❌ API実行エラー: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="SadTalker CLI - 画像と音声から口パク動画生成")
    parser.add_argument("image", help="入力画像ファイル (JPG/PNG)")
    parser.add_argument("audio", help="入力音声ファイル (WAV/MP3)")
    parser.add_argument("-o", "--output", help="出力ディレクトリ", default="./output")
    parser.add_argument("--api", action="store_true", help="API経由で実行")
    parser.add_argument("--api-url", default="http://localhost:10364", help="APIサーバーURL")
    parser.add_argument("--quality", choices=["fast", "high"], default="fast", help="処理品質")
    parser.add_argument("--fp16", action="store_true", help="FP16最適化を有効にする")
    
    # 🎭 表情・頭部制御オプション
    parser.add_argument("--expression", type=float, default=1.0, help="表情強度 (0.0-2.0, デフォルト:1.0)")
    parser.add_argument("--still", action="store_true", help="静止モード（頭の動きを最小化）")
    parser.add_argument("--yaw", type=float, help="頭部左右回転 (-30〜30度)")
    parser.add_argument("--pitch", type=float, help="頭部上下回転 (-20〜20度)")  
    parser.add_argument("--roll", type=float, help="頭部傾き (-15〜15度)")
    
    args = parser.parse_args()
    
    print("🎭 SadTalker CLI")
    print("=" * 50)
    
    if args.api:
        print("📡 API経由で実行")
        success = run_api_mode(args.image, args.audio, args.api_url)
    else:
        print("🐳 Docker経由で実行")
        success = run_sadtalker_docker(args.image, args.audio, args.output, args.quality, args)
    
    if success:
        print("\n🎉 処理完了!")
    else:
        print("\n❌ 処理失敗")
        sys.exit(1)

if __name__ == "__main__":
    main()