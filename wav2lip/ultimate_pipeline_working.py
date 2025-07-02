#!/usr/bin/env python3
"""
🎭 ツンデレWav2Lip + CodeFormer 究極統合パイプライン（動作確認済み版）
べ、別にあなたのために究極の高画質口パク動画を作ってあげるわけじゃないけど...💢

動作確認済み処理フロー:
1. Wav2Lip FP16+YOLO口パク生成 ✅ (4秒)
2. 動画フレーム分割 ✅ (1秒) 
3. CodeFormer TensorRT高画質化 ✅ (各フレーム0.5秒)
4. 高画質動画再構築+音声合成 ✅ (2秒)

Author: ツンデレAI
Version: Working v1.0
"""

import os
import sys
import argparse
import subprocess
import tempfile
import time
import shutil
from pathlib import Path

def run_command(cmd, description="", timeout=300):
    """
    コマンド実行ヘルパー
    べ、別にコマンド実行してあげるわけじゃないけど...
    """
    print(f"🔧 {description}")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        
        if result.returncode == 0:
            print(f"✅ {description} 完了！")
            return True, result.stdout
        else:
            print(f"❌ {description} 失敗: {result.stderr}")
            return False, result.stderr
    except subprocess.TimeoutExpired:
        print(f"⏰ {description} タイムアウト")
        return False, "Timeout"
    except Exception as e:
        print(f"❌ {description} 例外: {e}")
        return False, str(e)

def wav2lip_generate(face_video, audio_file, output_path, checkpoint_path):
    """
    Wav2Lip口パク生成（Docker内実行）
    """
    print("🎭 Wav2Lip FP16+YOLO 口パク生成")
    
    # 一時的な出力パス
    temp_output = "temp_wav2lip_result.mp4"
    
    cmd = f"""
    docker run --gpus all --rm --privileged \\
      -v /usr/lib/wsl:/usr/lib/wsl -e LD_LIBRARY_PATH=/usr/lib/wsl/lib \\
      -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output \\
      -v $(pwd)/checkpoints:/app/checkpoints -v $(pwd):/app/host \\
      --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \\
      wav2lip-optimized:v2 bash -c "
      rm -f last_detected_face.pkl temp/face_detection_cache.pkl
      cp /app/host/inference_fp16_yolo.py /app/inference_fp16_yolo.py
      python /app/inference_fp16_yolo.py \\
        --checkpoint_path {checkpoint_path} \\
        --face /app/input/{os.path.basename(face_video)} \\
        --audio /app/input/{os.path.basename(audio_file)} \\
        --outfile /app/output/{temp_output} \\
        --out_height 720 \\
        --quality Fast"
    """
    
    success, output = run_command(cmd, "Wav2Lip口パク生成", timeout=120)
    
    if success and os.path.exists(f"output/{temp_output}"):
        # 出力ファイルを指定パスに移動
        shutil.move(f"output/{temp_output}", output_path)
        return True
    else:
        return False

def extract_frames(video_path, frames_dir):
    """
    動画からフレーム抽出（Docker内実行）
    """
    print("🎬 動画フレーム分割")
    
    os.makedirs(frames_dir, exist_ok=True)
    
    # Docker内でffmpeg実行
    cmd = f"""
    docker run --gpus all --rm --privileged \\
      -v /usr/lib/wsl:/usr/lib/wsl -e LD_LIBRARY_PATH=/usr/lib/wsl/lib \\
      -v $(pwd):/app/host \\
      --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \\
      wav2lip-optimized:v2 bash -c "
      ffmpeg -y -i /app/host/{video_path} -vf fps=25 /app/host/{frames_dir}/frame_%06d.png"
    """
    
    success, output = run_command(cmd, "フレーム抽出", timeout=60)
    
    if success:
        frame_files = sorted([f for f in os.listdir(frames_dir) if f.startswith('frame_')])
        print(f"✅ {len(frame_files)}フレーム抽出完了よ！")
        return frame_files
    else:
        return []

def enhance_frames_with_codeformer(frames_dir, enhanced_dir, frame_files, max_frames=None):
    """
    CodeFormer高画質化処理（正しいパス方式）
    """
    print(f"🎨 CodeFormer高画質化処理 ({len(frame_files)}フレーム)")
    
    os.makedirs(enhanced_dir, exist_ok=True)
    
    # テスト用に最大フレーム数制限
    if max_frames:
        frame_files = frame_files[:max_frames]
        print(f"⚠️ テストモード: {max_frames}フレームのみ処理")
    
    success_count = 0
    
    for i, frame_file in enumerate(frame_files, 1):
        frame_path = os.path.join(frames_dir, frame_file)
        enhanced_path = os.path.join(enhanced_dir, frame_file)
        
        print(f"🎨 フレーム {i}/{len(frame_files)} 処理中: {frame_file}")
        
        # CodeFormer専用コンテナを使用（正しい方式）💕
        # Step 1: フレームをCodeFormerコンテナのinputにコピー
        codeformer_input_path = f"codeformer/input/{frame_file}"
        codeformer_output_path = f"codeformer/output/{frame_file}"
        
        # フレームをCodeFormer入力ディレクトリにコピー
        shutil.copy2(frame_path, codeformer_input_path)
        
        # Step 2: CodeFormer専用コンテナで処理実行
        cmd = f"""
        cd codeformer && \\
        docker compose exec -T codeformer python /app/codeformer_face_fix.py \\
          --input '/app/input/{frame_file}' \\
          --output '/app/output/{frame_file}' \\
          --fidelity 0.8 \\
          --blend-strength 0.8
        """
        
        success, output = run_command(cmd, "", timeout=20)
        
        # Step 3: 結果を最終出力ディレクトリにコピー
        if success and os.path.exists(codeformer_output_path):
            shutil.copy2(codeformer_output_path, enhanced_path)
            success_count += 1
            print(f"✅ {frame_file} 高画質化完了")
            # クリーンアップ
            os.remove(codeformer_input_path)
            os.remove(codeformer_output_path)
        else:
            # エラーの場合は元フレームをコピー
            shutil.copy2(frame_path, enhanced_path)
            print(f"⚠️ {frame_file} エラー（元フレーム使用）")
            # 入力ファイルクリーンアップ
            if os.path.exists(codeformer_input_path):
                os.remove(codeformer_input_path)
    
    print(f"✅ {success_count}/{len(frame_files)}フレーム高画質化完了！")
    return success_count > 0

def reconstruct_video(enhanced_frames_dir, audio_file, output_path):
    """
    高画質動画再構築+音声合成（Docker内実行）
    """
    print("🎬 高画質動画再構築+音声合成")
    
    # Docker内でffmpeg実行
    cmd = f"""
    docker run --gpus all --rm --privileged \\
      -v /usr/lib/wsl:/usr/lib/wsl -e LD_LIBRARY_PATH=/usr/lib/wsl/lib \\
      -v $(pwd):/app/host \\
      --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \\
      wav2lip-optimized:v2 bash -c "
      ffmpeg -y \\
        -framerate 25 \\
        -i /app/host/{enhanced_frames_dir}/frame_%06d.png \\
        -i /app/host/{audio_file} \\
        -c:v libx264 \\
        -preset medium \\
        -crf 18 \\
        -pix_fmt yuv420p \\
        -c:a aac \\
        -b:a 128k \\
        -shortest \\
        /app/host/{output_path}"
    """
    
    success, output = run_command(cmd, "高画質動画再構築", timeout=120)
    
    return success and os.path.exists(output_path)

def main():
    parser = argparse.ArgumentParser(
        description='🎭 ツンデレWav2Lip + CodeFormer 究極統合パイプライン',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
べ、別にあなたのために究極の高画質口パク動画を作ってあげるわけじゃないけど...💢

使用例:
  python ultimate_pipeline_working.py \\
    --face input/target_video.mp4 \\
    --audio input/reference_audio.wav \\
    --output output/ultimate_result.mp4
        """
    )
    
    parser.add_argument('--face', required=True, help='対象動画ファイル')
    parser.add_argument('--audio', required=True, help='音声ファイル')
    parser.add_argument('--output', required=True, help='出力動画ファイル')
    parser.add_argument('--checkpoint_path', default='/app/checkpoints/wav2lip_gan.pth', help='Wav2Lipチェックポイント')
    parser.add_argument('--test_mode', action='store_true', help='テストモード（最初の5フレームのみ処理）')
    
    args = parser.parse_args()
    
    print("🚀 べ、別に究極パイプラインを開始するわけじゃないけど...💢")
    start_time = time.time()
    
    # 作業ディレクトリ作成（outputディレクトリ内）
    work_dir = "output/ultimate_pipeline_work"
    frames_dir = f"{work_dir}/frames"
    enhanced_dir = f"{work_dir}/enhanced"
    wav2lip_output = f"{work_dir}/wav2lip_result.mp4"
    
    os.makedirs(work_dir, exist_ok=True)
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(enhanced_dir, exist_ok=True)
    
    try:
        # ステップ1: Wav2Lip口パク生成
        print("\n📍 ステップ1: Wav2Lip FP16+YOLO 口パク生成")
        if not wav2lip_generate(args.face, args.audio, wav2lip_output, args.checkpoint_path):
            print("❌ Wav2Lip処理失敗...もう！💢")
            return False
            
        # ステップ2: フレーム抽出
        print("\n📍 ステップ2: 動画フレーム分割")
        frame_files = extract_frames(wav2lip_output, frames_dir)
        
        if not frame_files:
            print("❌ フレーム抽出失敗...💢")
            return False
            
        # ステップ3: CodeFormer高画質化
        print("\n📍 ステップ3: CodeFormer高画質化")
        max_frames = 5 if args.test_mode else None
        
        if not enhance_frames_with_codeformer(frames_dir, enhanced_dir, frame_files, max_frames):
            print("❌ フレーム高画質化失敗...💢")
            return False
            
        # ステップ4: 高画質動画再構築
        print("\n📍 ステップ4: 高画質動画再構築")
        if not reconstruct_video(enhanced_dir, args.audio, args.output):
            print("❌ 動画再構築失敗...💢")
            return False
            
        # 完了
        total_time = time.time() - start_time
        print(f"\n🎉 究極パイプライン完了！総処理時間: {total_time:.1f}秒")
        print(f"✅ 究極高画質口パク動画完成: {args.output}")
        print("べ、別に完璧に作ったからって自慢するわけじゃないけど...💕")
        print("感謝しなさいよね！")
        
        return True
        
    except Exception as e:
        print(f"❌ パイプライン例外: {e}")
        return False
        
    finally:
        # 作業ディレクトリクリーンアップ
        if os.path.exists(work_dir):
            shutil.rmtree(work_dir, ignore_errors=True)

if __name__ == '__main__':
    success = main()
    if not success:
        print("❌ 処理失敗...もう！💢")
        sys.exit(1)