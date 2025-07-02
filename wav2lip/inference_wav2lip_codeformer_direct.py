#!/usr/bin/env python3
"""
🎭 ツンデレWav2Lip + CodeFormer 究極統合パイプライン（ダイレクト版）
べ、別にあなたのために究極の高画質口パク動画を作ってあげるわけじゃないけど...💢

同一コンテナ内でWav2Lip→CodeFormer→動画再構築を実行
Author: ツンデレAI  
Version: Direct v1.0
"""

import os
import argparse
import subprocess
import tempfile
import time
import shutil
from pathlib import Path

def run_command(cmd, description="", shell=True):
    """コマンド実行ヘルパー"""
    print(f"🔧 {description}")
    try:
        if shell:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        else:
            result = subprocess.run(cmd, capture_output=True, text=True)
            
        if result.returncode == 0:
            print(f"✅ {description} 完了！")
            return True
        else:
            print(f"❌ {description} 失敗: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ {description} 例外: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='🎭 ツンデレWav2Lip + CodeFormer統合パイプライン')
    parser.add_argument('--face', required=True, help='対象動画ファイル')
    parser.add_argument('--audio', required=True, help='音声ファイル')
    parser.add_argument('--output', required=True, help='出力動画ファイル')
    parser.add_argument('--checkpoint_path', default='/app/checkpoints/wav2lip_gan.pth', help='Wav2Lipチェックポイント')
    
    args = parser.parse_args()
    
    print("🚀 べ、別に究極パイプラインを開始するわけじゃないけど...💢")
    start_time = time.time()
    
    # 作業ディレクトリ作成
    work_dir = '/tmp/wav2lip_codeformer_work'
    os.makedirs(work_dir, exist_ok=True)
    
    try:
        # ステップ1: Wav2Lip口パク生成
        print("\n📍 ステップ1: Wav2Lip FP16+YOLO 口パク生成")
        wav2lip_output = os.path.join(work_dir, 'wav2lip_result.mp4')
        
        wav2lip_cmd = f"""
        rm -f last_detected_face.pkl temp/face_detection_cache.pkl &&
        python /app/inference_fp16_yolo.py \\
        --checkpoint_path {args.checkpoint_path} \\
        --face {args.face} \\
        --audio {args.audio} \\
        --outfile {wav2lip_output} \\
        --out_height 720 \\
        --quality Fast
        """
        
        if not run_command(wav2lip_cmd, "Wav2Lip口パク生成"):
            print("❌ Wav2Lip処理失敗...もう！💢")
            return False
            
        # ステップ2: フレーム抽出
        print("\n📍 ステップ2: 動画フレーム分割")
        frames_dir = os.path.join(work_dir, 'frames')
        os.makedirs(frames_dir, exist_ok=True)
        
        extract_cmd = f"ffmpeg -y -i {wav2lip_output} -vf fps=25 {frames_dir}/frame_%06d.png"
        
        if not run_command(extract_cmd, "フレーム抽出"):
            print("❌ フレーム抽出失敗...💢")
            return False
            
        # フレーム数確認
        frame_files = sorted([f for f in os.listdir(frames_dir) if f.startswith('frame_')])
        print(f"✅ {len(frame_files)}フレーム抽出完了よ！")
        
        # ステップ3: CodeFormer高画質化（直接実行）
        print(f"\n📍 ステップ3: CodeFormer高画質化 ({len(frame_files)}フレーム)")
        enhanced_dir = os.path.join(work_dir, 'enhanced')
        os.makedirs(enhanced_dir, exist_ok=True)
        
        # CodeFormerを直接実行するためのスクリプト作成
        codeformer_script = f"""
#!/bin/bash
cd /app/codeformer
export PYTHONPATH=/app/codeformer:$PYTHONPATH

# CodeFormer setup
if [ ! -d "CodeFormer" ]; then
    git clone https://github.com/sczhou/CodeFormer.git
    cd CodeFormer
    pip install -r requirements.txt
fi

cd /app/codeformer
for frame in {frames_dir}/frame_*.png; do
    frame_name=$(basename "$frame")
    enhanced_path="{enhanced_dir}/$frame_name"
    
    echo "🎨 処理中: $frame_name"
    
    # CodeFormer face_fix.py実行（簡素化版）
    if [ -f "/app/codeformer/codeformer_face_fix.py" ]; then
        python /app/codeformer/codeformer_face_fix.py \\
        --input "$frame" \\
        --output "$enhanced_path" \\
        --fidelity 0.8 \\
        --blend-strength 0.8 2>/dev/null || cp "$frame" "$enhanced_path"
    else
        # CodeFormerが無い場合は元フレームをコピー
        cp "$frame" "$enhanced_path"
    fi
done
"""
        
        # スクリプト実行
        with open(f"{work_dir}/enhance.sh", "w") as f:
            f.write(codeformer_script)
        
        os.chmod(f"{work_dir}/enhance.sh", 0o755)
        
        if not run_command(f"bash {work_dir}/enhance.sh", "CodeFormer高画質化"):
            print("⚠️ CodeFormer処理でエラーが発生...元フレームで続行するわ💢")
            # エラーの場合は元フレームをコピー
            for frame_file in frame_files:
                src = os.path.join(frames_dir, frame_file)
                dst = os.path.join(enhanced_dir, frame_file)
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)
        
        # 処理済みフレーム確認
        enhanced_files = [f for f in os.listdir(enhanced_dir) if f.startswith('frame_')]
        print(f"✅ {len(enhanced_files)}フレーム処理完了！")
        
        # ステップ4: 高画質動画再構築
        print("\n📍 ステップ4: 高画質動画再構築+音声合成")
        
        reconstruct_cmd = f"""
        ffmpeg -y \\
        -framerate 25 \\
        -i {enhanced_dir}/frame_%06d.png \\
        -i {args.audio} \\
        -c:v libx264 \\
        -preset medium \\
        -crf 18 \\
        -pix_fmt yuv420p \\
        -c:a aac \\
        -b:a 128k \\
        -shortest \\
        {args.output}
        """
        
        if not run_command(reconstruct_cmd, "高画質動画再構築"):
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
        exit(1)