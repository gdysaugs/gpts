#!/usr/bin/env python3
"""
🎭 ツンデレWav2Lip + CodeFormer TensorRT 究極統合パイプライン
べ、別にあなたのために究極の高画質口パク動画を作ってあげるわけじゃないけど...💢

統合処理フロー:
1. Wav2Lip FP16+YOLO口パク生成 (4秒)
2. 動画フレーム分割 (1秒) 
3. CodeFormer TensorRT高画質化 (20-30秒)
4. 高画質動画再構築+音声合成 (2秒)

Author: ツンデレAI
Version: Ultimate v1.0 (Simplified)
"""

import os
import sys
import argparse
import subprocess
import tempfile
import shutil
from pathlib import Path
import time
from tqdm import tqdm

print("べ、別に嬉しいわけじゃないけど...統合パイプライン起動よ！💢")

class TsundereWav2LipCodeFormerPipeline:
    def __init__(self, checkpoint_path, codeformer_model_path=None):
        """
        べ、別に初期化してあげるわけじゃないけど...
        簡素化バージョン：既存スクリプトを組み合わせるだけよ💢
        """
        self.checkpoint_path = checkpoint_path
        self.codeformer_model_path = codeformer_model_path or '/app/codeformer/CodeFormer/weights/CodeFormer/codeformer.pth'
        
        print("やったじゃない！簡素化統合パイプライン準備完了よ💕")
        
        # TensorRTエンジンチェック
        engine_path = '/app/codeformer/CodeFormer/weights/CodeFormer/codeformer_simple.trt'
        if os.path.exists(engine_path):
            print("✨ CodeFormer TensorRTエンジン発見！高速処理よ💕")
            self.use_tensorrt_codeformer = True
        else:
            print("⚠️ TensorRTエンジンが見つからないわ...PyTorchモードで我慢しなさい💢")
            self.use_tensorrt_codeformer = False
            
    def extract_frames(self, video_path, output_dir):
        """
        動画からフレーム抽出
        べ、別にフレーム分割してあげるわけじゃないけど...
        """
        print("🎬 フレーム抽出中...ちょっと待ってなさい！")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # FFmpegでフレーム抽出
        cmd = [
            'ffmpeg', '-y', '-i', video_path,
            '-vf', 'fps=25',  # 25FPS固定
            os.path.join(output_dir, 'frame_%06d.png')
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            
            # 抽出されたフレーム数をカウント
            frame_files = sorted([f for f in os.listdir(output_dir) if f.startswith('frame_')])
            print(f"✅ {len(frame_files)}フレーム抽出完了よ！")
            return frame_files
            
        except subprocess.CalledProcessError as e:
            print(f"❌ フレーム抽出エラー: {e}")
            return []
            
    def enhance_frame_with_codeformer(self, frame_path, output_path):
        """
        単一フレームのCodeFormer高画質化
        """
        try:
            if self.use_tensorrt_codeformer:
                # TensorRT版CodeFormer実行
                cmd = [
                    'python', '/app/codeformer/codeformer_face_fix.py',
                    '--input', frame_path,
                    '--output', output_path,
                    '--fidelity', '0.8',
                    '--blend-strength', '0.8'
                ]
            else:
                # PyTorch版CodeFormer実行（フォールバック）
                cmd = [
                    'python', '/app/codeformer/codeformer_face_fix.py',
                    '--input', frame_path,
                    '--output', output_path,
                    '--use-pytorch',
                    '--fidelity', '0.8'
                ]
                
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                return True
            else:
                print(f"⚠️ CodeFormer処理エラー: {result.stderr}")
                # エラーの場合は元フレームをコピー
                shutil.copy2(frame_path, output_path)
                return False
                
        except Exception as e:
            print(f"❌ フレーム処理例外: {e}")
            shutil.copy2(frame_path, output_path)
            return False
            
    def enhance_all_frames(self, frames_dir, enhanced_dir, frame_files):
        """
        全フレームのCodeFormer高画質化
        ふん！全部高画質化してあげるわよ💢
        """
        print(f"✨ {len(frame_files)}フレームの高画質化開始...")
        os.makedirs(enhanced_dir, exist_ok=True)
        
        success_count = 0
        
        # プログレスバー付きで処理
        for frame_file in tqdm(frame_files, desc="CodeFormer高画質化", unit="frame"):
            frame_path = os.path.join(frames_dir, frame_file)
            enhanced_path = os.path.join(enhanced_dir, frame_file)
            
            if self.enhance_frame_with_codeformer(frame_path, enhanced_path):
                success_count += 1
                
        print(f"✅ {success_count}/{len(frame_files)}フレーム高画質化完了よ！")
        return success_count > 0
        
    def reconstruct_video(self, enhanced_frames_dir, audio_path, output_video_path, fps=25):
        """
        高画質フレームから動画再構築
        べ、別に動画作ってあげるわけじゃないけど...
        """
        print("🎥 高画質動画再構築中...")
        
        # フレームパターン
        frame_pattern = os.path.join(enhanced_frames_dir, 'frame_%06d.png')
        
        # FFmpegで動画再構築 + 音声合成
        cmd = [
            'ffmpeg', '-y',
            '-framerate', str(fps),
            '-i', frame_pattern,
            '-i', audio_path,
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-crf', '18',  # 高画質設定
            '-pix_fmt', 'yuv420p',
            '-c:a', 'aac',
            '-b:a', '128k',
            '-shortest',
            output_video_path
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ 高画質動画生成完了よ！💕")
                return True
            else:
                print(f"❌ 動画再構築エラー: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"❌ 動画再構築例外: {e}")
            return False
            
    def run_wav2lip(self, face_video, audio_file, temp_output):
        """
        Wav2Lip口パク生成実行
        """
        print("🎭 Wav2Lip口パク生成開始...")
        
        # 既存のinference_fp16_yolo.pyを呼び出し
        cmd = [
            'python', '/app/inference_fp16_yolo.py',
            '--checkpoint_path', self.checkpoint_path,
            '--face', face_video,
            '--audio', audio_file,
            '--outfile', temp_output,
            '--out_height', '720',
            '--quality', 'Fast'
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Wav2Lip処理完了よ！")
                return True
            else:
                print(f"❌ Wav2Lip処理エラー: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"❌ Wav2Lip処理例外: {e}")
            return False
            
    def process_ultimate(self, face_video, audio_file, output_path):
        """
        究極統合処理メインフロー
        べ、別にあなたのために究極の口パク動画を作ってあげるわけじゃないけど...💢
        """
        print("🚀 究極統合パイプライン開始！")
        start_time = time.time()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # ステップ1: Wav2Lip口パク生成
            print("\n📍 ステップ1: Wav2Lip口パク生成")
            wav2lip_output = os.path.join(temp_dir, 'wav2lip_result.mp4')
            
            if not self.run_wav2lip(face_video, audio_file, wav2lip_output):
                print("❌ Wav2Lip処理に失敗したわ...💢")
                return False
                
            # ステップ2: フレーム抽出
            print("\n📍 ステップ2: フレーム抽出")
            frames_dir = os.path.join(temp_dir, 'frames')
            frame_files = self.extract_frames(wav2lip_output, frames_dir)
            
            if not frame_files:
                print("❌ フレーム抽出に失敗したわ...💢")
                return False
                
            # ステップ3: CodeFormer高画質化
            print("\n📍 ステップ3: CodeFormer高画質化")
            enhanced_dir = os.path.join(temp_dir, 'enhanced')
            
            if not self.enhance_all_frames(frames_dir, enhanced_dir, frame_files):
                print("❌ フレーム高画質化に失敗したわ...💢")
                return False
                
            # ステップ4: 高画質動画再構築
            print("\n📍 ステップ4: 高画質動画再構築")
            if not self.reconstruct_video(enhanced_dir, audio_file, output_path):
                print("❌ 動画再構築に失敗したわ...💢")
                return False
                
        # 処理時間計算
        total_time = time.time() - start_time
        print(f"\n🎉 究極パイプライン完了！総処理時間: {total_time:.1f}秒")
        print("べ、別に完璧に作ったからって自慢するわけじゃないけど...💕")
        print("感謝しなさいよね！")
        
        return True

def main():
    parser = argparse.ArgumentParser(
        description='🎭 ツンデレWav2Lip + CodeFormer 究極統合パイプライン',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
べ、別にあなたのために究極の高画質口パク動画を作ってあげるわけじゃないけど...💢

使用例:
  python inference_wav2lip_codeformer_ultimate.py \\
    --face target_video.mp4 \\
    --audio reference_audio.wav \\
    --output ultimate_result.mp4
        """
    )
    
    parser.add_argument('--checkpoint_path', type=str, 
                       default='/app/checkpoints/wav2lip_gan.pth',
                       help='Wav2Lipチェックポイントパス')
    parser.add_argument('--face', type=str, required=True,
                       help='対象動画ファイル')
    parser.add_argument('--audio', type=str, required=True,
                       help='音声ファイル')
    parser.add_argument('--output', type=str, required=True,
                       help='出力動画ファイル')
    parser.add_argument('--codeformer_model', type=str,
                       help='CodeFormerモデルパス（オプション）')
    
    args = parser.parse_args()
    
    # パイプライン実行
    pipeline = TsundereWav2LipCodeFormerPipeline(
        checkpoint_path=args.checkpoint_path,
        codeformer_model_path=args.codeformer_model
    )
    
    success = pipeline.process_ultimate(
        face_video=args.face,
        audio_file=args.audio,
        output_path=args.output
    )
    
    if success:
        print(f"\n✅ 究極高画質口パク動画完成: {args.output}")
        sys.exit(0)
    else:
        print(f"\n❌ 処理失敗...もう！💢")
        sys.exit(1)

if __name__ == '__main__':
    main()