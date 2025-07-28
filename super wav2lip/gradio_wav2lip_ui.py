#!/usr/bin/env python3
"""
Super Wav2Lip Gradio UI - Fixed Version
"""

import gradio as gr
import requests
import tempfile
import os
import time
import logging
import cv2
import numpy as np
from PIL import Image

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API URLs
SOVITS_API_URL = "http://gpt-sovits-api:8000"
WAV2LIP_API_URL = "http://super-wav2lip-fixed:8002"
SADTALKER_API_URL = "http://sadtalker-api:8000"  # NEW: SadTalker for images (internal port)

def image_to_video(image_path, duration=3.0, fps=30):
    """画像から動画を生成"""
    try:
        logger.info(f"🖼️ 画像から動画生成中: {image_path}")
        
        # 画像を読み込み
        img = cv2.imread(image_path)
        if img is None:
            # PILで試す（WebPなど他の形式）
            pil_img = Image.open(image_path)
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        
        height, width, channels = img.shape
        
        # 一時動画ファイルを作成
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_video:
            temp_video_path = temp_video.name
        
        # VideoWriterを設定
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))
        
        # 指定された秒数分のフレームを書き込み
        total_frames = int(duration * fps)
        for i in range(total_frames):
            out.write(img)
        
        out.release()
        logger.info(f"✅ 動画生成完了: {temp_video_path}")
        return temp_video_path
        
    except Exception as e:
        logger.error(f"画像→動画変換エラー: {e}")
        return None

def call_sadtalker_api(image_path, audio_path):
    """SadTalker API呼び出し（画像専用）"""
    try:
        logger.info(f"🎭 SadTalker API呼び出し中... URL: {SADTALKER_API_URL}")
        logger.info(f"🖼️ 画像ファイル: {image_path}")
        logger.info(f"🎵 音声ファイル: {audio_path}")
        
        with open(image_path, 'rb') as img_file, open(audio_path, 'rb') as aud_file:
            files = {
                'image': (os.path.basename(image_path), img_file, 'image/jpeg'),
                'audio': (os.path.basename(audio_path), aud_file, 'audio/wav')
            }
            data = {
                'quality': 'high',
                'fp16': 'true',
                'still_mode': 'true',                         # 静止モード固定
                'expression_scale': '1.0',                    # 表情強度1.0固定
                'preprocess': 'crop',                         # 前処理モードcrop固定
                'face_detector': 'retinaface',                # RetinaFace検出器固定
                'facerender_batch_size': '5',                 # 小さいバッチサイズで精密処理
                'crop_coord': 'auto'                          # 自動クロップ座標
            }
            
            logger.info(f"📡 POST {SADTALKER_API_URL}/generate")
            response = requests.post(
                f"{SADTALKER_API_URL}/generate", 
                files=files, 
                data=data, 
                timeout=180  # 3分タイムアウト（56秒処理）
            )
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                download_url = result.get('download_url')
                if download_url:
                    # 生成された動画をダウンロード
                    video_response = requests.get(f"{SADTALKER_API_URL}{download_url}")
                    if video_response.status_code == 200:
                        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_video_orig:
                            temp_video_orig.write(video_response.content)
                            temp_video_orig_path = temp_video_orig.name
                        
                        # 音声品質修復: 元の高品質音声で置き換え
                        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_video_fixed:
                            temp_video_fixed_path = temp_video_fixed.name
                        
                        # FFmpegでストリーム分離して高品質音声をマージ
                        import subprocess
                        ffmpeg_cmd = [
                            'ffmpeg', '-y',
                            '-i', temp_video_orig_path,  # SadTalker動画（映像のみ使用）
                            '-i', audio_path,            # 元の高品質音声
                            '-map', '0:v:0',             # 動画の映像ストリーム
                            '-map', '1:a:0',             # 元音声の音声ストリーム
                            '-c:v', 'copy',              # 映像は無変換
                            '-c:a', 'copy',              # 音声は無変換（品質保持）
                            '-shortest',                 # 短い方に合わせる
                            temp_video_fixed_path
                        ]
                        
                        try:
                            subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
                            logger.info(f"✅ 音声品質修復完了: 元音声品質を100%保持")
                            
                            # 一時ファイル削除
                            try:
                                os.unlink(temp_video_orig_path)
                            except:
                                pass
                            
                            return temp_video_fixed_path, f"SadTalker高品質動画生成完了（音声ノイズフリー） ({result.get('processing_time', 0):.1f}秒)"
                            
                        except subprocess.CalledProcessError as e:
                            logger.error(f"❌ 音声修復エラー: {e}")
                            # フォールバック: 元の動画を返す
                            return temp_video_orig_path, f"SadTalker動画生成完了（音声修復失敗） ({result.get('processing_time', 0):.1f}秒)"
            
            return None, f"SadTalkerエラー: {result.get('error', 'Unknown error')}"
        else:
            return None, f"SadTalker APIエラー: {response.status_code}"
    
    except Exception as e:
        logger.error(f"❌ SadTalker API呼び出しエラー: {str(e)}")
        return None, f"SadTalker接続エラー: {str(e)}"

def generate_complete_lipsync(text_input, media_file, ref_audio_file, enhancer="gfpgan", batch_size=8):
    """統合口パク動画生成"""
    try:
        logger.info("🎭 統合口パク動画生成開始")
        
        # 0. ファイル形式自動判定とルーティング
        if media_file:
            file_ext = os.path.splitext(media_file.lower())[1]
            is_image = file_ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
            
            if is_image:
                logger.info("🖼️ 画像ファイル検出 → SadTalker APIを使用")
                
                # 1. GPT-SoVITSで音声生成
                logger.info("🎤 音声生成中...")
                with open(ref_audio_file, 'rb') as f:
                    files = {"ref_audio": f}
                    data = {"ref_text": "サンプル", "target_text": text_input}
                    sovits_response = requests.post(f"{SOVITS_API_URL}/clone-voice-simple", 
                                                  files=files, data=data, timeout=60)
                
                if sovits_response.status_code != 200:
                    return None, f"音声生成エラー: {sovits_response.status_code}"
                
                # 音声を一時ファイルに保存
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
                    temp_audio.write(sovits_response.content)
                    temp_audio_path = temp_audio.name
                
                # 2. SadTalker APIで画像→動画生成
                logger.info("🎭 SadTalker高品質動画生成中（約56秒）...")
                result_video, status_msg = call_sadtalker_api(media_file, temp_audio_path)
                
                # 一時ファイル削除
                try:
                    os.unlink(temp_audio_path)
                except:
                    pass
                
                return result_video, status_msg
            
            else:
                logger.info("🎬 動画ファイル検出 → Wav2Lip APIを使用（高速処理）")
                video_file_path = media_file
                
                # 1. 音声生成（動画モード）
                logger.info("🎤 音声生成中...")
                
                with open(ref_audio_file, 'rb') as f:
                    files = {"ref_audio": f}
                    data = {"ref_text": "サンプル", "target_text": text_input}
                    
                    sovits_response = requests.post(f"{SOVITS_API_URL}/clone-voice-simple", 
                                                  files=files, data=data, timeout=60)
                
                if sovits_response.status_code != 200:
                    return None, f"音声生成エラー: {sovits_response.status_code}"
                
                # 音声を一時ファイルに保存
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
                    temp_audio.write(sovits_response.content)
                    temp_audio_path = temp_audio.name
                
                # 2. Wav2Lip口パク動画生成
                logger.info("🎬 Wav2Lip口パク動画生成中...")
                
                with open(video_file_path, 'rb') as vf, open(temp_audio_path, 'rb') as af:
                    files = {
                        "video_file": vf,
                        "audio_file": af
                    }
                    data = {
                        "enhancer": enhancer,
                        "batch_size": batch_size
                    }
                    
                    wav2lip_response = requests.post(f"{WAV2LIP_API_URL}/generate-lipsync",
                                                   files=files, data=data, timeout=120)
                
                # クリーンアップ
                os.unlink(temp_audio_path)
                
                if wav2lip_response.status_code == 200:
                    # 結果動画を一時ファイルに保存
                    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_result_video:
                        temp_result_video.write(wav2lip_response.content)
                        result_video_path = temp_result_video.name
                    
                    logger.info("✅ Wav2Lip動画処理完了!")
                    return result_video_path, "Wav2Lip高速口パク動画生成完了!"
                else:
                    return None, f"Wav2Lip生成エラー: {wav2lip_response.status_code}"
            
    except Exception as e:
        logger.error(f"統合処理エラー: {e}")
        # エラー時のクリーンアップ
        if 'temp_audio_path' in locals():
            try:
                os.unlink(temp_audio_path)
            except:
                pass
        if 'temp_video_path' in locals() and temp_video_path:
            try:
                os.unlink(temp_video_path)
            except:
                pass
        return None, f"エラー: {e}"

# Gradio インターフェース
def create_interface():
    with gr.Blocks(title="Super Wav2Lip") as interface:
        gr.HTML("<h1>🎭 Super Wav2Lip - AI音声クローン統合口パクシステム</h1>")
        
        with gr.Row():
            with gr.Column():
                text_input = gr.Textbox(
                    label="📝 生成したいテキスト",
                    placeholder="こんにちは！今日はいい天気ですね。",
                    lines=3
                )
                
                media_file = gr.File(
                    label="🎬 動画・画像ファイル", 
                    file_types=[".mp4", ".avi", ".mov", ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]
                )
                
                ref_audio_file = gr.File(
                    label="🎵 参照音声ファイル",
                    file_types=[".wav", ".mp3", ".m4a"]
                )
                
                with gr.Row():
                    enhancer = gr.Dropdown(
                        choices=["none", "gfpgan"],
                        value="none",
                        label="✨ 顔強化",
                        info="gfpgan: 実写顔用 | none: 無し"
                    )
                    
                    batch_size = gr.Slider(
                        minimum=1, maximum=16, value=8, step=1,
                        label="⚡ バッチサイズ"
                    )
                
                
                generate_btn = gr.Button("🚀 口パク動画生成開始", variant="primary")
                
            with gr.Column():
                status_output = gr.Textbox(label="📊 処理状況", lines=2)
                video_output = gr.Video(label="📺 生成された動画")
        
        generate_btn.click(
            fn=generate_complete_lipsync,
            inputs=[text_input, media_file, ref_audio_file, enhancer, batch_size],
            outputs=[video_output, status_output]
        )
    
    return interface

if __name__ == "__main__":
    interface = create_interface()
    interface.launch(server_name="0.0.0.0", server_port=7860)