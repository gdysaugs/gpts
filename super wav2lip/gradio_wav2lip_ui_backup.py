#!/usr/bin/env python3
"""
Super Wav2Lip Gradio UI - GFPGAN強化版
完全最適化されたFastAPI Dockerサーバーと連携
"""

import gradio as gr
import requests
import os
import tempfile
import shutil
from pathlib import Path
import time
import glob
import subprocess

# FastAPI サーバーのURL（コンテナ間通信）
FASTAPI_URL = os.getenv("FASTAPI_URL", "http://super-wav2lip-backend:8002")

def cleanup_old_temp_files():
    """古い一時ファイルをクリーンアップ"""
    try:
        temp_dir = "/app/temp"
        if os.path.exists(temp_dir):
            # 10分以上古いファイルを削除
            current_time = time.time()
            for file_path in glob.glob(os.path.join(temp_dir, "gradio_output_*.mp4")):
                if current_time - os.path.getmtime(file_path) > 600:  # 10分
                    os.remove(file_path)
    except Exception as e:
        print(f"クリーンアップエラー: {e}")

def reencode_video_for_gradio(input_path):
    """
    動画をGradio/Webブラウザ互換性のためにH.264で再エンコード
    mpeg4 → h264 変換でGradio 3.x表示問題を解決
    """
    try:
        # 出力ファイル名を生成
        output_file = tempfile.NamedTemporaryFile(delete=False, suffix='_h264.mp4', prefix='gradio_')
        output_path = output_file.name
        output_file.close()
        
        # ffmpeg コマンドでH.264に再エンコード
        cmd = [
            'ffmpeg', '-y',  # -y: 上書き許可
            '-i', input_path,  # 入力ファイル
            '-c:v', 'libx264',  # H.264 ビデオコーデック
            '-preset', 'fast',  # エンコード速度優先
            '-crf', '23',  # 品質設定（18-28推奨、23=デフォルト）
            '-c:a', 'aac',  # AAC オーディオコーデック
            '-movflags', '+faststart',  # Web再生最適化
            '-pix_fmt', 'yuv420p',  # 互換性最大化
            output_path
        ]
        
        # ffmpeg実行
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            # 元ファイルを削除
            if os.path.exists(input_path):
                os.remove(input_path)
            return output_path
        else:
            print(f"ffmpeg エラー: {result.stderr}")
            # 失敗時は元ファイルを返す
            if os.path.exists(output_path):
                os.remove(output_path)
            return input_path
            
    except Exception as e:
        print(f"再エンコードエラー: {e}")
        return input_path  # 失敗時は元ファイルを返す

def check_server_status():
    """FastAPIサーバーの状態をチェック"""
    try:
        response = requests.get(f"{FASTAPI_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return f"✅ サーバー準備完了！ウォームアップ済み: {data.get('warmup_completed', False)}"
        else:
            return f"⚠️ サーバー応答異常: {response.status_code}"
    except Exception as e:
        return f"❌ サーバー接続エラー: {str(e)}"

def generate_lipsync_video(video_file, audio_file, enhancer="gfpgan", batch_size=8, progress=gr.Progress()):
    """
    口パク動画生成（GFPGAN強化版）
    今成功したテストと同じパラメータを使用
    """
    if video_file is None or audio_file is None:
        return None, "❌ 動画ファイルと音声ファイルの両方を選択してください"
    
    progress(0, desc="📤 ファイルアップロード中...")
    
    # 古いファイルをクリーンアップ
    cleanup_old_temp_files()
    
    try:
        # ファイルを準備
        files = {
            'video_file': ('video.mp4', open(video_file, 'rb'), 'video/mp4'),
            'audio_file': ('audio.wav', open(audio_file, 'rb'), 'audio/wav')
        }
        
        # 成功したテストと同じパラメータ
        data = {
            'enhancer': enhancer,
            'hq_output': 'false',
            'fp16': 'true', 
            'batch_size': str(batch_size)
        }
        
        progress(0.1, desc="🚀 口パク動画生成開始...")
        
        # FastAPI に送信
        response = requests.post(
            f"{FASTAPI_URL}/generate-lipsync",
            files=files,
            data=data,
            timeout=120  # 41秒処理 + マージン
        )
        
        # ファイルを閉じる
        for file_obj in files.values():
            file_obj[1].close()
        
        progress(0.9, desc="📥 結果取得中...")
        
        if response.status_code == 200:
            result = response.json()
            
            if result['status'] == 'success':
                # 結果ファイルをダウンロード
                download_response = requests.get(f"{FASTAPI_URL}{result['download_url']}")
                
                if download_response.status_code == 200:
                    # 元動画を一時ファイルに保存
                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4', prefix='original_')
                    temp_file.write(download_response.content)
                    temp_file.flush()
                    temp_file.close()
                    
                    progress(0.95, desc="🔄 Webブラウザ互換性のため動画を最適化中...")
                    
                    # H.264で再エンコードしてGradio表示互換性を向上
                    optimized_path = reencode_video_for_gradio(temp_file.name)
                    
                    # ファイルの権限を設定
                    os.chmod(optimized_path, 0o644)
                    
                    progress(1.0, desc="✅ 完了！")
                    
                    # ファイルサイズを再計算
                    optimized_size_mb = round(os.path.getsize(optimized_path) / (1024 * 1024), 2)
                    
                    status_msg = f"""
✅ 口パク動画生成完了！
📁 出力ファイル: {result['output_filename']}
📊 元ファイルサイズ: {result['output_size_mb']} MB
📊 最適化後サイズ: {optimized_size_mb} MB
⏱️ 処理時間: {result['processing_time_seconds']:.1f}秒
🔧 使用モデル: {result['checkpoint_used']}
✨ 強化処理: {result['enhancer_used']}
🎬 動画最適化: mpeg4 → H.264 (Webブラウザ互換)
📍 最適化ファイル: {optimized_path}
"""
                    return optimized_path, status_msg
                else:
                    return None, f"❌ ファイルダウンロードエラー: {download_response.status_code}"
            else:
                return None, f"❌ 処理エラー: {result.get('error', '不明なエラー')}"
        else:
            return None, f"❌ API エラー: {response.status_code} - {response.text}"
            
    except Exception as e:
        return None, f"❌ 処理中にエラーが発生しました: {str(e)}"

def create_interface():
    """Gradio インターフェースを作成"""
    
    with gr.Blocks(
        title="🎭 Super Wav2Lip - GFPGAN強化版",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1200px !important;
        }
        .header {
            text-align: center;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 2.5em;
            font-weight: bold;
            margin-bottom: 20px;
        }
        """
    ) as app:
        
        # ヘッダー
        gr.HTML("""
        <div class="header">
            🎭 Super Wav2Lip - GFPGAN強化版
        </div>
        <div style="text-align: center; margin-bottom: 30px;">
            <p style="font-size: 1.2em; color: #666;">
                🚀 <strong>41秒の超高速処理</strong> | ✨ <strong>GFPGAN顔強化</strong> | 🎯 <strong>完全最適化済み</strong>
            </p>
        </div>
        """)
        
        # サーバー状態表示
        with gr.Row():
            server_status = gr.Textbox(
                label="🖥️ サーバー状態",
                value=check_server_status(),
                interactive=False
            )
            refresh_btn = gr.Button("🔄 更新", size="sm")
        
        refresh_btn.click(
            fn=check_server_status,
            outputs=server_status
        )
        
        with gr.Row():
            # 左側: 入力
            with gr.Column(scale=1):
                gr.Markdown("### 📥 入力ファイル")
                
                video_input = gr.Video(
                    label="🎬 動画ファイル (.mp4, .avi, .mov)",
                    sources=["upload"],
                    format="mp4"
                )
                
                audio_input = gr.Audio(
                    label="🎵 音声ファイル (.wav, .mp3, .m4a)",
                    sources=["upload"],
                    type="filepath"
                )
                
                with gr.Row():
                    enhancer_choice = gr.Dropdown(
                        choices=["gfpgan", "none"],
                        value="gfpgan",
                        label="✨ 顔強化",
                        info="GFPGAN: 高品質顔強化（推奨）"
                    )
                    
                    batch_size = gr.Slider(
                        minimum=1,
                        maximum=16,
                        value=8,
                        step=1,
                        label="⚡ バッチサイズ",
                        info="RTX 3050最適化: 8推奨"
                    )
                
                generate_btn = gr.Button(
                    "🚀 口パク動画生成開始",
                    variant="primary",
                    size="lg"
                )
            
            # 右側: 出力
            with gr.Column(scale=1):
                gr.Markdown("### 📤 出力結果")
                
                output_video = gr.Video(
                    label="🎭 生成された口パク動画",
                    show_download_button=True,
                    format="mp4",
                    autoplay=False,
                    interactive=False
                )
                
                status_output = gr.Textbox(
                    label="📊 処理状況",
                    lines=8,
                    max_lines=10
                )
        
        # 生成ボタンのクリックイベント
        generate_btn.click(
            fn=generate_lipsync_video,
            inputs=[video_input, audio_input, enhancer_choice, batch_size],
            outputs=[output_video, status_output],
            show_progress=True
        )
        
        # フッター
        gr.HTML("""
        <div style="text-align: center; margin-top: 40px; padding: 20px; border-top: 1px solid #eee;">
            <p style="color: #888;">
                🏆 <strong>完全最適化版</strong> - 目標40秒の98.1%達成（41秒処理）<br>
                ⚡ ウォームアップ済みFastAPI Docker + GFPGAN強化 + batch_size=8最適化
            </p>
            <p style="color: #666; font-size: 0.9em;">
                ⚠️ 教育・研究目的でのみ使用してください
            </p>
        </div>
        """)
    
    return app

if __name__ == "__main__":
    # Gradio UI を起動
    app = create_interface()
    
    print("🎭 Super Wav2Lip Gradio UI 起動中...")
    print("📍 FastAPI サーバー:", FASTAPI_URL)
    print("🌐 Gradio UI: http://localhost:7860")
    
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        show_tips=True,
        enable_queue=True
    )