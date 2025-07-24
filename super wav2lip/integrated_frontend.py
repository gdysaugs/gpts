#!/usr/bin/env python3
"""
統合Frontend - テキスト→音声→口パク動画システム
Gradio Web UI + API オーケストレーション

ワークフロー: テキスト入力 → SoVITS音声生成 → Wav2Lip口パク動画生成
"""

import os
import time
import logging
import asyncio
import tempfile
import requests
import gradio as gr
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import uuid
import shutil
import librosa
import soundfile as sf
import numpy as np

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API エンドポイント設定
SOVITS_API_URL = os.getenv("SOVITS_API", "http://sovits:8000")
WAV2LIP_API_URL = os.getenv("WAV2LIP_API", "http://wav2lip:8002") 
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")

# 共有ディレクトリ
SHARED_DIR = Path("/shared")
SHARED_INPUT = SHARED_DIR / "input"
SHARED_OUTPUT = SHARED_DIR / "output" 
SHARED_TEMP = SHARED_DIR / "temp"

# ローカルディレクトリ（フォールバック）
LOCAL_INPUT = Path("/app/input")
LOCAL_OUTPUT = Path("/app/output")
LOCAL_TEMP = Path("/app/temp")

# ディレクトリ作成
for dir_path in [SHARED_INPUT, SHARED_OUTPUT, SHARED_TEMP, LOCAL_INPUT, LOCAL_OUTPUT, LOCAL_TEMP]:
    dir_path.mkdir(parents=True, exist_ok=True)

class IntegratedWorkflow:
    """統合ワークフロー管理クラス"""
    
    def __init__(self):
        self.session_id = None
        self.intermediate_files = []
        
    def new_session(self) -> str:
        """新しいセッション開始"""
        self.session_id = str(uuid.uuid4())[:8]
        self.intermediate_files = []
        logger.info(f"🆕 新セッション開始: {self.session_id}")
        return self.session_id
    
    def cleanup_session(self):
        """セッション清理"""
        for file_path in self.intermediate_files:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except Exception as e:
                logger.warning(f"⚠️ ファイル削除失敗: {file_path}, {e}")
        
        self.intermediate_files = []
        logger.info(f"🧹 セッション清理完了: {self.session_id}")

# グローバルワークフローインスタンス
workflow = IntegratedWorkflow()

def check_api_health() -> Dict[str, bool]:
    """API サービス健康状態チェック"""
    health_status = {
        "sovits": False,
        "wav2lip": False
    }
    
    try:
        # SoVITS API チェック
        response = requests.get(f"{SOVITS_API_URL}/health", timeout=5)
        logger.info(f"🔍 SoVITS Response: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            logger.info(f"🔍 SoVITS Data: {data}")
            health_status["sovits"] = data.get("status") == "healthy"
        else:
            health_status["sovits"] = False
    except Exception as e:
        logger.warning(f"⚠️ SoVITS API接続失敗: {e}")
        health_status["sovits"] = False
    
    try:
        # Wav2Lip API チェック
        response = requests.get(f"{WAV2LIP_API_URL}/health", timeout=5)
        health_status["wav2lip"] = response.status_code == 200
    except Exception as e:
        logger.warning(f"⚠️ Wav2Lip API接続失敗: {e}")
    
    return health_status

def trim_audio_silence(audio_path: str, text: str) -> str:
    """
    音声ファイルの無音部分を適切にトリミングする
    - 開始時無音は削除（ただし文字が「。」「...」で始まる場合は保持）
    - 文中の句読点による自然な間は保持
    """
    try:
        # 音声読み込み
        audio, sr = librosa.load(audio_path, sr=None)
        
        # 音声の振幅レベル分析
        audio_abs = np.abs(audio)
        
        # 無音判定閾値（最大音量の2%以下を無音とする）
        silence_threshold = np.max(audio_abs) * 0.02
        
        # 音声が存在する部分を検出
        non_silent_indices = np.where(audio_abs > silence_threshold)[0]
        
        if len(non_silent_indices) == 0:
            logger.warning("⚠️ 音声が完全に無音です")
            return audio_path
        
        # 開始時無音削除の判定
        should_keep_start_silence = any(text.startswith(prefix) for prefix in ['。', '、', '...', '…', '　'])
        
        if should_keep_start_silence:
            logger.info(f"📝 文字が句読点で始まるため開始時無音を保持: '{text[:10]}'")
            start_idx = 0
        else:
            # 開始時無音を削除
            start_idx = non_silent_indices[0]
            # 少し余裕を持たせる（50ms程度）
            buffer_samples = int(sr * 0.05)  # 50ms
            start_idx = max(0, start_idx - buffer_samples)
            
            if start_idx > 0:
                removed_time = start_idx / sr
                logger.info(f"✂️ 開始時無音削除: {removed_time:.2f}秒")
        
        # 終了時は少し余裕を持たせてトリミング
        end_idx = non_silent_indices[-1]
        buffer_samples = int(sr * 0.1)  # 100ms
        end_idx = min(len(audio), end_idx + buffer_samples)
        
        # 音声をトリミング
        trimmed_audio = audio[start_idx:end_idx]
        
        # 修正されたファイルを保存
        trimmed_path = audio_path.replace('.wav', '_trimmed.wav')
        sf.write(trimmed_path, trimmed_audio, sr)
        
        original_duration = len(audio) / sr
        trimmed_duration = len(trimmed_audio) / sr
        
        logger.info(f"🎵 音声トリミング完了: {original_duration:.2f}s → {trimmed_duration:.2f}s")
        
        return trimmed_path
        
    except Exception as e:
        logger.error(f"❌ 音声トリミングエラー: {e}")
        return audio_path

def generate_voice_from_text(
    text: str,
    ref_audio_path: str,
    ref_text: str = "おはようございます",
    temperature: float = 1.0
) -> Tuple[bool, str, Optional[str]]:
    """テキストから音声生成（SoVITS API）"""
    
    try:
        logger.info(f"🎤 音声生成開始: '{text[:30]}...'")
        start_time = time.time()
        
        # 参照音声ファイルを送信するためのデータ準備
        files = {
            "ref_audio": open(ref_audio_path, "rb")
        }
        
        data = {
            "ref_text": ref_text,
            "target_text": text,
            "temperature": temperature
        }
        
        try:
            logger.info(f"🌐 API URL: {SOVITS_API_URL}/clone-voice-simple")
            logger.info(f"📊 Request data: {data}")
            logger.info(f"📁 Files: {list(files.keys())}")
            response = requests.post(
                f"{SOVITS_API_URL}/clone-voice-simple",
                files=files,
                data=data,
                timeout=60
            )
            logger.info(f"📋 Response status: {response.status_code}")
            logger.info(f"📄 Response headers: {dict(response.headers)}")
        finally:
            # ファイルハンドルを確実に閉じる
            files["ref_audio"].close()
        
        if response.status_code == 200:
            # 音声ファイル保存
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            safe_text = text[:20].replace("/", "_").replace("\\", "_").replace(" ", "_")
            audio_filename = f"generated_audio_{workflow.session_id}_{timestamp}_{safe_text}.wav"
            
            # 複数保存場所試行
            audio_paths = [
                SHARED_OUTPUT / audio_filename,
                LOCAL_OUTPUT / audio_filename
            ]
            
            saved_path = None
            for audio_path in audio_paths:
                try:
                    with open(audio_path, "wb") as f:
                        f.write(response.content)
                    saved_path = str(audio_path)
                    workflow.intermediate_files.append(saved_path)
                    break
                except Exception as e:
                    logger.warning(f"⚠️ 保存失敗: {audio_path}, {e}")
            
            if saved_path:
                # 音声の無音部分をトリミング
                logger.info("✂️ 音声トリミング処理開始...")
                trimmed_path = trim_audio_silence(saved_path, text)
                
                gen_time = time.time() - start_time
                logger.info(f"✅ 音声生成成功: {audio_filename} ({gen_time:.1f}秒)")
                
                # トリミングされた音声を使用
                if trimmed_path != saved_path:
                    workflow.intermediate_files.append(trimmed_path)
                    return True, f"音声生成＋トリミング成功 ({gen_time:.1f}秒)", trimmed_path
                else:
                    return True, f"音声生成成功 ({gen_time:.1f}秒)", saved_path
            else:
                return False, "音声ファイル保存失敗", None
        else:
            error_msg = f"SoVITS API エラー: {response.status_code}"
            logger.error(f"❌ {error_msg}")
            logger.error(f"📄 Response text: {response.text}")
            logger.error(f"📋 Response headers: {dict(response.headers)}")
            return False, error_msg, None
    
    except Exception as e:
        error_msg = f"音声生成エラー: {e}"
        logger.error(f"🚨 {error_msg}")
        logger.error(f"🔍 Exception type: {type(e)}")
        import traceback
        logger.error(f"📍 Traceback: {traceback.format_exc()}")
        return False, error_msg, None

def generate_lipsync_video(
    video_path: str,
    audio_path: str,
    enhancer: str = "gfpgan",
    batch_size: int = 8
) -> Tuple[bool, str, Optional[str]]:
    """口パク動画生成（Wav2Lip API）"""
    
    try:
        logger.info(f"🎭 口パク動画生成開始...")
        start_time = time.time()
        
        # ファイル準備
        files = {
            "video_file": open(video_path, "rb"),
            "audio_file": open(audio_path, "rb")
        }
        
        data = {
            "enhancer": enhancer,
            "batch_size": batch_size,
            "hq_output": False,
            "fp16": True,
            "face_mask": False
        }
        
        try:
            response = requests.post(
                f"{WAV2LIP_API_URL}/generate-lipsync",
                files=files,
                data=data,
                timeout=300  # 5分タイムアウト
            )
        finally:
            # ファイルハンドル閉じる
            for file_obj in files.values():
                file_obj.close()
        
        if response.status_code == 200:
            result = response.json()
            
            if result.get("success"):
                gen_time = time.time() - start_time
                output_filename = result.get("output_filename")
                download_url = result.get("download_url")
                
                # 結果ファイルパス構築
                result_paths = [
                    SHARED_OUTPUT / output_filename,
                    LOCAL_OUTPUT / output_filename
                ]
                
                result_path = None
                for path in result_paths:
                    if path.exists():
                        result_path = str(path)
                        break
                
                if result_path:
                    logger.info(f"✅ 口パク動画生成成功: {output_filename} ({gen_time:.1f}秒)")
                    return True, f"口パク動画生成成功 ({gen_time:.1f}秒)", result_path
                else:
                    return False, "生成ファイルが見つかりません", None
            else:
                error_msg = result.get("message", "不明なエラー")
                logger.error(f"❌ 口パク生成失敗: {error_msg}")
                return False, f"口パク生成失敗: {error_msg}", None
        else:
            error_msg = f"Wav2Lip API エラー: {response.status_code}"
            logger.error(error_msg)
            return False, error_msg, None
    
    except Exception as e:
        error_msg = f"口パク動画生成エラー: {e}"
        logger.error(error_msg)
        return False, error_msg, None

def integrated_generation(
    text: str,
    video_file,
    ref_audio_file,
    ref_text: str = "おはようございます",
    enhancer: str = "gfpgan",
    temperature: float = 1.0,
    batch_size: int = 8,
    progress=gr.Progress()
) -> Tuple[Optional[str], str]:
    """統合生成処理（メイン関数）"""
    
    # 新セッション開始
    session_id = workflow.new_session()
    
    try:
        # バリデーション
        if not text.strip():
            return None, "❌ テキストを入力してください"
        
        if video_file is None:
            return None, "❌ 動画ファイルをアップロードしてください"
        
        if ref_audio_file is None:
            return None, "❌ 参照音声ファイルをアップロードしてください"
        
        # API健康状態チェック
        progress(0.1, "🔍 API サービス確認中...")
        health = check_api_health()
        
        if not health["sovits"]:
            return None, "❌ SoVITS APIが利用できません"
        
        if not health["wav2lip"]:
            return None, "❌ Wav2Lip APIが利用できません"
        
        # ファイル保存
        progress(0.2, "📁 ファイル準備中...")
        
        # 一時ファイルパス生成
        temp_dir = SHARED_TEMP / session_id
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        video_path = temp_dir / f"input_video_{session_id}.mp4"
        ref_audio_path = temp_dir / f"ref_audio_{session_id}.wav"
        
        # ファイル保存
        shutil.copy2(video_file.name, video_path)
        shutil.copy2(ref_audio_file.name, ref_audio_path)
        
        workflow.intermediate_files.extend([str(video_path), str(ref_audio_path)])
        
        # Step 1: 音声生成
        progress(0.3, "🎤 音声生成中...")
        voice_success, voice_message, generated_audio_path = generate_voice_from_text(
            text=text,
            ref_audio_path=str(ref_audio_path),
            ref_text=ref_text,
            temperature=temperature
        )
        
        if not voice_success:
            return None, f"❌ 音声生成失敗: {voice_message}"
        
        if not generated_audio_path or not os.path.exists(generated_audio_path):
            return None, "❌ 生成音声ファイルが見つかりません"
        
        # Step 2: 口パク動画生成
        progress(0.6, "🎭 口パク動画生成中...")
        lipsync_success, lipsync_message, final_video_path = generate_lipsync_video(
            video_path=str(video_path),
            audio_path=generated_audio_path,
            enhancer=enhancer,
            batch_size=batch_size
        )
        
        if not lipsync_success:
            return None, f"❌ 口パク動画生成失敗: {lipsync_message}"
        
        if not final_video_path or not os.path.exists(final_video_path):
            return None, "❌ 最終動画ファイルが見つかりません"
        
        progress(1.0, "✅ 完了!")
        
        # 成功メッセージ
        success_message = f"""
✅ **生成完了！**

📝 **入力テキスト**: {text[:50]}{'...' if len(text) > 50 else ''}
🎤 **音声生成**: {voice_message}
🎭 **口パク動画**: {lipsync_message}
🆔 **セッションID**: {session_id}

動画をダウンロードして確認してください。
        """
        
        return final_video_path, success_message
    
    except Exception as e:
        error_msg = f"❌ 統合処理エラー: {e}"
        logger.error(error_msg)
        return None, error_msg
    
    finally:
        # セッション清理（オプション - ユーザーがダウンロード後）
        # workflow.cleanup_session()
        pass

# Gradio インターフェース構築
def create_interface():
    """Gradio UI作成"""
    
    with gr.Blocks(
        title="統合口パクシステム",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1200px !important;
        }
        .status-healthy { color: green; font-weight: bold; }
        .status-unhealthy { color: red; font-weight: bold; }
        """
    ) as interface:
        
        gr.Markdown("""
        # 🎭 統合口パクシステム
        
        **テキストを入力するだけで、指定した声で口パクする動画を生成**
        
        **フロー**: テキスト入力 → SoVITS音声生成 → Wav2Lip口パク動画生成
        """)
        
        # API ステータス表示
        with gr.Row():
            api_status = gr.HTML()
            
            def update_status():
                health = check_api_health()
                sovits_status = "🟢 正常" if health["sovits"] else "🔴 異常"
                wav2lip_status = "🟢 正常" if health["wav2lip"] else "🔴 異常"
                
                return f"""
                <div style="padding: 10px; border: 1px solid #ddd; border-radius: 5px;">
                    <h4>📡 APIサービス状態</h4>
                    <p><strong>SoVITS API</strong>: {sovits_status}</p>
                    <p><strong>Wav2Lip API</strong>: {wav2lip_status}</p>
                </div>
                """
            
            # 初期ステータス表示
            api_status.value = update_status()
        
        with gr.Row():
            with gr.Column(scale=1):
                # 入力セクション
                gr.Markdown("## 📝 入力")
                
                text_input = gr.Textbox(
                    label="生成したいテキスト",
                    placeholder="例: こんにちは！今日はいい天気ですね。",
                    lines=3,
                    max_lines=5
                )
                
                video_input = gr.File(
                    label="動画ファイル（口パクさせたい人物）",
                    file_types=[".mp4", ".avi", ".mov"],
                    type="filepath"
                )
                
                ref_audio_input = gr.File(
                    label="参照音声ファイル（クローンしたい声）",
                    file_types=[".wav", ".mp3", ".m4a"],
                    type="filepath"
                )
                
                ref_text_input = gr.Textbox(
                    label="参照音声のテキスト",
                    value="おはようございます",
                    placeholder="参照音声で話している内容"
                )
            
            with gr.Column(scale=1):
                # 設定セクション
                gr.Markdown("## ⚙️ 詳細設定")
                
                enhancer_select = gr.Dropdown(
                    choices=["none", "gfpgan", "gpen", "codeformer"],
                    value="gfpgan",
                    label="顔強化モード",
                    info="gfpgan: 高品質（推奨）, none: 高速"
                )
                
                temperature_slider = gr.Slider(
                    minimum=0.1,
                    maximum=2.0,
                    value=1.0,
                    step=0.1,
                    label="音声感情レベル",
                    info="1.0: 標準, 1.5: 感情豊か, 0.5: 落ち着き"
                )
                
                batch_size_select = gr.Dropdown(
                    choices=[4, 8, 16],
                    value=8,
                    label="バッチサイズ",
                    info="8: RTX 3050最適化（推奨）"
                )
        
        # 実行ボタン
        with gr.Row():
            generate_btn = gr.Button(
                "🚀 口パク動画生成開始",
                variant="primary",
                size="lg"
            )
            
            status_refresh_btn = gr.Button("🔄 API状態更新")
        
        # 結果セクション
        with gr.Row():
            with gr.Column():
                status_output = gr.Markdown()
                
                video_output = gr.Video(
                    label="生成された口パク動画",
                    visible=False
                )
                
                download_link = gr.File(
                    label="ダウンロード",
                    visible=False
                )
        
        # イベントハンドラー
        generate_btn.click(
            fn=integrated_generation,
            inputs=[
                text_input,
                video_input,
                ref_audio_input,
                ref_text_input,
                enhancer_select,
                temperature_slider,
                batch_size_select
            ],
            outputs=[video_output, status_output],
            show_progress=True
        ).then(
            fn=lambda video: (gr.update(visible=bool(video)), gr.update(visible=bool(video), value=video)),
            inputs=[video_output],
            outputs=[video_output, download_link]
        )
        
        status_refresh_btn.click(
            fn=update_status,
            outputs=[api_status]
        )
        
        # サンプル例
        gr.Markdown("""
        ## 💡 使用例
        
        1. **テキスト**: "こんにちは！今日は素晴らしい天気ですね。"
        2. **動画**: 口パクさせたい人物の動画（5-30秒推奨）
        3. **参照音声**: クローンしたい声の音声ファイル（5秒程度推奨）
        4. **参照テキスト**: 参照音声で話している内容
        
        **処理時間**: 通常50秒程度（SoVITS 3-5秒 + Wav2Lip 41秒）
        """)
    
    return interface

# アプリケーション起動
if __name__ == "__main__":
    logger.info("🚀 統合Frontend起動中...")
    
    # 初期API状態確認
    health = check_api_health()
    logger.info(f"📡 SoVITS API: {'🟢' if health['sovits'] else '🔴'}")
    logger.info(f"📡 Wav2Lip API: {'🟢' if health['wav2lip'] else '🔴'}")
    
    # Gradio インターフェース作成・起動
    interface = create_interface()
    
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=False,
        show_error=True,
        quiet=False
    )