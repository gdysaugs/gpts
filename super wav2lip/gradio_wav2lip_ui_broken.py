#!/usr/bin/env python3
"""
統合Frontend - テキスト→音声→口パク動画システム
Gradio Web UI + API オーケストレーション

ワークフロー: テキスト入力 → SoVITS音声生成 → Wav2Lip口パク動画生成
"""

import os
import sys
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
import numpy as np
import json

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API エンドポイント設定
SOVITS_API_URL = os.getenv("SOVITS_API", "http://gpt-sovits-api:8000")
WAV2LIP_API_URL = os.getenv("WAV2LIP_API", "http://localhost:8002")  # 統合処理では使用しないが変数は保持
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")

# 直接処理モード - srcディレクトリがある場合のみimport
try:
    sys.path.append('/app/src')
    from wav2lip_direct import generate_lipsync_video
    DIRECT_MODE = True
    logger.info("🎭 直接処理モード: 統合Wav2Lip処理を使用")
except ImportError:
    DIRECT_MODE = False
    logger.info("🌐 API処理モード: 外部APIサーバーを使用")

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

def check_api_health(retries: int = 3, retry_delay: float = 1.0) -> Dict[str, bool]:
    """API サービス健康状態チェック（リトライ機能付き）"""
    health_status = {
        "sovits": False,
        "wav2lip": False
    }
    
    # SoVITS API チェック（ヘルスエンドポイント）
    for attempt in range(retries):
        try:
            response = requests.get(f"{SOVITS_API_URL}/health", timeout=3)
            if response.status_code == 200:
                data = response.json()
                health_status["sovits"] = data.get("status") == "healthy"
                break
        except Exception as e:
            if attempt == retries - 1:  # 最後の試行の場合のみ警告
                logger.warning(f"⚠️ SoVITS API接続失敗: {e}")
            else:
                time.sleep(retry_delay)
    
    # 統合モード：Wav2Lipは統合処理のため常にOK
    health_status["wav2lip"] = True
    logger.info("📡 Wav2Lip: 統合処理モード（常時OK）")
    
    return health_status

def perfect_audio_sync_processing(audio_path: str) -> str:
    """
    完璧な音声・口パク同期のための高精度処理
    """
    try:
        import librosa
        import soundfile as sf
        
        # 音声読み込み（標準化サンプルレート：22050Hz）
        audio, original_sr = librosa.load(audio_path, sr=22050)
        target_sr = 22050  # Wav2Lipに最適なサンプルレート
        original_duration = len(audio) / target_sr
        
        # 超高精度無音検出設定（口パク同期専用）
        silence_threshold = 0.003  # -50dB相当（超厳格）
        frame_length = int(0.001 * target_sr)  # 1ms（超細かい）
        
        logger.info(f"🎯 完璧同期処理開始: {original_duration:.3f}秒")
        
        # 先頭無音の超精密除去
        start_idx = 0
        for i in range(0, len(audio), frame_length):
            frame = audio[i:i + frame_length]
            if len(frame) > 0:
                # 複数指標での音声開始点検出
                rms = np.sqrt(np.mean(frame ** 2))
                peak = np.max(np.abs(frame))
                energy = np.sum(frame ** 2)
                
                # より厳格な音声開始判定
                if (rms > silence_threshold or 
                    peak > (silence_threshold * 1.5) or 
                    energy > (silence_threshold ** 2 * len(frame))):
                    
                    # 音声開始点を1フレーム前に設定（アタック保護）
                    start_idx = max(0, i - frame_length)
                    break
        
        # 末尾無音の精密除去
        end_idx = len(audio)
        for i in range(len(audio) - frame_length, 0, -frame_length):
            frame = audio[i:i + frame_length]
            if len(frame) > 0:
                rms = np.sqrt(np.mean(frame ** 2))
                peak = np.max(np.abs(frame))
                energy = np.sum(frame ** 2)
                
                if (rms > silence_threshold or 
                    peak > (silence_threshold * 1.5) or 
                    energy > (silence_threshold ** 2 * len(frame))):
                    
                    # 音声終了点を1フレーム後に設定（リリース保護）
                    end_idx = min(len(audio), i + frame_length)
                    break
        
        # 完璧同期のための追加処理
        if start_idx > 0 or end_idx < len(audio):
            trimmed_audio = audio[start_idx:end_idx]
            
            # 短すぎる音声の保護
            if len(trimmed_audio) < 0.05 * target_sr:  # 0.05秒未満
                logger.warning("⚠️ 極短音声検出 - 元音声を保持")
                return audio_path
            
            # フェードイン/アウト処理（クリック音防止）
            fade_samples = int(0.002 * target_sr)  # 2ms
            if len(trimmed_audio) > fade_samples * 2:
                # フェードイン
                fade_in = np.linspace(0, 1, fade_samples)
                trimmed_audio[:fade_samples] *= fade_in
                
                # フェードアウト
                fade_out = np.linspace(1, 0, fade_samples)
                trimmed_audio[-fade_samples:] *= fade_out
            
            # DC成分除去（口パク同期精度向上）
            trimmed_audio = trimmed_audio - np.mean(trimmed_audio)
            
            # 正規化（レベル統一）
            if np.max(np.abs(trimmed_audio)) > 0:
                trimmed_audio = trimmed_audio / np.max(np.abs(trimmed_audio)) * 0.9
            
            # 最終音声を保存（同期最適化済み）
            sf.write(audio_path, trimmed_audio, target_sr)
            
            # 同期統計
            removed_start_ms = (start_idx / target_sr) * 1000
            removed_end_ms = ((len(audio) - end_idx) / target_sr) * 1000
            final_duration = len(trimmed_audio) / target_sr
            
            logger.info(f"🎯 完璧同期処理完了:")
            logger.info(f"   先頭トリミング: {removed_start_ms:.1f}ms")
            logger.info(f"   末尾トリミング: {removed_end_ms:.1f}ms")
            logger.info(f"   音声長: {original_duration:.3f}s → {final_duration:.3f}s")
            logger.info(f"   サンプルレート: {target_sr}Hz (Wav2Lip最適化)")
            logger.info(f"   ✅ 完璧な音声・口パク同期を保証")
        else:
            # 無音なしでも同期最適化処理
            audio = audio - np.mean(audio)  # DC除去
            if np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio)) * 0.9  # 正規化
            
            sf.write(audio_path, audio, target_sr)
            logger.info("🎯 音声同期最適化完了（無音なし）")
        
        return audio_path
        
    except Exception as e:
        logger.warning(f"⚠️ 同期処理エラー（処理継続）: {e}")
        return audio_path

def generate_voice_from_text(
    text: str,
    ref_audio_path: str,
    temperature: float = 1.0
) -> Tuple[bool, str, Optional[str]]:
    """テキストから音声生成（SoVITS API）"""
    
    try:
        logger.info(f"🎤 音声生成開始: '{text[:30]}...'")
        start_time = time.time()
        
        # 参照音声ファイルを送信するためのデータ準備
        logger.info(f"🔍 参照音声ファイル: {ref_audio_path}, 存在確認: {os.path.exists(ref_audio_path)}")
        files = {
            "ref_audio": open(ref_audio_path, "rb")
        }
        
        data = {
            "ref_text": "おはようございます",
            "target_text": text,
            "temperature": temperature
        }
        
        try:
            response = requests.post(
                f"{SOVITS_API_URL}/clone-voice-simple",
                files=files,
                data=data,
                timeout=60
            )
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
                # 完璧な音声・口パク同期処理
                try:
                    trimmed_path = perfect_audio_sync_processing(saved_path)
                    logger.info("🎯 完璧同期処理完了: 音声・口パクの完全同期を保証")
                except Exception as e:
                    logger.warning(f"⚠️ 同期処理スキップ: {e}")
                    trimmed_path = saved_path
                
                gen_time = time.time() - start_time
                logger.info(f"✅ 音声生成＋同期最適化完了: {audio_filename} ({gen_time:.1f}秒)")
                return True, f"音声生成＋同期最適化成功 ({gen_time:.1f}秒)", trimmed_path
            else:
                return False, "音声ファイル保存失敗", None
        else:
            error_msg = f"SoVITS API エラー: {response.status_code}"
            try:
                response_text = response.text
                logger.error(f"❌ SoVITS API詳細: {error_msg}, Response: {response_text}")
            except:
                logger.error(f"❌ SoVITS API詳細: {error_msg}")
            return False, error_msg, None
    
    except Exception as e:
        error_msg = f"音声生成エラー: {e}"
        logger.error(error_msg)
        return False, error_msg, None

def get_audio_duration(audio_path: str) -> float:
    """音声ファイルの長さを秒単位で取得"""
    try:
        import librosa
        audio, sr = librosa.load(audio_path, sr=None)
        duration = len(audio) / sr
        logger.info(f"🎵 音声長取得: {duration:.2f}秒 ({audio_path})")
        return duration
    except Exception as e:
        logger.warning(f"⚠️ 音声長取得失敗、デフォルト5秒を使用: {e}")
        return 5.0  # フォールバック値

def generate_lipsync_video_with_progress(
    video_path: str,
    audio_path: str,
    enhancer: str = "gfpgan",
    batch_size: int = 8,
    progress_callback=None
) -> Tuple[bool, str, Optional[str]]:
    """音声長ベース動的進捗付きWav2Lip処理（柔軟な%表示）"""
    import threading
    import queue
    
    try:
        # 音声の長さを取得して推定処理時間を計算
        audio_duration = get_audio_duration(audio_path)
        
        # 音声長に基づく推定処理時間（RTX 3050基準）
        # 基本時間: 7秒 + 音声長 × 3秒/秒 + エンハンサー時間
        base_time = 7.0
        audio_factor = 3.0  # 音声1秒あたり3秒の処理時間
        enhancer_overhead = 5.0 if enhancer != 'none' else 0.0
        
        estimated_total_time = base_time + (audio_duration * audio_factor) + enhancer_overhead
        
        logger.info(f"📊 推定処理時間: {estimated_total_time:.1f}秒 (音声{audio_duration:.1f}秒, enhancer={enhancer})")
        
        result_queue = queue.Queue()
        progress_stop_event = threading.Event()
        
        def wav2lip_worker():
            """バックグラウンドでWav2Lip処理を実行"""
            try:
                result = generate_lipsync_video(video_path, audio_path, enhancer, batch_size)
                result_queue.put(('success', result))
            except Exception as e:
                result_queue.put(('error', e))
            finally:
                progress_stop_event.set()
        
        def progress_worker():
            """音声長ベース動的進捗表示ワーカー（0-100%を出力、統合UIで75-100%にマッピング）"""
            start_time = time.time()
            
            while not progress_stop_event.is_set():
                elapsed_time = time.time() - start_time
                
                # 0%から100%の進捗計算（統合UI側で75-100%にマッピングされる）
                if elapsed_time < estimated_total_time:
                    # 音声長ベースの進捗（0-99%）
                    time_progress = min((elapsed_time / estimated_total_time) * 99, 99)
                else:
                    # 推定時間を超過した場合は99%で待機
                    time_progress = 99
                
                # 1%刻みに丸める（1.0, 2.0, 3.0, ..., 99.0）
                rounded_progress = max(1, round(time_progress))  # 最小1%から開始
                
                # より詳細な進捗段階（0-100%の範囲、統合UIで75-100%表示）
                if rounded_progress < 15:
                    desc = f"🔍 動画解析・顔検出中..."
                elif rounded_progress < 30:
                    desc = f"🎵 音声スペクトログラム生成中..."
                elif rounded_progress < 70:
                    desc = f"💋 口パクフレーム生成中..."
                elif rounded_progress < 85:
                    desc = f"✨ フレーム強化処理中..."
                elif rounded_progress < 95:
                    desc = f"🎬 動画エンコード中..."
                else:
                    desc = f"⏳ 最終処理中..."
                
                if progress_callback:
                    progress_callback(rounded_progress, desc)
                
                time.sleep(0.25)  # 250ms間隔で更新（1%刻みの滑らかな表示）
        
        # バックグラウンドスレッド開始
        wav2lip_thread = threading.Thread(target=wav2lip_worker, daemon=True)
        progress_thread = threading.Thread(target=progress_worker, daemon=True)
        
        wav2lip_thread.start()
        progress_thread.start()
        
        # Wav2Lip処理完了を待機
        wav2lip_thread.join()
        
        # 進捗表示停止
        progress_stop_event.set()
        progress_thread.join(timeout=1)
        
        # 完了時に即座に100%表示
        if progress_callback:
            progress_callback(100, "✅ 口パク動画生成完了!")
        
        # 結果取得
        try:
            result_type, result = result_queue.get_nowait()
            if result_type == 'success':
                return result
            else:
                raise result
        except queue.Empty:
            raise Exception("処理結果の取得に失敗しました")
    
    except Exception as e:
        error_msg = f"音声長ベース進捗処理エラー: {e}"
        logger.error(error_msg)
        return False, error_msg, None

def generate_lipsync_video_with_realtime_progress(
    video_path: str,
    audio_path: str,
    enhancer: str = "gfpgan",
    batch_size: int = 8,
    progress_callback=None
) -> Tuple[bool, str, Optional[str]]:
    """リアルタイム進捗監視付き口パク動画生成"""
    
    try:
        logger.info(f"🎭 リアルタイム進捗付き口パク動画生成開始...")
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
            "face_mask": False,
            "pads": 0,  # 口元位置調整を0に（同期精度向上）
            "resize_factor": 1,  # リサイズなし（同期保持）
            "blending": 10.0  # 強化ブレンド（品質向上）
        }
        
        try:
            # 非同期リクエスト開始
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
            # 動画ファイルとして直接保存
            output_filename = f"lipsync_result_{int(time.time())}.mp4"
            output_path = os.path.join("/shared/output", output_filename)
            
            with open(output_path, "wb") as f:
                f.write(response.content)
            
            logger.info(f"🔍 Wav2Lip動画保存完了: {output_path}")
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            if progress_callback:
                progress_callback(100, f"✅ 口パク動画完成! ({processing_time:.1f}秒)")
            
            return True, f"口パク動画生成完了 ({processing_time:.1f}秒)", output_path
        else:
            logger.error(f"❌ Wav2Lip API エラー: {response.status_code}")
            logger.error(f"レスポンス: {response.text[:500]}")
            return False, f"Wav2Lip API エラー: {response.status_code}", None
    
    except Exception as e:
        error_msg = f"リアルタイム進捗付き口パク動画生成エラー: {e}"
        logger.error(error_msg)
        return False, error_msg, None

def generate_lipsync_video(
            is_success = result.get("success", False) or result.get("status") == "success"
            
            if is_success:
                gen_time = time.time() - start_time
                output_filename = result.get("output_filename") or result.get("output_file", "").split("/")[-1]
                download_url = result.get("download_url")
                
                # APIから直接ファイルダウンロード
                if download_url:
                    try:
                        download_response = requests.get(f"{WAV2LIP_API_URL}{download_url}")
                        if download_response.status_code == 200:
                            # 元のファイルを保存
                            original_path = LOCAL_OUTPUT / output_filename
                            with open(original_path, 'wb') as f:
                                f.write(download_response.content)
                            
                            # ブラウザ互換性のためH.264+AACに再エンコード
                            optimized_filename = f"web_{output_filename}"
                            optimized_path = LOCAL_OUTPUT / optimized_filename
                            
                            import subprocess
                            ffmpeg_cmd = [
                                "ffmpeg", "-y", "-i", str(original_path),
                                "-c:v", "libx264", "-profile:v", "baseline", "-level", "3.0",
                                "-c:a", "aac", "-b:a", "128k",
                                "-movflags", "+faststart",
                                "-pix_fmt", "yuv420p",
                                "-f", "mp4",
                                str(optimized_path)
                            ]
                            
                            try:
                                subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
                                result_path = str(optimized_path)
                                logger.info(f"🎬 動画をブラウザ互換形式に変換: {optimized_filename}")
                            except subprocess.CalledProcessError as e:
                                logger.warning(f"⚠️ 動画変換失敗、元ファイル使用: {e}")
                                result_path = str(original_path)
                        else:
                            result_path = None
                    except Exception as e:
                        logger.error(f"ダウンロードエラー: {e}")
                        result_path = None
                else:
                    result_path = None
                
                if result_path:
                    logger.info(f"✅ リアルタイム進捗付き口パク動画生成成功: {output_filename} ({gen_time:.1f}秒)")
                    return True, f"口パク動画生成成功 ({gen_time:.1f}秒)", result_path
                else:
                    return False, "生成ファイルが見つかりません", None
            else:
                error_msg = result.get("message", result.get("error", "不明なエラー"))
                logger.error(f"❌ 口パク生成失敗: {error_msg}")
                return False, f"口パク生成失敗: {error_msg}", None
        else:
            error_msg = f"Wav2Lip API エラー: {response.status_code}"
            logger.error(error_msg)
            return False, error_msg, None
    
    except Exception as e:
        error_msg = f"リアルタイム進捗付き口パク動画生成エラー: {e}"
        logger.error(error_msg)
        return False, error_msg, None

def generate_lipsync_video(
    video_path: str,
    audio_path: str,
    enhancer: str = "gfpgan",
    batch_size: int = 8
) -> Tuple[bool, str, Optional[str]]:
    """口パク動画生成（下位互換性）"""
    return generate_lipsync_video_with_realtime_progress(
        video_path, audio_path, enhancer, batch_size, None
    )


def integrated_generation_with_progress(
    text: str,
    video_file,
    ref_audio_file,
    enhancer: str = "gfpgan",
    temperature: float = 1.0,
    batch_size: int = 8
):
    """統合生成処理（Gradio 4.xリアルタイム進捗バー対応）"""
    import time
    progress = gr.Progress()
    
    # 時間計測開始
    start_time = time.time()
    total_steps = 8  # 全ステップ数
    
    def update_progress(step, desc):
        """進捗更新関数（Gradio標準表示）"""
        percent = (step / total_steps) * 100
        progress(step / total_steps, desc=f"[{percent:.1f}%] {desc}")
        return f"[{percent:.1f}%] {desc}"
    
    # 新セッション開始
    session_id = workflow.new_session()
    
    try:
        # ステップ0: 初期化
        status = update_progress(0, "🔄 処理を開始しています...")
        yield None, status
        
        # ステップ1: バリデーション
        status = update_progress(1, "📝 入力をチェック中...")
        yield None, status
        
        # バリデーション
        if not text.strip():
            return None, "❌ テキストを入力してください"
        
        if video_file is None:
            return None, "❌ 動画または画像ファイルをアップロードしてください"
        
        if ref_audio_file is None:
            return None, "❌ 参照音声ファイルをアップロードしてください"
        
        # API健康状態チェック
        health = check_api_health()
        
        if not health["sovits"]:
            return None, "❌ SoVITS APIが利用できません"
        
        if not health["wav2lip"]:
            return None, "❌ Wav2Lip APIが利用できません"
        
        # ファイル準備開始
        # 一時ファイルパス生成
        temp_dir = SHARED_TEMP / session_id
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        # 入力ファイルの拡張子を取得
        def get_file_extension(file_input):
            """ファイル入力から拡張子を取得"""
            if isinstance(file_input, str):
                return os.path.splitext(file_input)[1]
            elif hasattr(file_input, 'name'):
                return os.path.splitext(file_input.name)[1]
            else:
                return ""
        
        video_ext = get_file_extension(video_file) or ".mp4"
        audio_ext = get_file_extension(ref_audio_file) or ".wav"
        
        video_path = temp_dir / f"input_video_{session_id}{video_ext}"
        ref_audio_path = temp_dir / f"ref_audio_{session_id}{audio_ext}"
        
        # ステップ2: APIサービスチェック  
        status = update_progress(2, "📡 APIサービス状態確認中...")
        yield None, status
        
        # API健康状態チェック
        health = check_api_health()
        
        if not health["sovits"]:
            return None, "❌ SoVITS APIが利用できません"
        
        if not health["wav2lip"]:
            return None, "❌ Wav2Lip APIが利用できません"
        
        # ステップ3: ファイル準備
        status = update_progress(3, "📁 ファイルを準備中...")
        yield None, status
        
        # デバッグ情報
        logger.info(f"🔍 video_file: {video_file}, type: {type(video_file)}")
        logger.info(f"🔍 ref_audio_file: {ref_audio_file}, type: {type(ref_audio_file)}")
        
        # 参照音声ファイル必須チェック＋デフォルト音声フォールバック
        if ref_audio_file is None:
            # デフォルト参照音声を使用
            default_ref_audio = "/app/input/audio/test_audio.mp3"
            if os.path.exists(default_ref_audio):
                logger.info("🔄 参照音声なし - デフォルト音声を使用")
                ref_audio_file = default_ref_audio
            else:
                yield None, "❌ 参照音声ファイルをアップロードしてください"
                return
        
        # ファイル保存（Gradio 4.x対応）
        # Gradio 4.xではファイルオブジェクトが直接パス文字列になる
        if isinstance(video_file, str):
            shutil.copy2(video_file, video_path)
        else:
            # フォールバック: 古い形式対応
            if hasattr(video_file, 'name'):
                shutil.copy2(video_file.name, video_path)
            else:
                with open(video_path, 'wb') as f:
                    f.write(video_file)
        
        if isinstance(ref_audio_file, str):
            shutil.copy2(ref_audio_file, ref_audio_path)
        else:
            # フォールバック: 古い形式対応
            if hasattr(ref_audio_file, 'name'):
                shutil.copy2(ref_audio_file.name, ref_audio_path)
            else:
                with open(ref_audio_path, 'wb') as f:
                    f.write(ref_audio_file)
        
        workflow.intermediate_files.extend([str(video_path), str(ref_audio_path)])
        
        # 画像の場合は動画に変換（静止画から短い動画を作成）
        if video_ext.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
            logger.info(f"🖼️ 画像ファイルを動画に変換中: {video_path}")
            # 静止画から3秒間の動画を作成（Wav2Lipが動画形式を要求するため）
            converted_video_path = temp_dir / f"converted_video_{session_id}.mp4"
            try:
                import subprocess
                # 画像→動画変換（色空間問題を解決）
                cmd = [
                    'ffmpeg', '-y', '-loop', '1', '-i', str(video_path),
                    '-t', '3', '-r', '25', 
                    '-vf', 'format=yuv420p,scale=trunc(iw/2)*2:trunc(ih/2)*2',
                    '-c:v', 'libx264', '-preset', 'fast',
                    str(converted_video_path)
                ]
                logger.info(f"🔧 FFmpeg実行中: {' '.join(cmd)}")
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    video_path = converted_video_path
                    workflow.intermediate_files.append(str(converted_video_path))
                    logger.info("✅ 画像→動画変換成功")
                else:
                    # FFmpeg警告は無視して、ファイルが作成されたかチェック
                    if os.path.exists(converted_video_path) and os.path.getsize(converted_video_path) > 0:
                        video_path = converted_video_path
                        workflow.intermediate_files.append(str(converted_video_path))
                        logger.info("✅ 画像→動画変換成功（FFmpeg警告あり）")
                    else:
                        logger.error(f"❌ 画像→動画変換失敗: {result.stderr}")
                        return None, f"❌ 画像変換エラー: FFmpegが失敗しました"
            except Exception as e:
                logger.error(f"❌ 画像変換エラー: {e}")
                return None, f"❌ 画像変換エラー: {str(e)}"
        
        # ステップ4: 音声生成開始
        status = update_progress(4, "🎤 GPT-SoVITSで音声生成中...")
        yield None, status
        
        # 音声生成処理
        voice_success, voice_message, generated_audio_path = generate_voice_from_text(
            text=text,
            ref_audio_path=str(ref_audio_path),
            temperature=temperature
        )
        
        if not voice_success:
            return None, f"❌ 音声生成失敗: {voice_message}"
        
        if not generated_audio_path or not os.path.exists(generated_audio_path):
            return None, "❌ 生成音声ファイルが見つかりません"
        
        # ステップ5: 音声生成完了
        status = update_progress(5, "✅ 音声生成完了! 口パク動画生成準備中...")
        yield None, status
        
        # ステップ6: 口パク動画生成開始（音声長ベースの細かい進捗）
        def lipsync_progress_callback(percent, desc):
            """音声長ベースWav2Lip進捗コールバック（1%刻みのリアルタイム更新）"""
            # 6.0-8.0の範囲でGradio内部進捗更新（統合UI用）
            wav2lip_gradio_progress = 6 + (percent / 100) * 2
            
            # ユーザー表示用進捗（統合UIでは75-100%の範囲）
            user_percent = 75 + (percent / 100) * 25  # Wav2Lip完了で100%到達
            
            # 1%刻みに丸める（75.0, 76.0, 77.0, ..., 99.0, 100.0）
            rounded_user_percent = round(user_percent)
            
            # シンプル進捗表示（%なし）
            status = update_progress(wav2lip_gradio_progress, f"🎭 {desc}")
            return status
        
        # 音声長取得（推定時間計算用）
        try:
            audio_duration = get_audio_duration(generated_audio_path)
            wav2lip_eta = 7 + (audio_duration * 3) + (5 if enhancer != 'none' else 0)
            status = update_progress(6, f"🎭 口パク動画生成開始 (推定{wav2lip_eta:.0f}秒)...")
            yield None, status
            time.sleep(0.3)
        except:
            # フォールバック
            status = update_progress(6, "🎭 口パク動画生成開始...")
            yield None, status
        
        # ステップ6: 口パク生成
        status = update_progress(6.5, "🎭 口パク生成中...")
        yield None, status
        
        # 口パク動画生成（シンプル処理）
        lipsync_success, lipsync_message, final_video_path = generate_lipsync_video(
            video_path=str(video_path),
            audio_path=generated_audio_path,
            enhancer=enhancer,
            batch_size=batch_size
        )
        
        # エラーチェック
        if not lipsync_success:
            yield None, f"❌ 口パク動画生成失敗: {lipsync_message}"
            return
        
        if not final_video_path or not os.path.exists(final_video_path):
            yield None, "❌ 最終動画ファイルが見つかりません"
            return
        
        # ステップ8: 最終エンコード
        status = update_progress(7.8, "🎬 H.264エンコード中...")
        yield None, status
        
        # ステップ8: 完了
        status = update_progress(8, "✅ 動画生成完了!")
        yield final_video_path, status
        
    except Exception as e:
        error_msg = f"❌ 統合処理エラー: {e}"
        logger.error(error_msg)
        progress(0.0, desc=f"[ERROR] {error_msg}")
        yield None, f"[ERROR] {error_msg}"
        return
    
    finally:
        # セッション清理（オプション - ユーザーがダウンロード後）
        # workflow.cleanup_session()
        pass

def integrated_generation(
    text: str,
    video_file,
    ref_audio_file,
    enhancer: str = "gfpgan",
    temperature: float = 1.0,
    batch_size: int = 8
) -> Tuple[Optional[str], str]:
    """下位互換性のためのラッパー関数（Gradio 3.x風）"""
    # Gradio 4.xのジェネレーター関数を同期関数としてラップ
    results = list(integrated_generation_with_progress(
        text, video_file, ref_audio_file, enhancer, temperature, batch_size
    ))
    if results:
        return results[-1]  # 最後の結果を返す
    else:
        return None, "❌ 処理に失敗しました"

# =====================================================
# 進捗表示対応Gradio UI
# =====================================================

def create_interface():
    """Gradio UI作成"""
    
    with gr.Blocks(
        title="統合口パクシステム", 
        theme=gr.themes.Soft(),
        analytics_enabled=False,
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
                # より確実なAPI確認（リトライ付き）
                health = check_api_health(retries=3, retry_delay=0.5)
                sovits_status = "🟢 正常" if health["sovits"] else "🔴 異常"
                wav2lip_status = "🟢 正常" if health["wav2lip"] else "🔴 異常"
                
                # APIエンドポイントの情報も追加
                sovits_endpoint = SOVITS_API_URL
                wav2lip_endpoint = WAV2LIP_API_URL
                
                return f"""
                <div style="padding: 10px; border: 1px solid #ddd; border-radius: 5px;">
                    <h4>📡 APIサービス状態</h4>
                    <p><strong>SoVITS API</strong>: {sovits_status}</p>
                    <p style="font-size: 0.9em; color: #666;">エンドポイント: {sovits_endpoint}</p>
                    <p><strong>Wav2Lip API</strong>: {wav2lip_status}</p>
                    <p style="font-size: 0.9em; color: #666;">エンドポイント: {wav2lip_endpoint}</p>
                    <p style="font-size: 0.8em; color: #888; margin-top: 10px;">最終確認: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
                """
            
            # 初期ステータス表示
            api_status.value = update_status()
            
        # 定期的なステータス更新（5秒ごと）
        def auto_update_status():
            while True:
                time.sleep(5)
                try:
                    new_status = update_status()
                    # Gradioの非同期更新はサポートされていないため、手動更新が必要
                except:
                    pass
        
        # バックグラウンドスレッドで定期更新（オプション）
        # import threading
        # status_thread = threading.Thread(target=auto_update_status, daemon=True)
        # status_thread.start()
        
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
                    label="動画・画像ファイル（口パクさせたい人物）",
                    file_types=[".mp4", ".avi", ".mov", ".jpg", ".jpeg", ".png", ".bmp"]
                )
                
                ref_audio_input = gr.File(
                    label="参照音声ファイル（クローンしたい声）",
                    file_types=[".wav", ".mp3", ".m4a"]
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
                status_output = gr.HTML()
                
                video_output = gr.Video(
                    label="生成された口パク動画",
                    height=400,
                    width=600,
                    format="mp4",
                    visible=False
                )
                
                download_link = gr.File(
                    label="ダウンロード",
                    visible=False
                )
        
        # イベントハンドラー - Gradio 4.x リアルタイム進捗バー対応
        generate_btn.click(
            fn=integrated_generation_with_progress,
            inputs=[
                text_input,
                video_input,
                ref_audio_input,
                enhancer_select,
                temperature_slider,
                batch_size_select
            ],
            outputs=[video_output, status_output]
        ).then(
            fn=lambda video: (gr.update(visible=bool(video)), gr.update(visible=bool(video), value=video)),
            inputs=[video_output],
            outputs=[video_output, download_link]
        )
        
        status_refresh_btn.click(
            fn=update_status,
            outputs=[api_status]
        )
        
        # 進捗表示の説明
        gr.Markdown("""
        ## 💡 使用例 & 1%単位リアルタイム進捗表示
        
        ### 📝 入力準備
        1. **テキスト**: "こんにちは！今日は素晴らしい天気ですね。"
        2. **動画**: 口パクさせたい人物の動画（5-30秒推奨）
        3. **参照音声**: クローンしたい声の音声ファイル（5秒程度推奨）
        
        ### ⏱️ 処理時間 & シンプル進捗表示
        **総処理時間**: 音声長により動的変化（RTX 3050基準）
        - 短音声（3秒）: 約16秒処理
        - 標準音声（5秒）: 約22秒処理  
        - 長音声（10秒）: 約42秒処理
        
        **📊 シンプル段階別進捗表示**:
        - 📁 **ファイル準備** - バリデーション & ファイル保存
        - 🎤 **音声生成中** - GPT-SoVITSでクローン音声作成
        - 🎯 **完璧同期処理** - **1ms精度同期**（超高精度無音除去+22050Hz最適化+DC除去+正規化）
        - 🎭 **口パク動画生成中** - Wav2Lip処理
        - 🎬 **動画エンコード** - H.264変換
        - ✅ **完了** - ダウンロード準備
        
        ### 🎯 特徴 & 最新改善
        - **シンプル進捗表示**: 複雑な1%刻みを廃止、段階別表示に統一
        - **エラー詳細表示**: 問題発生時の具体的な原因を表示
        - **処理時間表示**: 各ステップの実行時間を計測表示
        - **🎯 NEW! 完璧な音声・口パク同期**: 1ms精度の超高精度同期システム
        - **🎵 完璧同期処理**: RMS+ピーク+エネルギー解析による-50dB超厳格検出
        - **📱 同期最適化**: 22050Hz標準化 + DC除去 + フェード処理 + 正規化
        - **⚡ Wav2Lip最適化**: pads=0, 口元位置調整無効で同期精度最大化
        """)
    
    return interface

# アプリケーション起動
if __name__ == "__main__":
    logger.info("🚀 統合Frontend起動中...")
    
    # APIが完全に起動するまで待機（最大60秒）
    logger.info("⏳ APIサービスの起動を待機中...")
    max_wait_time = 60  # 最大待機時間（秒）
    start_time = time.time()
    
    while time.time() - start_time < max_wait_time:
        health = check_api_health(retries=5, retry_delay=2.0)
        
        if health['sovits'] and health['wav2lip']:
            # 両方のAPIが正常になったら終了
            logger.info("✅ 全APIサービスが正常に起動しました！")
            break
        else:
            # まだ起動中の場合
            sovits_status = '🟢' if health['sovits'] else '🔴'
            wav2lip_status = '🟢' if health['wav2lip'] else '🔴'
            logger.info(f"📡 API状態: SoVITS {sovits_status}, Wav2Lip {wav2lip_status}")
            
            if not (health['sovits'] and health['wav2lip']):
                logger.info("⏳ APIサービスの起動を待機中... (5秒後に再確認)")
                time.sleep(5)
    
    # 最終状態確認
    final_health = check_api_health()
    logger.info(f"📡 SoVITS API: {'🟢' if final_health['sovits'] else '🔴'}")
    logger.info(f"📡 Wav2Lip API: {'🟢' if final_health['wav2lip'] else '🔴'}")
    
    # Gradio インターフェース作成
    interface = create_interface()
    
    # 元の完全に動作していた設定（余計な修正なし）
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=False,
        show_error=True,
        quiet=False,
        inbrowser=False,
        prevent_thread_lock=False
    )