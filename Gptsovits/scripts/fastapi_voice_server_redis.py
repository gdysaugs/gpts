#!/usr/bin/env python3
"""
FastAPI GPT-SoVITS 常駐音声生成サーバー (Redis Lock統合版)
初期化1回のみ、各リクエスト2-3秒で高速応答
GPU排他制御: Redis分散ロック対応
"""

import os
import sys
import time
import logging
import asyncio
from pathlib import Path
from typing import Optional
import uuid
import tempfile
import base64
import urllib.request

# FastAPI
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Redis GPU Lock統合
from gpu_redis_lock import GPUResourceManager, init_gpu_manager

# GPT-SoVITS
os.chdir('/app')
sys.path.insert(0, '/app')
sys.path.insert(0, '/app/GPT_SoVITS')

import torch
import soundfile as sf
import numpy as np

# ロギング設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# グローバル変数
app = FastAPI(title="GPT-SoVITS Voice Cloning API (Redis)", version="2.0.0")
MODELS_LOADED = False
CUSTOM_SOVITS_PATH = None

# Redis GPU排他制御
gpu_manager: Optional[GPUResourceManager] = None
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")

# プリロード済みモデルのキャッシュ
PRELOADED_LANGDETECT = None
CACHE_DIR = "/app/cache"

# CORS設定（開発用）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 本番では適切に設定
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# リクエストモデル
class VoiceCloneRequest(BaseModel):
    ref_text: str
    target_text: str
    ref_audio_base64: Optional[str] = None  # Base64エンコード音声
    temperature: float = 1.0
    top_k: int = 5
    top_p: float = 1.0

class VoiceCloneResponse(BaseModel):
    success: bool
    message: str
    audio_base64: Optional[str] = None
    generation_time: float
    audio_duration: float
    realtime_factor: float

def setup_torch_optimizations():
    """Torch最適化セットアップ"""
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.cuda.empty_cache()
        logger.info("🚀 RTX 3050 TensorCore最適化有効")
    
    torch.set_float32_matmul_precision('medium')
    torch.set_num_threads(8)
    logger.info("✅ PyTorch最適化完了")

def comprehensive_monkey_patch():
    """モンキーパッチ適用"""
    try:
        import inference_webui
        
        # カスタムSoVITSモデルローダー
        def load_sovits_new():
            """新しいSoVITSモデルローダー"""
            global CUSTOM_SOVITS_PATH
            
            if CUSTOM_SOVITS_PATH and os.path.exists(CUSTOM_SOVITS_PATH):
                logger.info(f"🎭 カスタムSoVITSモデル読み込み: {CUSTOM_SOVITS_PATH}")
                device = "cuda" if torch.cuda.is_available() else "cpu"
                
                checkpoint = torch.load(CUSTOM_SOVITS_PATH, map_location=device)
                
                if "weight" in checkpoint:
                    return checkpoint["weight"]
                else:
                    return checkpoint
            else:
                logger.info("📦 標準v2モデル使用")
                return inference_webui.get_weights_names()[0]
        
        # change_sovits_weights関数への統合
        original_change_sovits = inference_webui.change_sovits_weights
        
        def change_sovits_weights(sovits_path):
            """SoVITS重み変更（カスタムモデル対応）"""
            global CUSTOM_SOVITS_PATH
            CUSTOM_SOVITS_PATH = sovits_path
            
            if sovits_path and os.path.exists(sovits_path):
                logger.info(f"🔄 SoVITSモデル切り替え: {sovits_path}")
                custom_weights = load_sovits_new()
                return original_change_sovits(custom_weights)
            else:
                return original_change_sovits(sovits_path)
        
        # モンキーパッチ適用
        inference_webui.change_sovits_weights = change_sovits_weights
        logger.info("🐒 モンキーパッチ適用完了")
        
    except Exception as e:
        logger.error(f"❌ モンキーパッチエラー: {e}")
        
def preload_dependencies():
    """依存関係の事前ロード"""
    global PRELOADED_LANGDETECT, CACHE_DIR
    
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        
        # 1. 言語検出ライブラリ事前ダウンロード
        logger.info("📥 言語検出モデル事前ダウンロード開始...")
        import langdetect
        from langdetect import detect
        detect("This is a test sentence for preloading.")
        logger.info("✅ 言語検出モデル事前ロード完了")
        
        # 2. Open JTalk辞書事前ダウンロード
        logger.info("📥 Open JTalk辞書事前ダウンロード開始...")
        import openjtalk
        openjtalk.g2p("テストです", kana=True)
        logger.info("✅ Open JTalk辞書事前ロード完了")
        
        # 3. 音声処理ライブラリ初期化
        logger.info("📥 音声処理ライブラリ初期化...")
        import jieba
        jieba.initialize()
        logger.info("✅ 音声処理ライブラリ初期化完了")
        
        logger.info("🚀 全依存関係事前ロード完了！")
        
    except Exception as e:
        logger.warning(f"⚠️ 事前ロードエラー（続行可能）: {e}")

async def initialize_models():
    """モデル初期化（1回のみ実行）"""
    global MODELS_LOADED, CUSTOM_SOVITS_PATH
    
    if MODELS_LOADED:
        return
    
    logger.info("🔄 GPT-SoVITS初期化開始...")
    start_time = time.time()
    
    try:
        # 1. 事前ダウンロード＆最適化
        logger.info("📥 事前ダウンロード実行中...")
        preload_dependencies()
        
        # 2. PyTorch最適化
        setup_torch_optimizations()
        
        # 3. GPT-SoVITS初期化
        logger.info("🎭 GPT-SoVITS本体初期化...")
        sys.path.insert(0, '/app/GPT_SoVITS')
        
        import inference_webui
        
        # 4. モンキーパッチ適用
        comprehensive_monkey_patch()
        
        # 5. 日本語特化モデル設定
        ja_model_path = "/app/GPT_SoVITS/pretrained_models/gpt-sovits-ja-h/hscene-e17.ckpt"
        if os.path.exists(ja_model_path):
            logger.info(f"🇯🇵 日本語特化モデル検出: {ja_model_path}")
            CUSTOM_SOVITS_PATH = ja_model_path
        else:
            logger.info("📦 標準v2モデル使用")
        
        # 6. 基本パラメータ設定
        logger.info("⚙️ 基本設定適用...")
        inference_webui.change_choices()
        
        # 7. GPT and SoVITS weights initialization
        if CUSTOM_SOVITS_PATH:
            logger.info(f"🎯 カスタムモデル初期化: {CUSTOM_SOVITS_PATH}")
            inference_webui.change_sovits_weights(CUSTOM_SOVITS_PATH)
        
        MODELS_LOADED = True
        
        init_time = time.time() - start_time
        logger.info(f"✅ GPT-SoVITS初期化完了！({init_time:.1f}秒)")
        
    except Exception as e:
        logger.error(f"❌ 初期化エラー: {e}")
        raise HTTPException(status_code=500, detail=f"Model initialization failed: {e}")

async def generate_voice_fast(
    ref_audio_path: str,
    ref_text: str, 
    target_text: str,
    temperature: float = 1.0,
    top_k: int = 5,
    top_p: float = 1.0
) -> dict:
    """高速音声生成（optimized）"""
    
    start_time = time.time()
    
    try:
        logger.info(f"🎤 音声生成開始: '{target_text[:30]}...'")
        
        # GPT-SoVITS処理
        sys.path.insert(0, '/app/GPT_SoVITS')
        import inference_webui
        
        # 短文処理対応（20文字未満は延長）
        if len(target_text) < 20:
            target_text = target_text + "。" + target_text[:10] + "。"
            logger.info(f"📝 短文延長: {target_text}")
        
        # 音声生成実行
        result = inference_webui.get_tts_wav(
            ref_wav_path=ref_audio_path,
            prompt_text=ref_text,
            prompt_language="日本語",
            text=target_text,
            text_language="日本語",
            how_to_cut="不切",
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            ref_free=True
        )
        
        generation_time = time.time() - start_time
        
        if result and len(result) >= 2:
            sr, audio_data = result[1], result[0]
            
            # NumPy配列に変換
            if isinstance(audio_data, tuple):
                audio_data = audio_data[1]
            
            if isinstance(audio_data, torch.Tensor):
                audio_data = audio_data.cpu().numpy()
            
            audio_data = np.array(audio_data, dtype=np.float32)
            
            # 正規化
            if audio_data.max() > 1.0:
                audio_data = audio_data / 32768.0
            
            # 品質チェック
            duration = len(audio_data) / sr if sr > 0 else 0
            rms = np.sqrt(np.mean(audio_data**2)) if len(audio_data) > 0 else 0
            non_silence_ratio = np.sum(np.abs(audio_data) > 0.01) / len(audio_data) if len(audio_data) > 0 else 0
            realtime_factor = generation_time / duration if duration > 0 else 0
            
            logger.info(f"✅ 生成完了: {generation_time:.2f}s, 長さ{duration:.1f}s, RMS{rms:.3f}, 品質{non_silence_ratio:.1%}, RTF{realtime_factor:.1f}")
            
            return {
                "success": True,
                "message": f"音声生成成功 ({generation_time:.1f}秒)",
                "audio_data": audio_data,
                "sample_rate": sr,
                "generation_time": generation_time,
                "audio_duration": duration,
                "realtime_factor": realtime_factor,
                "quality_metrics": {
                    "rms": float(rms),
                    "non_silence_ratio": float(non_silence_ratio),
                    "sample_rate": int(sr)
                }
            }
        else:
            raise Exception("音声生成結果が無効です")
            
    except Exception as e:
        error_time = time.time() - start_time
        logger.error(f"❌ 音声生成エラー ({error_time:.1f}s): {e}")
        return {
            "success": False,
            "message": f"音声生成失敗: {e}",
            "generation_time": error_time,
            "audio_duration": 0,
            "realtime_factor": 0
        }

@app.on_event("startup")
async def startup_event():
    """サーバー起動時イベント"""
    global gpu_manager
    
    logger.info("🚀 GPT-SoVITS FastAPI サーバー (Redis版) 起動中...")
    
    # Redis GPU Manager初期化
    try:
        gpu_manager = init_gpu_manager(REDIS_URL)
        logger.info(f"🔒 Redis GPU Manager初期化完了: {REDIS_URL}")
    except Exception as e:
        logger.error(f"❌ Redis接続エラー: {e}")
        raise
    
    # モデル初期化
    await initialize_models()
    
    # ウォームアップ実行
    await warmup_with_redis_lock()
    
    logger.info("🎭 GPT-SoVITS サーバー (Redis版) 準備完了！")

async def warmup_with_redis_lock():
    """Redis Lock使用のウォームアップ"""
    global gpu_manager
    
    try:
        # モデルがロードされていない場合は初期化
        if not MODELS_LOADED:
            await initialize_models()
        
        # Redis Lock使用のテスト音声生成
        logger.info("🔒 Redis Lock使用ウォームアップ開始...")
        
        gpu_lock = gpu_manager.get_gpu_lock(gpu_id="sovits")
        
        try:
            async with gpu_lock.async_acquire():
                logger.info("🎤 Redis Lock取得成功 - テスト音声生成中...")
                
                # デフォルトの参照音声を使用
                ref_audio_path = "/app/input/reference_5sec.wav"
                
                # テスト音声生成
                result = await generate_voice_fast(
                    ref_audio_path=ref_audio_path,
                    ref_text="おはようございます",
                    target_text="Redis Lockテスト音声です。",
                    temperature=1.0,
                    top_k=5,
                    top_p=1.0
                )
                
                if result["success"]:
                    logger.info("✅ Redis Lock ウォームアップ完了！初回から高速処理可能です")
                else:
                    logger.warning(f"⚠️ ウォームアップ警告: {result['message']}")
                    
        except Exception as e:
            logger.error(f"❌ Redis Lock ウォームアップエラー: {e}")
            # フォールバック実行
            logger.info("🔄 フォールバック実行...")
            ref_audio_path = "/app/input/reference_5sec.wav"
            result = await generate_voice_fast(
                ref_audio_path=ref_audio_path,
                ref_text="おはようございます", 
                target_text="フォールバックテスト音声です。",
                temperature=1.0,
                top_k=5,
                top_p=1.0
            )
            logger.info("✅ フォールバック ウォームアップ完了")
        
    except Exception as e:
        logger.error(f"❌ ウォームアップ全体エラー: {e}")

@app.get("/")
async def root():
    """サーバー情報"""
    return {
        "service": "GPT-SoVITS Voice Cloning API (Redis版)",
        "version": "2.0.0", 
        "status": "ready" if MODELS_LOADED else "initializing",
        "redis_url": REDIS_URL.replace("redis://", "redis://***"),
        "gpu_available": torch.cuda.is_available(),
        "message": "Redis分散ロック対応の高速音声クローニングAPI"
    }

@app.get("/health")
async def health_check():
    """ヘルスチェック"""
    global gpu_manager
    
    try:
        # Redis接続テスト
        stats = await gpu_manager.get_usage_stats()
        
        return {
            "status": "healthy",
            "models_loaded": MODELS_LOADED,
            "gpu_available": torch.cuda.is_available(),
            "redis_connected": True,
            "gpu_stats": stats
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "models_loaded": MODELS_LOADED,
            "gpu_available": torch.cuda.is_available(),
            "redis_connected": False
        }

@app.get("/clone-voice-simple")
async def clone_voice_simple_get(
    ref_text: str,
    target_text: str,
    temperature: float = 1.0,
    background_tasks: BackgroundTasks = None
):
    """シンプル音声クローニング（Redis Lock対応）"""
    global gpu_manager
    
    if not MODELS_LOADED:
        await initialize_models()
    
    # Redis Lock使用の音声生成
    gpu_lock = gpu_manager.get_gpu_lock(gpu_id="sovits", timeout=30)
    
    try:
        async with gpu_lock.async_acquire():
            return await _execute_voice_synthesis(ref_text, target_text, temperature, background_tasks)
    except Exception as e:
        logger.error(f"❌ Redis Lock エラー: {e}")
        # フォールバック実行
        logger.warning("⚠️ Redis Lock失敗 - フォールバック実行")
        return await _execute_voice_synthesis(ref_text, target_text, temperature, background_tasks)

async def _execute_voice_synthesis(ref_text: str, target_text: str, temperature: float, background_tasks):
    """GPU処理を実行する内部関数"""
    try:
        result = await generate_voice_fast(
            ref_audio_path="/app/input/reference_5sec.wav",
            ref_text=ref_text,
            target_text=target_text,
            temperature=temperature,
            top_k=5,
            top_p=1.0
        )
        
        if result["success"]:
            # 音声ファイル保存
            audio_data = result["audio_data"]
            sample_rate = result["sample_rate"]
            
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            safe_text = target_text[:30].replace("/", "_").replace("\\", "_")
            filename = f"fastapi_{timestamp}_{safe_text}.wav"
            output_path = f"/app/output/{filename}"
            
            # ファイル保存
            sf.write(output_path, audio_data, sample_rate)
            
            logger.info(f"💾 音声ファイル保存: {output_path}")
            
            if background_tasks:
                background_tasks.add_task(lambda: logger.info(f"📁 バックグラウンド処理完了: {filename}"))
            
            return FileResponse(
                output_path,
                media_type="audio/wav",
                filename=filename,
                headers={
                    "X-Generation-Time": str(result["generation_time"]),
                    "X-Audio-Duration": str(result["audio_duration"]),
                    "X-Realtime-Factor": str(result["realtime_factor"])
                }
            )
        else:
            raise HTTPException(status_code=500, detail=result["message"])
            
    except Exception as e:
        logger.error(f"❌ 音声合成実行エラー: {e}")
        raise HTTPException(status_code=500, detail=f"音声生成エラー: {e}")

@app.get("/gpu-stats")
async def get_gpu_stats():
    """GPU使用統計"""
    global gpu_manager
    
    try:
        stats = await gpu_manager.get_usage_stats()
        return {
            "success": True,
            "stats": stats,
            "torch_gpu_available": torch.cuda.is_available(),
            "torch_gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--redis-url", default="redis://redis:6379/0")
    
    args = parser.parse_args()
    
    # Redis URL設定
    REDIS_URL = args.redis_url
    
    logger.info(f"🚀 サーバー起動: {args.host}:{args.port}")
    logger.info(f"🔒 Redis URL: {REDIS_URL}")
    
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info"
    )