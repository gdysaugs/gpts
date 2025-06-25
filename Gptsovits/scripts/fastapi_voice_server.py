#!/usr/bin/env python3
"""
FastAPI GPT-SoVITS 常駐音声生成サーバー
初期化1回のみ、各リクエスト2-3秒で高速応答
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

# FastAPI
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

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
app = FastAPI(title="GPT-SoVITS Voice Cloning API", version="1.0.0")
MODELS_LOADED = False
CUSTOM_SOVITS_PATH = None

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
    from GPT_SoVITS import inference_webui
    
    original_load_sovits = inference_webui.load_sovits_new
    original_change_sovits = inference_webui.change_sovits_weights
    
    def custom_load_sovits_new(sovits_path):
        global CUSTOM_SOVITS_PATH
        if CUSTOM_SOVITS_PATH and os.path.exists(CUSTOM_SOVITS_PATH):
            actual_path = CUSTOM_SOVITS_PATH
        else:
            actual_path = sovits_path
        
        if actual_path.endswith('.ckpt'):
            checkpoint = torch.load(actual_path, map_location='cpu')
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            dict_s2 = {
                'weight': state_dict,
                'config': {
                    'model': {
                        'version': 'v2',
                        'semantic_frame_rate': '25hz',
                        'inter_channels': 192,
                        'hidden_channels': 192,
                        'filter_channels': 768,
                        'n_heads': 2,
                        'n_layers': 6,
                        'kernel_size': 3,
                        'p_dropout': 0.1,
                        'ssl_dim': 768,
                        'n_speakers': 300
                    },
                    'data': {
                        'sampling_rate': 32000,
                        'filter_length': 2048,
                        'hop_length': 640,
                        'win_length': 2048,
                        'n_speakers': 300,
                        'cleaned_text': True,
                        'add_blank': True,
                        'n_symbols': 178
                    }
                }
            }
            return dict_s2
        else:
            return original_load_sovits(actual_path)
    
    def custom_change_sovits_weights(sovits_path, prompt_language=None, text_language=None):
        global CUSTOM_SOVITS_PATH
        if CUSTOM_SOVITS_PATH:
            sovits_path = "/app/GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth"
        return original_change_sovits(sovits_path, prompt_language, text_language)
    
    inference_webui.load_sovits_new = custom_load_sovits_new
    inference_webui.change_sovits_weights = custom_change_sovits_weights
    logger.info("🔧 モンキーパッチ適用完了")

def apply_torch_compile_optimization():
    """Torch.compile最適化適用"""
    try:
        if not hasattr(torch, 'compile'):
            logger.warning("⚠️ PyTorch 2.0+が必要（Torch.compile非対応）")
            return
        
        from GPT_SoVITS import inference_webui
        
        # SoVITSモデル最適化
        if hasattr(inference_webui, 'vq_model') and inference_webui.vq_model is not None:
            logger.info("🔥 SoVITSモデルcompile最適化中...")
            inference_webui.vq_model = torch.compile(
                inference_webui.vq_model,
                mode="max-autotune",
                dynamic=True,
                backend="inductor"
            )
        
        # GPTモデル最適化
        if hasattr(inference_webui, 't2s_model') and inference_webui.t2s_model is not None:
            logger.info("🔥 GPTモデルcompile最適化中...")
            inference_webui.t2s_model = torch.compile(
                inference_webui.t2s_model,
                mode="max-autotune",
                dynamic=True,
                backend="inductor"
            )
        
        # HuBERTモデル最適化
        if hasattr(inference_webui, 'hubert_model') and inference_webui.hubert_model is not None:
            logger.info("🔥 HuBERTモデルcompile最適化中...")
            inference_webui.hubert_model = torch.compile(
                inference_webui.hubert_model,
                mode="reduce-overhead",
                dynamic=True,
                backend="inductor"
            )
        
        logger.info("🚀 Torch.compile最適化完了")
        
    except Exception as e:
        logger.error(f"❌ Torch.compile最適化エラー: {e}")

async def initialize_models():
    """モデル初期化（サーバー起動時1回のみ）"""
    global MODELS_LOADED, CUSTOM_SOVITS_PATH
    
    if MODELS_LOADED:
        return
    
    logger.info("🚀 === サーバー初期化開始 ===")
    init_start = time.time()
    
    try:
        # カスタムモデルパス設定
        CUSTOM_SOVITS_PATH = "/app/GPT_SoVITS/pretrained_models/gpt-sovits-ja-h/hscene-e17.ckpt"
        
        # 最適化設定
        setup_torch_optimizations()
        comprehensive_monkey_patch()
        
        # モデルロード
        from GPT_SoVITS.inference_webui import change_sovits_weights
        default_path = "/app/GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth"
        change_sovits_weights(default_path)
        
        # Warm-up推論
        logger.info("🔥 Warm-up推論実行中...")
        from GPT_SoVITS.inference_webui import get_tts_wav
        
        dummy_gen = get_tts_wav(
            ref_wav_path="/app/input/reference_5sec.wav",
            prompt_text="こんにちは",
            prompt_language="Japanese",
            text="テスト",
            text_language="Japanese",
            how_to_cut="不切",
            top_k=5,
            top_p=1.0,
            temperature=1.0,
            ref_free=True
        )
        
        # 一つだけ生成してWarm-up完了
        for i, item in enumerate(dummy_gen):
            if i == 0:
                break
        
        # Torch.compile最適化
        apply_torch_compile_optimization()
        
        # CUDAキャッシュ最適化
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        
        MODELS_LOADED = True
        init_time = time.time() - init_start
        
        logger.info(f"✅ === サーバー初期化完了: {init_time:.2f}秒 ===")
        logger.info("🎯 以降のリクエストは2-3秒で応答予定")
        
    except Exception as e:
        logger.error(f"❌ サーバー初期化エラー: {e}")
        raise

def ensure_text_length(text: str, min_length: int = 20) -> str:
    """テキストが指定文字数未満の場合、自然に延長する"""
    if len(text) >= min_length:
        return text
    
    # 短いテキストの場合の延長パターン
    extensions = [
        "とても良い音声ですね。",
        "素晴らしい結果だと思います。",
        "音声クローニングの技術は凄いです。",
        "これは高品質な音声生成です。",
        "日本語特化モデルで生成されました。"
    ]
    
    # 文末に句点がない場合は追加
    if not text.endswith(('。', '！', '？')):
        text += "。"
    
    # 20文字以上になるまで延長
    while len(text) < min_length:
        # 一番適当な延長を選択（句点の重複を避ける）
        extension = extensions[len(text) % len(extensions)]
        text += extension
    
    logger.info(f"📝 テキスト自動延長: {len(text)}文字 → {text}")
    return text

async def generate_voice_fast(ref_audio_path: str, ref_text: str, target_text: str, 
                             temperature: float = 1.0, top_k: int = 5, top_p: float = 1.0) -> dict:
    """高速音声生成（初期化済み前提）"""
    
    if not MODELS_LOADED:
        raise HTTPException(status_code=500, detail="モデルが初期化されていません")
    
    # テキストを20文字以上に自動延長
    target_text = ensure_text_length(target_text, 20)
    
    generation_start = time.time()
    
    try:
        from GPT_SoVITS.inference_webui import get_tts_wav
        
        # 音声生成実行
        result_generator = get_tts_wav(
            ref_wav_path=ref_audio_path,
            prompt_text=ref_text,
            prompt_language="Japanese",
            text=target_text,
            text_language="Japanese",
            how_to_cut="不切",
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            ref_free=True
        )
        
        # 結果処理
        audio_segments = []
        for i, item in enumerate(result_generator):
            if isinstance(item, tuple) and len(item) == 2:
                sample_rate, audio_data = item
                audio_segments.append(audio_data)
        
        if not audio_segments:
            raise Exception("音声生成失敗")
        
        # 音声連結
        if len(audio_segments) > 1:
            final_audio = np.concatenate(audio_segments)
        else:
            final_audio = audio_segments[0]
        
        generation_time = time.time() - generation_start
        audio_duration = len(final_audio) / 32000
        realtime_factor = audio_duration / generation_time
        
        # 音声品質統計
        audio_rms = float(np.sqrt(np.mean(final_audio ** 2)))
        non_silence_ratio = float(np.sum(np.abs(final_audio) > np.max(np.abs(final_audio)) * 0.01) / len(final_audio))
        
        return {
            'audio_data': final_audio,
            'sample_rate': 32000,
            'generation_time': generation_time,
            'audio_duration': audio_duration,
            'realtime_factor': realtime_factor,
            'audio_rms': audio_rms,
            'non_silence_ratio': non_silence_ratio
        }
        
    except Exception as e:
        logger.error(f"❌ 音声生成エラー: {e}")
        raise HTTPException(status_code=500, detail=f"音声生成失敗: {str(e)}")

# === FastAPI エンドポイント ===

@app.on_event("startup")
async def startup_event():
    """サーバー起動時初期化"""
    await initialize_models()

@app.get("/")
async def root():
    """ヘルスチェック"""
    return {
        "message": "GPT-SoVITS Voice Cloning API",
        "models_loaded": MODELS_LOADED,
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name() if torch.cuda.is_available() else None
    }

@app.post("/clone-voice", response_model=VoiceCloneResponse)
async def clone_voice_endpoint(
    ref_text: str = Form(...),
    target_text: str = Form(...),
    temperature: float = Form(1.0),
    top_k: int = Form(5),
    top_p: float = Form(1.0),
    ref_audio: UploadFile = File(...)
):
    """音声クローニングAPI"""
    
    try:
        # 参照音声を一時ファイルに保存
        temp_audio_path = f"/tmp/{uuid.uuid4()}.wav"
        
        with open(temp_audio_path, "wb") as f:
            f.write(await ref_audio.read())
        
        # 音声生成
        result = await generate_voice_fast(
            ref_audio_path=temp_audio_path,
            ref_text=ref_text,
            target_text=target_text,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )
        
        # 生成音声を一時ファイルに保存
        output_path = f"/tmp/{uuid.uuid4()}_output.wav"
        sf.write(output_path, result['audio_data'], result['sample_rate'])
        
        # Base64エンコード
        with open(output_path, "rb") as f:
            audio_base64 = base64.b64encode(f.read()).decode('utf-8')
        
        # クリーンアップ
        os.remove(temp_audio_path)
        os.remove(output_path)
        
        return VoiceCloneResponse(
            success=True,
            message="音声生成成功",
            audio_base64=audio_base64,
            generation_time=result['generation_time'],
            audio_duration=result['audio_duration'],
            realtime_factor=result['realtime_factor']
        )
        
    except Exception as e:
        logger.error(f"❌ API エラー: {e}")
        return VoiceCloneResponse(
            success=False,
            message=f"エラー: {str(e)}",
            generation_time=0,
            audio_duration=0,
            realtime_factor=0
        )

@app.get("/clone-voice-simple")
async def clone_voice_simple(
    ref_text: str,
    target_text: str,
    temperature: float = 1.0,
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """シンプル音声クローニング（固定参照音声）"""
    
    try:
        result = await generate_voice_fast(
            ref_audio_path="/app/input/reference_5sec.wav",
            ref_text=ref_text,
            target_text=target_text,
            temperature=temperature
        )
        
        # 一時ファイルに保存してレスポンス
        temp_output_path = f"/tmp/{uuid.uuid4()}_output.wav"
        sf.write(temp_output_path, result['audio_data'], result['sample_rate'])
        
        # outputディレクトリにも永続保存
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_target = target_text.replace(' ', '_').replace('！', '').replace('？', '')[:30]
        permanent_path = f"/app/output/fastapi_{timestamp}_{safe_target}.wav"
        sf.write(permanent_path, result['audio_data'], result['sample_rate'])
        
        # 送信後にファイル削除をバックグラウンドタスクに追加
        background_tasks.add_task(os.remove, temp_output_path)
        
        return FileResponse(
            temp_output_path,
            media_type="audio/wav",
            filename="generated_voice.wav"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # サーバー起動
    uvicorn.run(
        "fastapi_voice_server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )