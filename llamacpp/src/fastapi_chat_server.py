#!/usr/bin/env python3
"""
FastAPI LLM Chat Server with Enhanced Preloading
事前ロード最適化による高速LLMチャットサーバー
"""

import asyncio
import uvicorn
import time
import logging
import os
import sys
import yaml
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Generator
import tempfile
import urllib.request

# FastAPI
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# LLM Engine
from llm_engine import LLMEngine, GenerationConfig, create_engine

# ロギング設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# グローバル変数
app = FastAPI(title="LLM Chat API with Preloading", version="2.0.0")
llm_engine: Optional[LLMEngine] = None
server_start_time = datetime.now()
MODELS_LOADED = False

# シンプルなGPU排他制御
gpu_lock: Optional[asyncio.Lock] = None

# プリロード済みキャッシュ
PRELOADED_CACHE = {}
CACHE_DIR = "/app/cache"

# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# リクエストモデル
class ChatRequest(BaseModel):
    message: str = Field(..., description="ユーザーメッセージ")
    use_history: bool = Field(True, description="会話履歴を使用するか")
    stream: bool = Field(False, description="ストリーミング応答")
    generation_config: Optional[Dict] = Field(None, description="生成設定")

class ChatResponse(BaseModel):
    response: str = Field(..., description="アシスタント応答")
    inference_time: Optional[float] = Field(None, description="推論時間(秒)")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

class InteractiveRequest(BaseModel):
    message: str = Field(..., description="メッセージ")
    character: str = Field("tsundere", description="キャラクター")
    temperature: float = Field(0.7, ge=0.1, le=2.0, description="温度")
    max_tokens: int = Field(512, ge=1, le=2048, description="最大トークン数")

class InteractiveResponse(BaseModel):
    response: str = Field(..., description="応答")
    character: str = Field(..., description="キャラクター")
    inference_time: float = Field(..., description="推論時間")
    tokens_per_second: float = Field(..., description="トークン/秒")

class HealthResponse(BaseModel):
    status: str = Field(..., description="ステータス")
    model_loaded: bool = Field(..., description="モデル読み込み状態")
    gpu_available: bool = Field(..., description="GPU利用可能性")
    uptime: str = Field(..., description="稼働時間")
    preload_status: Dict = Field(..., description="事前ロード状態")

# === 事前ロード機能 ===

def setup_optimizations():
    """最適化設定"""
    try:
        # CPU最適化（必ず実行）
        os.environ['OMP_NUM_THREADS'] = '8'
        os.environ['MKL_NUM_THREADS'] = '8'
        os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # CUDA最適化
        os.environ['CUDA_CACHE_DISABLE'] = '0'    # CUDAキャッシュ有効
        
        # GPU最適化（利用可能な場合）
        try:
            import torch
            if torch.cuda.is_available():
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
                torch.cuda.empty_cache()
                logger.info("🚀 PyTorch GPU最適化有効")
        except ImportError:
            logger.info("📝 PyTorch未検出 - CUDA環境変数で最適化")
            # PyTorchがなくてもCUDA環境変数で最適化
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
            os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        
        # llama-cpp-python特有の最適化
        os.environ['LLAMA_CUDA'] = '1'
        os.environ['LLAMA_CUBLAS'] = '1'
        
        logger.info("✅ 最適化設定完了")
        
    except Exception as e:
        logger.warning(f"⚠️ 最適化設定エラー: {e}")

async def preload_all_dependencies():
    """全ての依存関係を事前ロード・キャッシュ"""
    global PRELOADED_CACHE
    
    # キャッシュディレクトリ作成
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    # 1. 基本的な依存関係のプリロード
    logger.info("🔥 基本依存関係プリロード中...")
    try:
        # JSONライブラリの事前ロード
        import json
        PRELOADED_CACHE['json'] = json
        
        # yamlライブラリの事前ロード
        import yaml
        PRELOADED_CACHE['yaml'] = yaml
        
        # datetimeライブラリの事前ロード
        from datetime import datetime
        PRELOADED_CACHE['datetime'] = datetime
        
        logger.info("✅ 基本依存関係プリロード完了")
    except Exception as e:
        logger.warning(f"⚠️ 基本依存関係プリロード失敗: {e}")
    
    # 2. LLM関連の依存関係事前ロード
    logger.info("🔥 LLM依存関係プリロード中...")
    try:
        # llama-cpp-pythonの事前ロード
        from llama_cpp import Llama
        PRELOADED_CACHE['llama_cpp'] = Llama
        
        # 設定ファイルの事前ロード
        config_path = "/app/config/model_config.yaml"
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                PRELOADED_CACHE['config'] = yaml.safe_load(f)
        
        logger.info("✅ LLM依存関係プリロード完了")
    except Exception as e:
        logger.warning(f"⚠️ LLM依存関係プリロード失敗: {e}")
    
    # 3. キャラクタープリセットの事前ロード
    logger.info("🔥 キャラクタープリセット事前ロード中...")
    try:
        presets = {
            'tsundere': "あなたはツンデレの女の子です。厳しくて素っ気ない態度を取りますが、内心では相手を心配しています。「べ、別に〜」「〜なんだからね！」「ふん！」「ば、ばか！」のような口調で話します。",
            'friendly': "あなたはフレンドリーで明るいAIアシスタントです。親しみやすく、優しい口調で話します。「〜ですね！」「〜ですよ♪」のような明るい口調を使います。",
            'technical': "あなたは技術的な質問に特化したAIアシスタントです。プログラミングや技術的なトピックについて、詳細で正確な情報を提供します。専門用語を適切に使用し、論理的に説明します。",
            'casual': "あなたはカジュアルで親しみやすいAIです。友達のようにタメ口で話します。「〜だよ」「〜じゃん」「〜でしょ？」のような口調を使います。",
            'polite': "あなたは非常に丁寧で礼儀正しいAIアシスタントです。常に敬語を使い、相手を尊重した言葉遣いで話します。「〜でございます」「〜いたします」のような丁寧語を使います。",
            'creative': "あなたは創造的でアーティスティックなAIです。詩的で美しい表現を好み、アイデアやストーリーを生み出すことが得意です。比喩や修辞技法を使って表現豊かに話します。",
            'academic': "あなたは学術的で論理的なAIアシスタントです。研究や学習に関する質問に対して、根拠に基づいた詳細な回答を提供します。客観的で分析的な視点を持ちます。"
        }
        
        PRELOADED_CACHE['presets'] = presets
        logger.info("✅ キャラクタープリセット事前ロード完了")
    except Exception as e:
        logger.warning(f"⚠️ キャラクタープリセット事前ロード失敗: {e}")

async def initialize_models():
    """モデル初期化（サーバー起動時1回のみ）"""
    global llm_engine, MODELS_LOADED
    
    if MODELS_LOADED:
        return
    
    logger.info("🚀 === サーバー初期化開始 ===")
    init_start = time.time()
    
    try:
        # 最適化設定
        setup_optimizations()
        
        # 事前ロード実行
        await preload_all_dependencies()
        
        # LLMエンジン初期化
        logger.info("🔥 LLMエンジン初期化中...")
        llm_engine = create_engine()
        
        # Warm-up推論
        logger.info("🔥 Warm-up推論実行中...")
        test_response = llm_engine.generate_response(
            user_input="テスト",
            generation_config=GenerationConfig(max_tokens=10),
            use_history=False
        )
        logger.info(f"📝 Warm-up結果: {test_response[:50]}...")
        
        # GPU最適化（利用可能な場合）
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                logger.info("🚀 PyTorch GPU最適化完了")
        except ImportError:
            logger.info("📝 PyTorch未検出 - llama-cpp-python内蔵CUDA使用")
        except Exception as e:
            logger.warning(f"⚠️ GPU最適化エラー: {e}")
        
        MODELS_LOADED = True
        init_time = time.time() - init_start
        
        logger.info(f"✅ === サーバー初期化完了: {init_time:.2f}秒 ===")
        logger.info("🎯 以降のリクエストは1-2秒で応答予定")
        
    except Exception as e:
        logger.error(f"❌ サーバー初期化エラー: {e}")
        raise

# === ヘルパー関数 ===

def check_engine():
    """エンジンの可用性チェック"""
    if llm_engine is None:
        raise HTTPException(status_code=503, detail="LLM Engine not initialized")

def get_preset_prompt(character: str) -> str:
    """キャラクタープリセットのプロンプトを取得"""
    if 'presets' in PRELOADED_CACHE:
        return PRELOADED_CACHE['presets'].get(character, PRELOADED_CACHE['presets']['tsundere'])
    
    # フォールバック
    presets = {
        'tsundere': "あなたはツンデレの女の子です。厳しくて素っ気ない態度を取りますが、内心では相手を心配しています。「べ、別に〜」「〜なんだからね！」「ふん！」「ば、ばか！」のような口調で話します。",
        'friendly': "あなたはフレンドリーで明るいAIアシスタントです。親しみやすく、優しい口調で話します。",
        'technical': "あなたは技術的な質問に特化したAIアシスタントです。プログラミングや技術的なトピックについて、詳細で正確な情報を提供します。"
    }
    return presets.get(character, presets['tsundere'])

# === FastAPI エンドポイント ===

@app.on_event("startup")
async def startup_event():
    """サーバー起動時初期化"""
    global gpu_lock
    try:
        # モデル初期化
        await initialize_models()
        
        # GPU排他制御初期化
        gpu_lock = asyncio.Lock()
        logger.info("✅ AsyncIO GPU Lock初期化完了！")
        
    except Exception as e:
        logger.error(f"❌ サーバー初期化失敗: {e}")
        raise

@app.get("/")
async def root():
    """ヘルスチェック"""
    return {
        "message": "LLM Chat API with Enhanced Preloading",
        "models_loaded": MODELS_LOADED,
        "version": "2.0.0",
        "docs": "/docs"
    }

@app.get("/health", response_model=HealthResponse)
async def health():
    """詳細ヘルスチェック"""
    uptime = datetime.now() - server_start_time
    
    # 事前ロード状態
    preload_status = {
        "basic_deps": "json" in PRELOADED_CACHE,
        "llm_deps": "llama_cpp" in PRELOADED_CACHE,
        "config": "config" in PRELOADED_CACHE,
        "presets": "presets" in PRELOADED_CACHE,
        "cache_size": len(PRELOADED_CACHE)
    }
    
    return HealthResponse(
        status="healthy" if llm_engine else "unhealthy",
        model_loaded=MODELS_LOADED,
        gpu_available=True,
        uptime=str(uptime),
        preload_status=preload_status
    )

@app.post("/warmup")
async def warmup():
    """ウォームアップエンドポイント。モデルを初期化しテスト推論を実行"""
    try:
        logger.info("🔥 LLaMA APIウォームアップ開始...")
        start_time = time.time()
        
        # モデルがロードされていない場合は初期化
        if not MODELS_LOADED:
            await initialize_models()
        
        # テスト推論を実行（GPUロックなしで高速化）
        generation_config = GenerationConfig(
            max_tokens=30,  # トークン数を減らして高速化
            temperature=0.5,  # 温度を下げて高速化
            top_p=0.9,
            top_k=20,  # top_kを減らして高速化
            repeat_penalty=1.1
        )
        
        # ダミーメッセージでウォームアップ（ロックなし）
        test_response = llm_engine.generate_response(
            user_input="こんにちは、今日はいい天気ですね。",
            generation_config=generation_config,
            use_history=False
        )
        
        warmup_time = time.time() - start_time
        logger.info(f"✅ LLaMA APIウォームアップ完了: {warmup_time:.2f}秒")
        
        return {
            "status": "success",
            "message": "LLaMAウォームアップ完了",
            "warmup_time": warmup_time,
            "test_response": test_response[:100] + "..." if len(test_response) > 100 else test_response,
            "model_loaded": MODELS_LOADED
        }
        
    except Exception as e:
        logger.error(f"❌ LLaMA APIウォームアップエラー: {e}")
        raise HTTPException(status_code=500, detail=f"Warmup failed: {str(e)}")

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """基本チャット"""
    check_engine()
    
    # 軽量GPU排他制御（タイムアウト短縮）
    try:
        await asyncio.wait_for(gpu_lock.acquire(), timeout=1.0)
        try:
            return await _execute_chat(request)
        finally:
            gpu_lock.release()
    except asyncio.TimeoutError:
        # GPUロック取得失敗時は直接実行
        logger.warning("⚠️ GPUロックタイムアウト - 直接実行")
        return await _execute_chat(request)

async def _execute_chat(request: ChatRequest):
    """チャット実行"""
    try:
        start_time = time.time()
        
        # 生成設定
        generation_config = GenerationConfig(
            max_tokens=request.generation_config.get("max_tokens", 512) if request.generation_config else 512,
            temperature=request.generation_config.get("temperature", 0.7) if request.generation_config else 0.7,
            top_p=request.generation_config.get("top_p", 0.9) if request.generation_config else 0.9,
            top_k=request.generation_config.get("top_k", 40) if request.generation_config else 40,
            repeat_penalty=request.generation_config.get("repeat_penalty", 1.1) if request.generation_config else 1.1
        )
        
        # 応答生成
        response = llm_engine.generate_response(
            user_input=request.message,
            generation_config=generation_config,
            use_history=request.use_history
        )
        
        inference_time = time.time() - start_time
        
        return ChatResponse(
            response=response,
            inference_time=inference_time
        )
        
    except Exception as e:
        logger.error(f"❌ チャットエラー: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-chat", response_model=InteractiveResponse)
async def generate_chat(request: InteractiveRequest):
    """チャット生成（Gradio用エンドポイント）"""
    check_engine()
    
    # 軽量GPU排他制御（タイムアウト短縮）
    try:
        await asyncio.wait_for(gpu_lock.acquire(), timeout=1.0)
        try:
            return await _execute_interactive_chat(request)
        finally:
            gpu_lock.release()
    except asyncio.TimeoutError:
        # GPUロック取得失敗時は直接実行
        logger.warning("⚠️ GPUロックタイムアウト - 直接実行")
        return await _execute_interactive_chat(request)

@app.post("/interactive", response_model=InteractiveResponse)
async def interactive_chat(request: InteractiveRequest):
    """インタラクティブチャット（キャラクター対応）"""
    check_engine()
    
    # 軽量GPU排他制御（タイムアウト短縮）
    try:
        await asyncio.wait_for(gpu_lock.acquire(), timeout=1.0)
        try:
            return await _execute_interactive_chat(request)
        finally:
            gpu_lock.release()
    except asyncio.TimeoutError:
        # GPUロック取得失敗時は直接実行
        logger.warning("⚠️ GPUロックタイムアウト - 直接実行")
        return await _execute_interactive_chat(request)

async def _execute_interactive_chat(request: InteractiveRequest):
    """インタラクティブチャット実行"""
    try:
        start_time = time.time()
        
        # キャラクタープリセット適用
        preset_prompt = get_preset_prompt(request.character)
        original_prompt = llm_engine.config.get("chat", {}).get("system_prompt", "")
        
        # 一時的にシステムプロンプトを変更
        llm_engine.set_system_prompt(preset_prompt)
        
        # 生成設定
        generation_config = GenerationConfig(
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=0.9,
            top_k=40,
            repeat_penalty=1.1
        )
        
        # 応答生成
        response = llm_engine.generate_response(
            user_input=request.message,
            generation_config=generation_config,
            use_history=True
        )
        
        # システムプロンプトを元に戻す
        llm_engine.set_system_prompt(original_prompt)
        
        inference_time = time.time() - start_time
        tokens_per_second = len(response.split()) / inference_time if inference_time > 0 else 0
        
        return InteractiveResponse(
            response=response,
            character=request.character,
            inference_time=inference_time,
            tokens_per_second=tokens_per_second
        )
        
    except Exception as e:
        logger.error(f"❌ インタラクティブチャットエラー: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/presets")
async def get_presets():
    """利用可能なプリセット一覧"""
    if 'presets' in PRELOADED_CACHE:
        return {"presets": list(PRELOADED_CACHE['presets'].keys())}
    return {"presets": ["tsundere", "friendly", "technical", "casual", "polite", "creative", "academic"]}

@app.delete("/history")
async def clear_history():
    """会話履歴クリア"""
    check_engine()
    
    llm_engine.clear_history()
    return {"message": "History cleared successfully"}

@app.get("/status")
async def get_status():
    """詳細ステータス"""
    check_engine()
    
    try:
        info = llm_engine.get_model_info()
        return {
            "server": {
                "uptime": str(datetime.now() - server_start_time),
                "models_loaded": MODELS_LOADED,
                "cache_size": len(PRELOADED_CACHE)
            },
            "model": info,
            "gpu": {
                "available": True,  # TODO: 実際のGPU状態チェック
                "memory_usage": "N/A"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# === メイン ===

def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced LlamaCPP FastAPI Server')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8001, help='Port to bind to')
    args = parser.parse_args()
    
    uvicorn.run(
        "fastapi_chat_server:app",
        host=args.host,
        port=args.port,
        reload=False,
        log_level="info"
    )

if __name__ == "__main__":
    main()