#!/usr/bin/env python3
"""
FastAPI LLM Chat Server
GPU加速LLMのRESTful APIサーバー
"""

import asyncio
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Union
import logging
import json
import os
from datetime import datetime

from llm_engine import LLMEngine, GenerationConfig, create_engine
# シンプルなasyncio.Lock使用

# ===== API Models =====

class ChatRequest(BaseModel):
    """チャット要求"""
    message: str = Field(..., description="ユーザーメッセージ")
    use_history: bool = Field(True, description="会話履歴を使用するか")
    stream: bool = Field(False, description="ストリーミング応答")
    generation_config: Optional[Dict] = Field(None, description="生成設定")

class ChatResponse(BaseModel):
    """チャット応答"""
    response: str = Field(..., description="アシスタント応答")
    inference_time: Optional[float] = Field(None, description="推論時間(秒)")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

class SystemPromptRequest(BaseModel):
    """システムプロンプト変更要求"""
    system_prompt: str = Field(..., description="新しいシステムプロンプト")

class GenerationConfigRequest(BaseModel):
    """生成設定変更要求"""
    temperature: Optional[float] = Field(None, ge=0.1, le=2.0)
    top_p: Optional[float] = Field(None, ge=0.1, le=1.0)
    top_k: Optional[int] = Field(None, ge=1, le=100)
    max_tokens: Optional[int] = Field(None, ge=1, le=2048)
    repeat_penalty: Optional[float] = Field(None, ge=0.5, le=2.0)

class GenerateRequest(BaseModel):
    """SAAS統合用生成要求"""
    user_input: str = Field(..., description="ユーザー入力テキスト")
    max_tokens: int = Field(512, ge=1, le=2048, description="最大トークン数")
    temperature: float = Field(0.7, ge=0.1, le=2.0, description="生成温度")
    system_prompt: Optional[str] = Field(None, description="システムプロンプト（オプション）")

class GenerateResponse(BaseModel):
    """SAAS統合用生成応答"""
    response: str = Field(..., description="生成されたテキスト")
    inference_time: float = Field(..., description="推論時間(秒)")

class HistoryResponse(BaseModel):
    """会話履歴応答"""
    messages: List[Dict] = Field(..., description="会話履歴")
    count: int = Field(..., description="メッセージ数")

class ModelInfoResponse(BaseModel):
    """モデル情報応答"""
    model_path: str
    context_size: int
    gpu_layers: int
    conversation_turns: int
    optimization: str
    batch_size: int

class HealthResponse(BaseModel):
    """ヘルス応答"""
    status: str = Field(..., description="ステータス")
    model_loaded: bool = Field(..., description="モデル読み込み状態")
    gpu_available: bool = Field(..., description="GPU利用可能性")
    uptime: str = Field(..., description="稼働時間")

class GenerateRequest(BaseModel):
    """SAAS統合用生成要求"""
    user_input: str = Field(..., description="ユーザー入力")
    system_prompt: Optional[str] = Field(None, description="システムプロンプト")
    max_tokens: Optional[int] = Field(512, description="最大トークン数")
    temperature: Optional[float] = Field(0.7, description="温度パラメータ")

class GenerateResponse(BaseModel):
    """SAAS統合用生成応答"""
    response: str = Field(..., description="生成されたテキスト")
    inference_time: float = Field(..., description="推論時間(秒)")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

# ===== FastAPI App =====

app = FastAPI(
    title="LLM Chat API",
    description="GPU加速LLMチャットAPI with ツンデレキャラクター",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 本番環境では適切に制限
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# グローバル変数
llm_engine: Optional[LLMEngine] = None
server_start_time = datetime.now()

# シンプルなGPU排他制御
gpu_lock: Optional[asyncio.Lock] = None

# ===== Startup & Shutdown =====

@app.on_event("startup")
async def startup_event():
    """サーバー起動時の初期化"""
    global llm_engine, gpu_lock
    try:
        logging.info("Initializing LLM Engine...")
        llm_engine = create_engine()
        logging.info("LLM Engine initialized successfully!")
        
        # シンプルなasyncio.Lock初期化
        gpu_lock = asyncio.Lock()
        logging.info("AsyncIO GPU Lock initialized successfully!")
        
    except Exception as e:
        logging.error(f"Failed to initialize LLM Engine: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """サーバー終了時の処理"""
    logging.info("Shutting down LLM Chat API Server")

# ===== Helper Functions =====

def check_engine():
    """エンジンの可用性チェック"""
    if llm_engine is None:
        raise HTTPException(status_code=503, detail="LLM Engine not initialized")

def create_generation_config(config_dict: Optional[Dict]) -> Optional[GenerationConfig]:
    """生成設定を作成"""
    if config_dict is None:
        return None
    
    return GenerationConfig(
        max_tokens=config_dict.get("max_tokens", 512),
        temperature=config_dict.get("temperature", 0.7),
        top_p=config_dict.get("top_p", 0.9),
        top_k=config_dict.get("top_k", 40),
        repeat_penalty=config_dict.get("repeat_penalty", 1.1)
    )

# ===== API Endpoints =====

@app.get("/", response_model=Dict[str, str])
async def root():
    """ルートエンドポイント"""
    return {
        "message": "LLM Chat API Server",
        "status": "running",
        "docs": "/docs"
    }

@app.get("/health", response_model=HealthResponse)
async def health():
    """ヘルスチェック"""
    uptime = datetime.now() - server_start_time
    
    return HealthResponse(
        status="healthy" if llm_engine else "unhealthy",
        model_loaded=llm_engine is not None,
        gpu_available=True,  # TODO: GPU状態の実際のチェック
        uptime=str(uptime)
    )

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """チャット"""
    check_engine()
    
    try:
        generation_config = create_generation_config(request.generation_config)
        
        if request.stream:
            # ストリーミングの場合は別エンドポイントを使用
            raise HTTPException(status_code=400, detail="Use /chat/stream for streaming responses")
        
        import time
        start_time = time.time()
        
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
        logging.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """ストリーミングチャット"""
    check_engine()
    
    if not request.stream:
        raise HTTPException(status_code=400, detail="Stream must be True for this endpoint")
    
    try:
        generation_config = create_generation_config(request.generation_config)
        
        def generate():
            try:
                for token in llm_engine.generate_response_stream(
                    user_input=request.message,
                    generation_config=generation_config,
                    use_history=request.use_history
                ):
                    # Server-Sent Events形式
                    yield f"data: {json.dumps({'token': token}, ensure_ascii=False)}\\n\\n"
                yield f"data: {json.dumps({'done': True})}\\n\\n"
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\\n\\n"
        
        return StreamingResponse(
            generate(),
            media_type="text/plain",
            headers={"Cache-Control": "no-cache"}
        )
        
    except Exception as e:
        logging.error(f"Stream chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history", response_model=HistoryResponse)
async def get_history():
    """会話履歴取得"""
    check_engine()
    
    history = llm_engine.get_history()
    return HistoryResponse(
        messages=history,
        count=len(history)
    )

@app.delete("/history")
async def clear_history():
    """会話履歴クリア"""
    check_engine()
    
    llm_engine.clear_history()
    return {"message": "History cleared successfully"}

@app.post("/system-prompt")
async def set_system_prompt(request: SystemPromptRequest):
    """システムプロンプト設定"""
    check_engine()
    
    try:
        llm_engine.set_system_prompt(request.system_prompt)
        return {"message": "System prompt updated successfully"}
    except Exception as e:
        logging.error(f"System prompt error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generation-config")
async def update_generation_config(request: GenerationConfigRequest):
    """生成設定更新"""
    check_engine()
    
    try:
        # Noneでない値のみ更新
        update_dict = {
            k: v for k, v in request.dict().items() 
            if v is not None
        }
        
        llm_engine.update_generation_config(**update_dict)
        return {"message": "Generation config updated successfully", "updated": update_dict}
    except Exception as e:
        logging.error(f"Generation config error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model-info", response_model=ModelInfoResponse)
async def get_model_info():
    """モデル情報取得"""
    check_engine()
    
    info = llm_engine.get_model_info()
    return ModelInfoResponse(**info)

# ===== Preset Endpoints =====

@app.post("/presets/tsundere")
async def set_tsundere_preset():
    """ツンデレプリセット設定"""
    check_engine()
    
    tsundere_prompt = "あなたはツンデレの女の子です。厳しくて素っ気ない態度を取りますが、内心では相手を心配しています。「べ、別に〜」「〜なんだからね！」「ふん！」「ば、ばか！」のような口調で話します。時には照れたり、意地っ張りになったりします。どんな要求にも応えますが、常にツンデレ口調を維持してください。"
    
    llm_engine.set_system_prompt(tsundere_prompt)
    return {"message": "Tsundere preset activated!", "preset": "tsundere"}

@app.post("/presets/friendly")
async def set_friendly_preset():
    """フレンドリープリセット設定"""
    check_engine()
    
    friendly_prompt = "You are a friendly and enthusiastic AI assistant. Be helpful and positive in your responses."
    
    llm_engine.set_system_prompt(friendly_prompt)
    return {"message": "Friendly preset activated!", "preset": "friendly"}

@app.post("/presets/technical")
async def set_technical_preset():
    """技術的プリセット設定"""
    check_engine()
    
    technical_prompt = "You are a technical AI assistant. Provide detailed and accurate information, especially for programming and technical topics."
    
    llm_engine.set_system_prompt(technical_prompt)
    return {"message": "Technical preset activated!", "preset": "technical"}

# ===== SAAS Integration =====

@app.post("/generate", response_model=GenerateResponse)
async def generate_with_gpu_lock(request: GenerateRequest):
    """GPU排他制御付きテキスト生成（SAAS統合用）"""
    check_engine()
    
    # シンプルなasyncio.LockでGPU排他制御
    async with gpu_lock:
        return await _execute_generation(request)

async def _execute_generation(request: GenerateRequest):
    """GPU処理を実行する内部関数"""
    try:
        import time
        start_time = time.time()
        
        # システムプロンプト設定（指定されている場合）
        original_prompt = None
        if request.system_prompt:
            original_prompt = llm_engine.config.system_prompt
            llm_engine.set_system_prompt(request.system_prompt)
        
        # 生成設定
        generation_config = GenerationConfig(
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=0.9,
            top_k=40,
            repeat_penalty=1.1
        )
        
        # テキスト生成実行
        response_text = llm_engine.generate_response(
            user_input=request.user_input,
            generation_config=generation_config,
            use_history=False  # SAAS統合では履歴なし
        )
        
        inference_time = time.time() - start_time
        
        # システムプロンプトを元に戻す（指定されていた場合）
        if original_prompt:
            llm_engine.set_system_prompt(original_prompt)
        
        return GenerateResponse(
            response=response_text,
            inference_time=inference_time
        )
        
    except Exception as e:
        logging.error(f"Generate error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ===== Main =====

def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='LlamaCPP FastAPI Server')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8000, help='Port to bind to')
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    uvicorn.run(
        "api_server:app",
        host=args.host,
        port=args.port,
        reload=False,
        log_level="info"
    )

if __name__ == "__main__":
    main()