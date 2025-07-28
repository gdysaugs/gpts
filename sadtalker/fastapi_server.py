#!/usr/bin/env python3
"""
SadTalker FastAPI Server - モデル常駐型高速APIサーバー
事前ロード型エンジンによる超高速動画生成
"""

import os
import asyncio
import tempfile
import aiofiles
from pathlib import Path
from typing import Optional, List
from contextlib import asynccontextmanager
import time
import uuid

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from sadtalker_engine import SadTalkerEngine, SadTalkerConfig

# グローバル変数
sadtalker_engine: Optional[SadTalkerEngine] = None
processing_queue = asyncio.Queue(maxsize=5)  # 同時処理制限
active_requests = 0

class GenerationRequest(BaseModel):
    """動画生成リクエスト"""
    quality: str = Field(default="fast", pattern="^(fast|high)$")
    fp16: bool = Field(default=False)
    expression_scale: float = Field(default=1.0, ge=0.0, le=2.0)
    still_mode: bool = Field(default=True)
    yaw: Optional[float] = Field(default=None, ge=-30, le=30)
    pitch: Optional[float] = Field(default=None, ge=-20, le=20)
    roll: Optional[float] = Field(default=None, ge=-15, le=15)

class GenerationResponse(BaseModel):
    """動画生成レスポンス"""
    success: bool
    message: str
    processing_time: Optional[float] = None
    config: Optional[dict] = None
    error: Optional[str] = None
    download_url: Optional[str] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """アプリケーションライフサイクル管理"""
    global sadtalker_engine
    
    print("🚀 SadTalker FastAPIサーバー起動中...")
    
    # 起動時: モデル事前ロード
    try:
        sadtalker_engine = SadTalkerEngine()
        print("✅ SadTalkerエンジン常駐準備完了")
    except Exception as e:
        print(f"❌ エンジン初期化失敗: {e}")
        raise
    
    yield
    
    # 終了時: クリーンアップ
    print("🔄 SadTalkerサーバー終了中...")

# FastAPIアプリ作成
app = FastAPI(
    title="SadTalker API Server",
    description="モデル常駐型超高速動画生成API",
    version="2.0.0",
    lifespan=lifespan
)

# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 静的ファイル配信
os.makedirs("output", exist_ok=True)
app.mount("/output", StaticFiles(directory="output"), name="output")

@app.get("/", response_class=HTMLResponse)
async def web_ui():
    """WebUI - ブラウザインターフェース"""
    html_content = """
    <!DOCTYPE html>
    <html lang="ja">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>SadTalker - 超高速動画生成</title>
        <style>
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; }
            .container { max-width: 800px; margin: 0 auto; background: rgba(255,255,255,0.1); padding: 30px; border-radius: 15px; backdrop-filter: blur(10px); }
            h1 { text-align: center; margin-bottom: 30px; font-size: 2.5em; text-shadow: 2px 2px 4px rgba(0,0,0,0.3); }
            .form-group { margin-bottom: 20px; }
            label { display: block; margin-bottom: 8px; font-weight: bold; }
            input, select { width: 100%; padding: 12px; border: none; border-radius: 8px; font-size: 16px; }
            button { background: linear-gradient(45deg, #FF6B6B, #4ECDC4); color: white; padding: 15px 30px; border: none; border-radius: 8px; font-size: 18px; cursor: pointer; width: 100%; margin-top: 20px; }
            button:hover { transform: translateY(-2px); box-shadow: 0 5px 15px rgba(0,0,0,0.2); }
            .status { margin-top: 20px; padding: 15px; border-radius: 8px; background: rgba(255,255,255,0.2); }
            .progress { display: none; text-align: center; margin-top: 20px; }
            .result { margin-top: 20px; text-align: center; }
            .options { display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-top: 15px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎭 SadTalker - 超高速動画生成</h1>
            <p style="text-align: center; margin-bottom: 30px;">モデル常駐型 - 事前ロード済みで超高速処理</p>
            
            <form id="uploadForm" enctype="multipart/form-data">
                <div class="form-group">
                    <label for="image">📸 画像ファイル (JPG/PNG):</label>
                    <input type="file" id="image" name="image" accept="image/*" required>
                </div>
                
                <div class="form-group">
                    <label for="audio">🎵 音声ファイル (WAV/MP3):</label>
                    <input type="file" id="audio" name="audio" accept="audio/*" required>
                </div>
                
                <div class="options">
                    <div class="form-group">
                        <label for="quality">🔥 品質モード:</label>
                        <select id="quality" name="quality">
                            <option value="fast">⚡ 高速モード</option>
                            <option value="high">🔥 高画質 (GFPGAN)</option>
                        </select>
                    </div>
                    
                    <div class="form-group">
                        <label for="fp16">🚀 FP16最適化:</label>
                        <select id="fp16" name="fp16">
                            <option value="false">標準精度</option>
                            <option value="true">FP16高速化</option>
                        </select>
                    </div>
                    
                    <div class="form-group">
                        <label for="still_mode">🎯 静止モード:</label>
                        <select id="still_mode" name="still_mode">
                            <option value="true">有効（推奨）</option>
                            <option value="false">無効</option>
                        </select>
                    </div>
                    
                    <div class="form-group">
                        <label for="expression_scale">🎭 表情強度 (0.0-2.0):</label>
                        <input type="number" id="expression_scale" name="expression_scale" value="1.0" min="0" max="2" step="0.1">
                    </div>
                </div>
                
                <button type="submit">🚀 超高速動画生成開始</button>
            </form>
            
            <div id="progress" class="progress">
                <h3>⚡ 処理中...</h3>
                <p>モデル常駐により高速処理中です</p>
            </div>
            
            <div id="status" class="status" style="display: none;"></div>
            <div id="result" class="result"></div>
        </div>

        <script>
            document.getElementById('uploadForm').addEventListener('submit', async function(e) {
                e.preventDefault();
                
                const formData = new FormData();
                formData.append('image', document.getElementById('image').files[0]);
                formData.append('audio', document.getElementById('audio').files[0]);
                formData.append('quality', document.getElementById('quality').value);
                formData.append('fp16', document.getElementById('fp16').value);
                formData.append('still_mode', document.getElementById('still_mode').value);
                formData.append('expression_scale', document.getElementById('expression_scale').value);
                
                document.getElementById('progress').style.display = 'block';
                document.getElementById('status').style.display = 'none';
                document.getElementById('result').innerHTML = '';
                
                try {
                    const response = await fetch('/generate', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const result = await response.json();
                    document.getElementById('progress').style.display = 'none';
                    
                    if (result.success) {
                        document.getElementById('status').innerHTML = `
                            <h3>✅ 生成成功!</h3>
                            <p>処理時間: ${result.processing_time?.toFixed(2)}秒</p>
                            <p>品質: ${result.config?.quality} ${result.config?.fp16 ? '+ FP16' : ''}</p>
                        `;
                        document.getElementById('result').innerHTML = `
                            <video controls width="100%" style="max-width: 500px; border-radius: 10px;">
                                <source src="${result.download_url}" type="video/mp4">
                            </video>
                            <br><br>
                            <a href="${result.download_url}" download style="color: #4ECDC4; text-decoration: none; font-weight: bold;">📥 動画ダウンロード</a>
                        `;
                    } else {
                        document.getElementById('status').innerHTML = `
                            <h3>❌ 生成失敗</h3>
                            <p>エラー: ${result.error}</p>
                        `;
                    }
                    document.getElementById('status').style.display = 'block';
                    
                } catch (error) {
                    document.getElementById('progress').style.display = 'none';
                    document.getElementById('status').innerHTML = `
                        <h3>❌ 通信エラー</h3>
                        <p>${error.message}</p>
                    `;
                    document.getElementById('status').style.display = 'block';
                }
            });
        </script>
    </body>
    </html>
    """
    return html_content

@app.post("/generate", response_model=GenerationResponse)
async def generate_video(
    background_tasks: BackgroundTasks,
    image: UploadFile = File(..., description="入力画像 (JPG/PNG)"),
    audio: UploadFile = File(..., description="入力音声 (WAV/MP3)"),
    quality: str = Form(default="fast", pattern="^(fast|high)$"),
    fp16: bool = Form(default=False),
    expression_scale: float = Form(default=1.0, ge=0.0, le=2.0),
    still_mode: bool = Form(default=True),
    yaw: Optional[float] = Form(default=None, ge=-30, le=30),
    pitch: Optional[float] = Form(default=None, ge=-20, le=20),
    roll: Optional[float] = Form(default=None, ge=-15, le=15)
):
    """
    動画生成API - モデル常駐による超高速処理
    """
    global active_requests
    
    if active_requests >= 3:  # 同時処理制限
        raise HTTPException(status_code=429, detail="サーバーが混雑しています。しばらく待ってから再試行してください。")
    
    active_requests += 1
    start_time = time.time()
    
    try:
        # ファイルタイプ検証
        if not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="画像ファイルが必要です")
        if not audio.content_type.startswith('audio/'):
            raise HTTPException(status_code=400, detail="音声ファイルが必要です")
        
        # ファイルサイズ制限 (50MB)
        max_size = 50 * 1024 * 1024
        if image.size > max_size or audio.size > max_size:
            raise HTTPException(status_code=413, detail="ファイルサイズが大きすぎます (最大50MB)")
        
        # 設定作成
        config = SadTalkerConfig(
            quality=quality,
            fp16=fp16,
            expression_scale=expression_scale,
            still_mode=still_mode,
            yaw=yaw,
            pitch=pitch,
            roll=roll
        )
        
        # 一時ファイル保存
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # ファイル保存
            image_path = temp_path / f"input_image_{uuid.uuid4().hex[:8]}{Path(image.filename).suffix}"
            audio_path = temp_path / f"input_audio_{uuid.uuid4().hex[:8]}{Path(audio.filename).suffix}"
            
            async with aiofiles.open(image_path, 'wb') as f:
                content = await image.read()
                await f.write(content)
            
            async with aiofiles.open(audio_path, 'wb') as f:
                content = await audio.read()
                await f.write(content)
            
            # 🚀 動画生成（事前ロード済みモデル使用）
            result = sadtalker_engine.generate_video(
                str(image_path), 
                str(audio_path), 
                config
            )
            
            processing_time = time.time() - start_time
            
            if result["success"]:
                # 出力ファイル保存
                output_filename = f"sadtalker_{uuid.uuid4().hex[:8]}.mp4"
                output_path = Path("output") / output_filename
                
                # ファイルコピー
                import shutil
                shutil.copy2(result["video_path"], output_path)
                
                # クリーンアップをバックグラウンドで実行
                background_tasks.add_task(cleanup_old_files)
                
                return GenerationResponse(
                    success=True,
                    message="動画生成成功",
                    processing_time=processing_time,
                    config=config.__dict__,
                    download_url=f"/output/{output_filename}"
                )
            else:
                return GenerationResponse(
                    success=False,
                    message="動画生成失敗",
                    processing_time=processing_time,
                    error=result["error"]
                )
    
    except Exception as e:
        processing_time = time.time() - start_time
        return GenerationResponse(
            success=False,
            message="サーバーエラー",
            processing_time=processing_time,
            error=str(e)
        )
    
    finally:
        active_requests -= 1

@app.get("/status")
async def get_status():
    """サーバー状態確認"""
    if sadtalker_engine is None:
        return {"status": "initializing", "message": "エンジン初期化中"}
    
    engine_status = sadtalker_engine.get_status()
    return {
        "status": "ready" if engine_status["models_loaded"] else "error",
        "active_requests": active_requests,
        "engine_status": engine_status,
        "message": "サーバー正常稼働中" if engine_status["models_loaded"] else "エンジンエラー"
    }

async def cleanup_old_files():
    """古いファイルクリーンアップ（バックグラウンドタスク）"""
    output_dir = Path("output")
    if not output_dir.exists():
        return
    
    import time
    current_time = time.time()
    
    # 1時間以上古いファイルを削除
    for file_path in output_dir.glob("*.mp4"):
        if current_time - file_path.stat().st_mtime > 3600:
            try:
                file_path.unlink()
                print(f"🗑️ 古いファイル削除: {file_path.name}")
            except:
                pass

if __name__ == "__main__":
    import uvicorn
    print("🚀 SadTalker FastAPIサーバー起動")
    print("📡 WebUI: http://localhost:8000")
    print("📚 API Docs: http://localhost:8000/docs")
    
    uvicorn.run(
        "fastapi_server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # プロダクション用
        workers=1      # GPU共有のため1ワーカー
    )