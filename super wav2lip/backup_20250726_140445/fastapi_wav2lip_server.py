#!/usr/bin/env python3
"""
Wav2Lip FastAPI Server
Port 8002でWav2Lip処理を提供
"""

import os
import sys
import tempfile
import subprocess
import traceback
import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# ログ設定
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# FastAPIアプリ作成
app = FastAPI(
    title="Super Wav2Lip API",
    description="AI Lip-sync Video Generation API",
    version="1.0.0"
)

# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 作業ディレクトリ
WORK_DIR = Path("/app")
INPUT_DIR = WORK_DIR / "input"
OUTPUT_DIR = WORK_DIR / "output"
TEMP_DIR = WORK_DIR / "temp"
MODELS_DIR = WORK_DIR / "models"

# ディレクトリ作成
for dir_path in [INPUT_DIR, OUTPUT_DIR, TEMP_DIR]:
    dir_path.mkdir(exist_ok=True)

# グローバル例外ハンドラー
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    logger.error(f"Traceback: {traceback.format_exc()}")
    return HTTPException(
        status_code=500,
        detail=f"Internal server error: {str(exc)}"
    )

@app.get("/")
async def root():
    """ルートエンドポイント"""
    return {"message": "Super Wav2Lip FastAPI Server", "status": "running"}

@app.get("/health")
async def health_check():
    """ヘルスチェック"""
    return {"status": "healthy", "service": "wav2lip"}

@app.post("/generate-lipsync")
async def generate_lipsync(
    video_file: UploadFile = File(...),
    audio_file: UploadFile = File(...),
    enhancer: str = Form("none"),
    batch_size: int = Form(8),
    hq_output: bool = Form(False)
):
    """
    口パク動画生成エンドポイント
    """
    try:
        logger.info(f"Starting lipsync generation - enhancer: {enhancer}, batch_size: {batch_size}, hq_output: {hq_output}")
        
        # 一時ファイル保存
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as video_temp:
            video_content = await video_file.read()
            video_temp.write(video_content)
            video_path = video_temp.name
        logger.info(f"Video saved to: {video_path}")
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as audio_temp:
            audio_content = await audio_file.read()
            audio_temp.write(audio_content)
            audio_path = audio_temp.name
        logger.info(f"Audio saved to: {audio_path}")
        
        # 出力ファイルパス
        output_path = TEMP_DIR / f"result_{os.getpid()}.mp4"
        logger.info(f"Output path: {output_path}")
        
        # Wav2Lip実行（実際の推論スクリプトを使用）
        cmd = [
            "python", "inference_onnxModel.py",
            "--checkpoint_path", "/app/original_source/checkpoints/wav2lip_gan.onnx",
            "--face", video_path,
            "--audio", audio_path,
            "--outfile", str(output_path),
            "--enhancer", enhancer
        ]
        
        if hq_output:
            cmd.append("--hq_output")
        
        logger.info(f"Running command: {' '.join(cmd)}")
        logger.info(f"Working directory: /app/original_source")
        
        # headless環境のための環境変数設定
        env = os.environ.copy()
        env['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'
        env['OPENCV_VIDEOIO_PRIORITY_INTEL_MFX'] = '0'
        # 完全ヘッドレス化 - Qt無効化
        env['QT_QPA_PLATFORM'] = 'offscreen'
        env['QT_LOGGING_RULES'] = '*.debug=false'
        env['MPLBACKEND'] = 'Agg'  # matplotlib headless backend
        env['OPENCV_LOG_LEVEL'] = 'ERROR'
        # 完全にDISPLAY削除
        if 'DISPLAY' in env:
            del env['DISPLAY']
        
        # 仮想ディスプレイなしで直接実行（完全ヘッドレス）
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, cwd="/app/original_source", env=env)
        
        logger.info(f"Command return code: {result.returncode}")
        logger.info(f"Command stdout: {result.stdout}")
        logger.info(f"Command stderr: {result.stderr}")
        
        # クリーンアップ
        os.unlink(video_path)
        os.unlink(audio_path)
        
        if result.returncode == 0 and output_path.exists():
            return FileResponse(
                path=str(output_path),
                media_type="video/mp4",
                filename="lipsync_result.mp4"
            )
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Wav2Lip processing failed: {result.stderr}"
            )
            
    except subprocess.TimeoutExpired as e:
        logger.error(f"Subprocess timeout: {e}")
        # クリーンアップ
        if 'video_path' in locals():
            os.unlink(video_path)
        if 'audio_path' in locals():
            os.unlink(audio_path)
        
        raise HTTPException(
            status_code=408,
            detail="Processing timeout (5 minutes exceeded)"
        )
    except Exception as e:
        logger.error(f"Exception in generate_lipsync: {e}")
        logger.error(f"Exception traceback: {traceback.format_exc()}")
        
        # クリーンアップ
        if 'video_path' in locals():
            try:
                os.unlink(video_path)
            except:
                pass
        if 'audio_path' in locals():
            try:
                os.unlink(audio_path)
            except:
                pass
        
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@app.get("/models")
async def list_models():
    """利用可能モデル一覧"""
    models = []
    if MODELS_DIR.exists():
        for model_file in MODELS_DIR.glob("*.pth"):
            models.append(model_file.name)
        for model_file in MODELS_DIR.glob("*.onnx"):
            models.append(model_file.name)
    
    return {"models": models}

if __name__ == "__main__":
    print("🎬 Starting Wav2Lip FastAPI Server on port 8002...")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8002,
        log_level="info"
    )