#!/usr/bin/env python3
"""
Custom FastAPI Implementation for SadTalker
More flexible and customizable than the original main.py
"""

import os
import sys
import time
import base64
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any
import logging

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Add SadTalker to path
sys.path.append('/home/SadTalker')
sys.path.append('/home/SadTalker/src')

from src.utils.preprocess import CropAndExtract
from src.test_audio2coeff import Audio2Coeff  
from src.facerender.animate import AnimateFromCoeff
from src.generate_batch import get_data
from src.generate_facerender_batch import get_facerender_data
from src.utils.init_path import init_path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model instances
preprocess_model = None
audio_to_coeff = None
animate_from_coeff = None

class TextRequest(BaseModel):
    text: str
    voice_preset: Optional[str] = "default"
    enhance: Optional[bool] = True
    output_format: Optional[str] = "mp4"

class AudioRequest(BaseModel):
    enhance: Optional[bool] = True
    output_format: Optional[str] = "mp4"

class StatusResponse(BaseModel):
    status: str
    message: str
    models_loaded: bool
    processing: bool

app = FastAPI(
    title="SadTalker Custom API",
    description="Custom FastAPI implementation for SadTalker with enhanced features",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def init_models():
    """Initialize all SadTalker models"""
    global preprocess_model, audio_to_coeff, animate_from_coeff
    
    try:
        logger.info("🚀 Initializing SadTalker models...")
        
        # Initialize paths
        device = "cuda"
        
        # Initialize preprocessing model
        logger.info("📋 Loading preprocessing model...")
        preprocess_model = CropAndExtract()
        
        # Initialize audio to coefficient model  
        logger.info("🎵 Loading audio to coefficient model...")
        audio_to_coeff = Audio2Coeff(device)
        
        # Initialize animation model
        logger.info("🎬 Loading animation model...")
        animate_from_coeff = AnimateFromCoeff(device)
        
        logger.info("✅ All models loaded successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize models: {e}")
        return False

async def generate_speech_from_text(text: str, voice_preset: str = "default") -> str:
    """Generate speech from text using TTS service"""
    try:
        import requests
        
        # Call TTS service (assuming it's running on port 9566)
        tts_url = "http://localhost:9566/tts"
        data = {"text": text, "voice": voice_preset}
        
        response = requests.post(tts_url, json=data, timeout=30)
        response.raise_for_status()
        
        # Save audio file
        audio_path = f"/tmp/tts_{int(time.time())}.wav"
        with open(audio_path, "wb") as f:
            f.write(response.content)
            
        return audio_path
        
    except Exception as e:
        logger.error(f"❌ TTS generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {e}")

def process_sadtalker(audio_path: str, image_path: str, enhance: bool = True) -> str:
    """Process SadTalker generation"""
    try:
        logger.info("🎭 Starting SadTalker processing...")
        
        # Set parameters
        size = 256
        expression_scale = 1.0
        crop_size = (1, 0, 0, 0)  # top, bottom, left, right
        
        # Step 1: Process input image
        logger.info("📸 Processing input image...")
        processed_dir = preprocess_model.generate(
            image_path, crop_size, size, crop_size
        )
        
        # Step 2: Generate coefficients from audio
        logger.info("🎵 Generating coefficients from audio...")
        batch = get_data(processed_dir, audio_path, device="cuda")
        coeff_path, crop_info = audio_to_coeff.generate(
            batch, processed_dir, expression_scale
        )
        
        # Step 3: Generate animation
        logger.info("🎬 Generating animation...")
        data = get_facerender_data(
            coeff_path, crop_info, processed_dir, 
            still_mode=True, use_enhancer=enhance,
            preprocess="crop"
        )
        
        result_path = animate_from_coeff.generate(
            data, processed_dir, pic_path=processed_dir, 
            crop_info=crop_info, enhancer_strength=0.5 if enhance else 0,
            device="cuda"
        )
        
        logger.info(f"✅ SadTalker processing complete: {result_path}")
        return result_path
        
    except Exception as e:
        logger.error(f"❌ SadTalker processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {e}")

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    logger.info("🚀 Starting SadTalker Custom API...")
    success = init_models()
    if not success:
        logger.error("❌ Failed to initialize models, API may not work properly")

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "SadTalker Custom API",
        "version": "1.0.0",
        "description": "Custom FastAPI implementation for SadTalker",
        "endpoints": {
            "/generate/text": "Generate talking video from text input",
            "/generate/audio": "Generate talking video from audio file", 
            "/generate/video": "Generate talking video from audio and image files",
            "/status": "Check API status",
            "/health": "Health check"
        }
    }

@app.get("/status", response_model=StatusResponse)
async def get_status():
    """Get API status"""
    models_loaded = all([
        preprocess_model is not None,
        audio_to_coeff is not None, 
        animate_from_coeff is not None
    ])
    
    return StatusResponse(
        status="running",
        message="API is operational", 
        models_loaded=models_loaded,
        processing=False
    )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": time.time()}

@app.post("/generate/text")
async def generate_from_text(request: TextRequest):
    """Generate talking video from text input"""
    try:
        # Check if models are loaded
        if not all([preprocess_model, audio_to_coeff, animate_from_coeff]):
            raise HTTPException(status_code=503, detail="Models not loaded")
        
        logger.info(f"📝 Processing text request: {request.text[:50]}...")
        
        # Generate speech from text
        audio_path = await generate_speech_from_text(request.text, request.voice_preset)
        
        # Use default image
        default_image = "/home/SadTalker/sadtalker_default.jpeg"
        if not os.path.exists(default_image):
            raise HTTPException(status_code=404, detail="Default image not found")
        
        # Process with SadTalker
        result_path = process_sadtalker(audio_path, default_image, request.enhance)
        
        # Return result
        if request.output_format == "base64":
            with open(result_path, "rb") as f:
                video_data = base64.b64encode(f.read()).decode()
            return {"video_base64": video_data, "path": result_path}
        else:
            return FileResponse(result_path, media_type="video/mp4")
            
    except Exception as e:
        logger.error(f"❌ Text generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate/audio")
async def generate_from_audio(
    audio: UploadFile = File(...),
    enhance: bool = Form(True),
    output_format: str = Form("mp4")
):
    """Generate talking video from uploaded audio file"""
    try:
        # Check if models are loaded
        if not all([preprocess_model, audio_to_coeff, animate_from_coeff]):
            raise HTTPException(status_code=503, detail="Models not loaded")
        
        logger.info(f"🎵 Processing audio file: {audio.filename}")
        
        # Save uploaded audio
        audio_path = f"/tmp/upload_{int(time.time())}_{audio.filename}"
        with open(audio_path, "wb") as f:
            content = await audio.read()
            f.write(content)
        
        # Use default image
        default_image = "/home/SadTalker/sadtalker_default.jpeg"
        if not os.path.exists(default_image):
            raise HTTPException(status_code=404, detail="Default image not found")
        
        # Process with SadTalker
        result_path = process_sadtalker(audio_path, default_image, enhance)
        
        # Return result
        if output_format == "base64":
            with open(result_path, "rb") as f:
                video_data = base64.b64encode(f.read()).decode()
            return {"video_base64": video_data, "path": result_path}
        else:
            return FileResponse(result_path, media_type="video/mp4")
            
    except Exception as e:
        logger.error(f"❌ Audio generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate/video")
async def generate_from_files(
    audio: UploadFile = File(...),
    image: UploadFile = File(...),
    enhance: bool = Form(True),
    output_format: str = Form("mp4")
):
    """Generate talking video from uploaded audio and image files"""
    try:
        # Check if models are loaded
        if not all([preprocess_model, audio_to_coeff, animate_from_coeff]):
            raise HTTPException(status_code=503, detail="Models not loaded")
        
        logger.info(f"🎬 Processing files: {audio.filename}, {image.filename}")
        
        # Save uploaded files
        timestamp = int(time.time())
        audio_path = f"/tmp/upload_{timestamp}_{audio.filename}"
        image_path = f"/tmp/upload_{timestamp}_{image.filename}"
        
        with open(audio_path, "wb") as f:
            content = await audio.read()
            f.write(content)
            
        with open(image_path, "wb") as f:
            content = await image.read()
            f.write(content)
        
        # Process with SadTalker
        result_path = process_sadtalker(audio_path, image_path, enhance)
        
        # Return result
        if output_format == "base64":
            with open(result_path, "rb") as f:
                video_data = base64.b64encode(f.read()).decode()
            return {"video_base64": video_data, "path": result_path}
        else:
            return FileResponse(result_path, media_type="video/mp4")
            
    except Exception as e:
        logger.error(f"❌ Video generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "custom_api:app",
        host="0.0.0.0",
        port=10364,
        reload=False,
        log_level="info"
    )