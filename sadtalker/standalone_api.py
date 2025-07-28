#!/usr/bin/env python3
"""
SadTalker スタンドアロンAPI
音声ファイルと画像ファイルを直接アップロードして口パク動画を生成
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.responses import FileResponse
import os
import sys
import shutil
from time import strftime
from pathlib import Path
import tempfile

# SadTalkerモジュールをインポート
sys.path.append('/home/SadTalker/src')
try:
    from src.utils.preprocess import CropAndExtract
    from src.test_audio2coeff import Audio2Coeff  
    from src.facerender.animate_onnx import AnimateFromCoeff
    from src.generate_batch import get_data
    from src.generate_facerender_batch import get_facerender_data
    from src.utils.init_path import init_path
except ImportError as e:
    print(f"SadTalkerモジュールのインポートに失敗: {e}")
    print("Dockerコンテナ内で実行してください")

# 設定
CHECKPOINTS_DIR = "./checkpoints"
CONFIG_DIR = "./src/config"
RESULTS_DIR = "./results"

# SadTalker初期化
try:
    sadtalker_paths = init_path(CHECKPOINTS_DIR, CONFIG_DIR, "256", False, "full")
    preprocess_model = CropAndExtract(sadtalker_paths, "cuda")
    audio_to_coeff = Audio2Coeff(sadtalker_paths, "cuda")
    animate_from_coeff = AnimateFromCoeff(sadtalker_paths, "cuda")
    print("✅ SadTalkerモデル初期化完了")
except Exception as e:
    print(f"❌ SadTalkerモデル初期化失敗: {e}")
    preprocess_model = None
    audio_to_coeff = None
    animate_from_coeff = None

app = FastAPI(title="SadTalker Standalone API", version="1.0.0")

@app.get("/")
async def root():
    return {"message": "SadTalker Standalone API", "status": "running"}

@app.get("/health")
async def health_check():
    """ヘルスチェック"""
    if all([preprocess_model, audio_to_coeff, animate_from_coeff]):
        return {"status": "healthy", "models": "loaded"}
    else:
        raise HTTPException(status_code=500, detail="Models not loaded")

@app.post("/generate_video")
async def generate_talking_video(
    image: UploadFile = File(..., description="ソース画像ファイル (JPG/PNG)"),
    audio: UploadFile = File(..., description="ソース音声ファイル (WAV/MP3)")
):
    """
    画像と音声から口パク動画を生成
    """
    if not all([preprocess_model, audio_to_coeff, animate_from_coeff]):
        raise HTTPException(status_code=500, detail="SadTalker models not loaded")
    
    # ファイル形式チェック
    if not image.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="画像ファイルを選択してください")
    
    if not audio.content_type.startswith('audio/'):
        raise HTTPException(status_code=400, detail="音声ファイルを選択してください")
    
    # 一時ディレクトリ作成
    timestamp = strftime("%Y_%m_%d_%H_%M_%S")
    save_dir = os.path.join(RESULTS_DIR, f"generation_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    
    try:
        # アップロードファイルを保存
        image_path = os.path.join(save_dir, f"input_image.{image.filename.split('.')[-1]}")
        audio_path = os.path.join(save_dir, f"input_audio.{audio.filename.split('.')[-1]}")
        
        with open(image_path, "wb") as f:
            shutil.copyfileobj(image.file, f)
        
        with open(audio_path, "wb") as f:
            shutil.copyfileobj(audio.file, f)
        
        print(f"📁 ファイル保存完了: {image_path}, {audio_path}")
        
        # SadTalker処理開始
        print("🔄 SadTalker処理開始...")
        
        # 1. 前処理 - 顔検出とクロップ
        first_frame_dir = os.path.join(save_dir, 'first_frame_dir')
        os.makedirs(first_frame_dir, exist_ok=True)
        
        first_coeff_path, crop_pic_path, crop_info = preprocess_model.generate(
            image_path, first_frame_dir, "full", source_image_flag=True
        )
        print("✅ 顔検出・前処理完了")
        
        # 2. 音声解析 - 音声から顔の動きを予測
        ref_eyeblink_coeff_path = None
        ref_pose_coeff_path = None
        batch = get_data(first_coeff_path, audio_path, "cuda", ref_eyeblink_coeff_path, still=True)
        coeff_path = audio_to_coeff.generate(batch, save_dir, 0, ref_pose_coeff_path)
        print("✅ 音声解析完了")
        
        # 3. 動画生成 - 顔アニメーション生成（バッチサイズ維持）
        facerender_batch_size = 3  # 現在の設定を維持
        data = get_facerender_data(
            coeff_path, crop_pic_path, first_coeff_path, audio_path, 
            facerender_batch_size, None, None, None,
            expression_scale=0.7, still_mode=True, preprocess="crop"  # 最適化済み設定
        )
        
        video_path = animate_from_coeff.generate_deploy(
            data, save_dir, image_path, crop_info,
            enhancer="gfpgan", background_enhancer=None, preprocess="crop"  # 高速化設定
        )
        print("✅ 動画生成完了")
        
        # 結果確認
        if os.path.exists(video_path):
            file_size = os.path.getsize(video_path)
            print(f"🎉 成功! 動画生成完了: {video_path} ({file_size} bytes)")
            
            # ファイルレスポンスで動画を返す
            return FileResponse(
                path=video_path,
                media_type='video/mp4',
                filename=f"sadtalker_result_{timestamp}.mp4"
            )
        else:
            raise HTTPException(status_code=500, detail="動画生成に失敗しました")
            
    except Exception as e:
        import traceback
        error_msg = f"処理中にエラーが発生: {str(e)}"
        traceback_msg = traceback.format_exc()
        print(f"❌ {error_msg}")
        print(f"Traceback: {traceback_msg}")
        raise HTTPException(status_code=500, detail=error_msg)

@app.post("/generate_video_simple")
async def generate_simple(
    image_file: str = "input/source_image.jpg",
    audio_file: str = "input/source_audio.mp3"
):
    """
    ローカルファイルから口パク動画を生成（テスト用）
    """
    if not all([preprocess_model, audio_to_coeff, animate_from_coeff]):
        raise HTTPException(status_code=500, detail="SadTalker models not loaded")
    
    # ファイル存在確認
    if not os.path.exists(image_file):
        raise HTTPException(status_code=404, detail=f"画像ファイルが見つかりません: {image_file}")
    
    if not os.path.exists(audio_file):
        raise HTTPException(status_code=404, detail=f"音声ファイルが見つかりません: {audio_file}")
    
    timestamp = strftime("%Y_%m_%d_%H_%M_%S")
    save_dir = os.path.join(RESULTS_DIR, f"simple_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    
    try:
        print(f"🔄 処理開始: {image_file} + {audio_file}")
        
        # 前処理
        first_frame_dir = os.path.join(save_dir, 'first_frame_dir')
        os.makedirs(first_frame_dir, exist_ok=True)
        
        first_coeff_path, crop_pic_path, crop_info = preprocess_model.generate(
            image_file, first_frame_dir, "full", source_image_flag=True
        )
        
        # 音声解析
        batch = get_data(first_coeff_path, audio_file, "cuda", None, still=True)
        coeff_path = audio_to_coeff.generate(batch, save_dir, 0, None)
        
        # 動画生成
        data = get_facerender_data(
            coeff_path, crop_pic_path, first_coeff_path, audio_file, 
            10, None, None, None, expression_scale=1, still_mode=True, preprocess="full"
        )
        
        video_path = animate_from_coeff.generate_deploy(
            data, save_dir, image_file, crop_info,
            enhancer="gfpgan", background_enhancer=None, preprocess="full"
        )
        
        if os.path.exists(video_path):
            file_size = os.path.getsize(video_path)
            return {
                "success": True,
                "video_path": video_path,
                "file_size": file_size,
                "message": f"動画生成完了: {file_size} bytes"
            }
        else:
            raise HTTPException(status_code=500, detail="動画生成に失敗")
            
    except Exception as e:
        import traceback
        error_msg = f"エラー: {str(e)}"
        traceback_msg = traceback.format_exc()
        print(f"❌ {error_msg}")
        print(f"Traceback: {traceback_msg}")
        raise HTTPException(status_code=500, detail=error_msg)

if __name__ == "__main__":
    import uvicorn
    print("🚀 SadTalker Standalone API を起動中...")
    uvicorn.run(app, host="0.0.0.0", port=10364)