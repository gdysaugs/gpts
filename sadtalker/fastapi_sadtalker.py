#!/usr/bin/env python3
"""
SadTalker FastAPI サーバー
最適化された設定で高速な動画生成を提供
"""

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
import os
import sys
import shutil
import tempfile
import time
from pathlib import Path
import torch
import subprocess

# SadTalkerモジュールパス追加
sys.path.insert(0, './src')

# インポート
from src.utils.preprocess import CropAndExtract
from src.test_audio2coeff import Audio2Coeff
from src.facerender.animate import AnimateFromCoeff
from src.generate_batch import get_data
from src.generate_facerender_batch import get_facerender_data
from src.utils.init_path import init_path

# CUDA最適化設定
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

# FastAPIアプリ初期化
app = FastAPI(title="SadTalker API", version="1.0.0")

# グローバル変数
device = "cuda"
checkpoint_dir = "./checkpoints"
config_dir = "./src/config"
size = 256
old_version = False

# モデル初期化（起動時に一度だけ実行）
print("🔄 SadTalkerモデル初期化中...")
try:
    sadtalker_paths = init_path(checkpoint_dir, config_dir, size, old_version, "crop")
    preprocess_model = CropAndExtract(sadtalker_paths, device)
    audio_to_coeff = Audio2Coeff(sadtalker_paths, device)
    animate_from_coeff = AnimateFromCoeff(sadtalker_paths, device)
    print("✅ SadTalkerモデル初期化完了")
except Exception as e:
    print(f"❌ モデル初期化失敗: {e}")
    preprocess_model = None
    audio_to_coeff = None
    animate_from_coeff = None

@app.get("/")
async def root():
    """ルートエンドポイント"""
    return {"message": "SadTalker API", "status": "running"}

@app.get("/health")
async def health_check():
    """ヘルスチェック"""
    if all([preprocess_model, audio_to_coeff, animate_from_coeff]):
        return {"status": "healthy", "models": "loaded"}
    else:
        raise HTTPException(status_code=500, detail="Models not loaded")

@app.post("/generate")
async def generate_video(
    image: UploadFile = File(..., description="ソース画像 (JPG/PNG)"),
    audio: UploadFile = File(..., description="ソース音声 (WAV/MP3)"),
    expression_scale: float = Form(0.7, description="表情スケール (0.0-2.0)"),
    still_mode: bool = Form(True, description="静止モード"),
    use_gfpgan: bool = Form(True, description="GFPGAN品質向上"),
    preprocess: str = Form("crop", description="前処理モード (crop/full)")
):
    """
    画像と音声から口パク動画を生成
    
    最適化済みパラメータ:
    - expression_scale: 0.7 (自然な表情)
    - still_mode: True (控えめな動き)
    - preprocess: crop (高速処理)
    - batch_size: 3 (RTX 3050用)
    """
    if not all([preprocess_model, audio_to_coeff, animate_from_coeff]):
        raise HTTPException(status_code=500, detail="SadTalker models not loaded")
    
    # 処理開始時刻
    start_time = time.time()
    
    # 一時ディレクトリ作成
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        try:
            # ファイル保存
            image_path = temp_path / f"input_{int(time.time())}.jpg"
            audio_path = temp_path / f"audio_{int(time.time())}.wav"
            
            with open(image_path, "wb") as f:
                shutil.copyfileobj(image.file, f)
            
            with open(audio_path, "wb") as f:
                shutil.copyfileobj(audio.file, f)
            
            print(f"📁 入力ファイル保存完了")
            
            # 結果ディレクトリ
            save_dir = temp_path / "results"
            save_dir.mkdir(exist_ok=True)
            
            # 1. 前処理
            first_frame_dir = save_dir / 'first_frame_dir'
            first_frame_dir.mkdir(exist_ok=True)
            
            print("🔄 前処理開始...")
            preprocess_start = time.time()
            
            first_coeff_path, crop_pic_path, crop_info = preprocess_model.generate(
                str(image_path), str(first_frame_dir), preprocess, 
                source_image_flag=True, pic_size=size
            )
            
            preprocess_time = time.time() - preprocess_start
            print(f"✅ 前処理完了: {preprocess_time:.1f}秒")
            
            if first_coeff_path is None:
                raise HTTPException(status_code=400, detail="顔が検出されませんでした")
            
            # 2. 音声解析
            print("🔄 音声解析開始...")
            audio_start = time.time()
            
            batch = get_data(first_coeff_path, str(audio_path), device, 
                           ref_eyeblink_coeff_path=None, still=still_mode)
            
            coeff_path = audio_to_coeff.generate(batch, str(save_dir), 0, None)
            
            audio_time = time.time() - audio_start
            print(f"✅ 音声解析完了: {audio_time:.1f}秒")
            
            # 3. 動画生成
            print("🔄 動画生成開始...")
            render_start = time.time()
            
            # バッチサイズは3を維持（RTX 3050用）
            batch_size = 3
            
            # 最適化済みパラメータでデータ準備
            data = get_facerender_data(
                coeff_path, crop_pic_path, first_coeff_path, str(audio_path),
                batch_size, None, None, None,
                expression_scale=expression_scale,
                still_mode=still_mode,
                preprocess=preprocess,
                img_size=size
            )
            
            # GFPGAN設定
            enhancer = "gfpgan" if use_gfpgan else None
            
            # 動画生成
            result_path = animate_from_coeff.generate(
                data, str(save_dir), str(image_path), crop_info,
                enhancer=enhancer, background_enhancer=None,
                preprocess=preprocess, img_size=size
            )
            
            render_time = time.time() - render_start
            print(f"✅ 動画生成完了: {render_time:.1f}秒")
            
            # 合計処理時間
            total_time = time.time() - start_time
            
            # 結果確認
            if result_path and os.path.exists(result_path):
                file_size = os.path.getsize(result_path)
                print(f"🎉 成功! 処理時間: {total_time:.1f}秒, サイズ: {file_size} bytes")
                
                # 出力ディレクトリに移動
                output_dir = Path("output")
                output_dir.mkdir(exist_ok=True)
                
                # 権限修正（Docker用）
                subprocess.run([
                    "docker", "run", "--rm", "-v", f"{os.getcwd()}:/work",
                    "busybox", "chown", "-R", f"{os.getuid()}:{os.getgid()}", "/work/output"
                ], capture_output=True)
                
                timestamp = int(time.time())
                final_path = output_dir / f"sadtalker_api_{timestamp}.mp4"
                shutil.copy2(result_path, final_path)
                
                # ファイルレスポンス
                return FileResponse(
                    path=str(final_path),
                    media_type='video/mp4',
                    filename=f"sadtalker_result_{timestamp}.mp4",
                    headers={
                        "X-Processing-Time": f"{total_time:.1f}",
                        "X-File-Size": str(file_size)
                    }
                )
            else:
                raise HTTPException(status_code=500, detail="動画生成に失敗しました")
                
        except HTTPException:
            raise
        except Exception as e:
            print(f"❌ エラー: {str(e)}")
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"処理中にエラーが発生: {str(e)}")

@app.get("/info")
async def get_info():
    """API情報取得"""
    return {
        "version": "1.0.0",
        "models_loaded": all([preprocess_model, audio_to_coeff, animate_from_coeff]),
        "device": device,
        "optimizations": {
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "tf32_enabled": torch.backends.cuda.matmul.allow_tf32,
            "default_expression_scale": 0.7,
            "default_still_mode": True,
            "default_preprocess": "crop",
            "batch_size": 3
        }
    }

if __name__ == "__main__":
    import uvicorn
    print("🚀 SadTalker FastAPI サーバー起動中...")
    uvicorn.run(app, host="0.0.0.0", port=8000)