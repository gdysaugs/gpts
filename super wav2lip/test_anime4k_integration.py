#!/usr/bin/env python3
"""
Anime4K Integration Test
アニメ画像でのAnime4K vs GFPGAN比較テスト
"""

import cv2
import numpy as np
import requests
import tempfile
import os
import time
import logging
from PIL import Image

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_test_anime_face():
    """テスト用のアニメ風顔画像を生成"""
    # 512x512のアニメ風テスト画像を作成
    img = np.zeros((512, 512, 3), dtype=np.uint8)
    
    # 背景（薄いピンク）
    img[:] = (240, 220, 255)
    
    # 顔の輪郭（円形）
    cv2.circle(img, (256, 256), 200, (255, 240, 220), -1)
    
    # 目（大きなアニメ風の目）
    # 左目
    cv2.circle(img, (200, 220), 40, (0, 0, 0), -1)
    cv2.circle(img, (195, 215), 15, (255, 255, 255), -1)
    cv2.circle(img, (190, 210), 8, (100, 150, 255), -1)
    
    # 右目
    cv2.circle(img, (312, 220), 40, (0, 0, 0), -1)
    cv2.circle(img, (317, 215), 15, (255, 255, 255), -1)
    cv2.circle(img, (322, 210), 8, (100, 150, 255), -1)
    
    # 鼻（小さな点）
    cv2.circle(img, (256, 260), 3, (200, 180, 180), -1)
    
    # 口（小さな弧）
    cv2.ellipse(img, (256, 300), (20, 10), 0, 0, 180, (255, 100, 120), 3)
    
    # 髪（上部に簡単な線）
    cv2.rectangle(img, (150, 80), (362, 180), (100, 50, 20), -1)
    
    return img

def create_test_video_from_image(image_path, duration=3.0, fps=30):
    """画像から短いテスト動画を作成"""
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    height, width, channels = img.shape
    
    # 一時動画ファイルを作成
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_video:
        temp_video_path = temp_video.name
    
    # VideoWriterを設定
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))
    
    # 指定された秒数分のフレームを書き込み
    total_frames = int(duration * fps)
    for i in range(total_frames):
        out.write(img)
    
    out.release()
    return temp_video_path

def test_enhancer_comparison():
    """Anime4K vs GFPGAN の比較テスト"""
    try:
        logger.info("🎨 Starting Anime4K vs GFPGAN comparison test...")
        
        # 1. テスト用アニメ画像を作成
        anime_face = create_test_anime_face()
        
        # 一時画像ファイルに保存
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_img:
            cv2.imwrite(temp_img.name, anime_face)
            test_image_path = temp_img.name
        
        # 2. 画像から動画を作成
        test_video_path = create_test_video_from_image(test_image_path, duration=2.0)
        if not test_video_path:
            logger.error("Failed to create test video")
            return False
        
        # 3. テスト用音声（サイレント音声）を作成
        import wave
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
            # 2秒のサイレント音声を作成
            sample_rate = 16000
            duration = 2.0
            frames = int(sample_rate * duration)
            
            with wave.open(temp_audio.name, 'wb') as wav_file:
                wav_file.setnchannels(1)  # モノラル
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(sample_rate)
                
                # サイレント音声データ
                silent_data = b'\x00\x00' * frames
                wav_file.writeframes(silent_data)
            
            test_audio_path = temp_audio.name
        
        # 4. FastAPI サーバーが起動しているかチェック
        try:
            response = requests.get("http://localhost:8002/", timeout=5)
            logger.info("✅ Wav2Lip FastAPI server is running")
        except:
            logger.error("❌ Wav2Lip FastAPI server is not running")
            logger.info("Please start the server with: docker-compose -f docker-compose-fixed.yml up -d")
            return False
        
        # 5. Anime4Kで処理
        logger.info("🎨 Testing with Anime4K enhancer...")
        start_time = time.time()
        
        with open(test_video_path, 'rb') as vf, open(test_audio_path, 'rb') as af:
            files = {
                "video_file": vf,
                "audio_file": af
            }
            data = {
                "enhancer": "anime4k",
                "batch_size": 8
            }
            
            anime4k_response = requests.post(
                "http://localhost:8002/generate-lipsync",
                files=files, data=data, timeout=60
            )
        
        anime4k_time = time.time() - start_time
        
        if anime4k_response.status_code == 200:
            # 結果を保存
            anime4k_output = "/tmp/test_anime4k_result.mp4"
            with open(anime4k_output, 'wb') as f:
                f.write(anime4k_response.content)
            logger.info(f"✅ Anime4K test completed in {anime4k_time:.2f}s")
            logger.info(f"   Output saved: {anime4k_output}")
        else:
            logger.error(f"❌ Anime4K test failed: {anime4k_response.status_code}")
        
        # 6. GFPGANで処理
        logger.info("🎭 Testing with GFPGAN enhancer...")
        start_time = time.time()
        
        with open(test_video_path, 'rb') as vf, open(test_audio_path, 'rb') as af:
            files = {
                "video_file": vf,
                "audio_file": af
            }
            data = {
                "enhancer": "gfpgan",
                "batch_size": 8
            }
            
            gfpgan_response = requests.post(
                "http://localhost:8002/generate-lipsync",
                files=files, data=data, timeout=60
            )
        
        gfpgan_time = time.time() - start_time
        
        if gfpgan_response.status_code == 200:
            # 結果を保存
            gfpgan_output = "/tmp/test_gfpgan_result.mp4"
            with open(gfpgan_output, 'wb') as f:
                f.write(gfpgan_response.content)
            logger.info(f"✅ GFPGAN test completed in {gfpgan_time:.2f}s")
            logger.info(f"   Output saved: {gfpgan_output}")
        else:
            logger.error(f"❌ GFPGAN test failed: {gfpgan_response.status_code}")
        
        # 7. 結果比較
        logger.info("📊 Test Results Summary:")
        logger.info("=" * 50)
        logger.info(f"Anime4K Processing Time: {anime4k_time:.2f}s")
        logger.info(f"GFPGAN Processing Time:  {gfpgan_time:.2f}s")
        
        if anime4k_response.status_code == 200 and gfpgan_response.status_code == 200:
            anime4k_size = len(anime4k_response.content)
            gfpgan_size = len(gfpgan_response.content)
            logger.info(f"Anime4K Output Size:     {anime4k_size:,} bytes")
            logger.info(f"GFPGAN Output Size:      {gfpgan_size:,} bytes")
            
            if anime4k_time < gfpgan_time:
                logger.info("🚀 Anime4K is faster!")
            else:
                logger.info("🐌 GFPGAN is faster")
        
        # クリーンアップ
        for path in [test_image_path, test_video_path, test_audio_path]:
            try:
                os.unlink(path)
            except:
                pass
        
        logger.info("✅ Comparison test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False

def test_anime4k_standalone():
    """Anime4K エンハンサー単体テスト"""
    try:
        logger.info("🎨 Testing Anime4K enhancer standalone...")
        
        # sys.pathにAnime4Kパスを追加
        import sys
        sys.path.append('/home/adama/project/gpts/super wav2lip/models/enhancers/Anime4K')
        
        from anime4k_enhancer import Anime4KEnhancer
        
        # エンハンサー初期化
        enhancer = Anime4KEnhancer(device="cpu")  # CPUでテスト
        
        # テスト画像作成
        test_face = create_test_anime_face()
        
        # 強化実行
        enhanced = enhancer.enhance_face(test_face)
        
        # 結果確認
        if enhanced is not None and enhanced.shape == test_face.shape:
            logger.info("✅ Anime4K standalone test successful")
            logger.info(f"   Input shape: {test_face.shape}")
            logger.info(f"   Output shape: {enhanced.shape}")
            
            # 結果を保存
            cv2.imwrite("/tmp/anime4k_test_input.png", test_face)
            cv2.imwrite("/tmp/anime4k_test_output.png", enhanced)
            logger.info("   Test images saved to /tmp/anime4k_test_*.png")
            
            return True
        else:
            logger.error("❌ Anime4K enhancement failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Standalone test failed: {e}")
        return False

if __name__ == "__main__":
    logger.info("🧪 Starting Anime4K Integration Tests...")
    
    # 1. スタンドアロンテスト
    success1 = test_anime4k_standalone()
    
    # 2. 比較テスト
    success2 = test_enhancer_comparison()
    
    if success1 and success2:
        logger.info("🎉 All tests passed!")
    else:
        logger.error("❌ Some tests failed")
    
    logger.info("🧪 Test completed.")