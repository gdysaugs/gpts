#!/usr/bin/env python3
"""
Super Wav2Lip FastAPI Client Test
ツンデレClaude作成のテストクライアント (｡•̀ᴗ-)✧
"""

import requests
import time
import os
from pathlib import Path

# サーバーURL
BASE_URL = "http://localhost:8002"

def test_server_health():
    """サーバーヘルスチェック"""
    print("🔍 サーバーヘルスチェック...")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ サーバー正常: {data}")
            return True
        else:
            print(f"❌ サーバー異常: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ サーバー接続エラー: {e}")
        return False

def test_root_endpoint():
    """ルートエンドポイントテスト"""
    print("🔍 ルートエンドポイントテスト...")
    try:
        response = requests.get(f"{BASE_URL}/", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ ルート応答: {data}")
            return True
        else:
            print(f"❌ ルート応答異常: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ ルート接続エラー: {e}")
        return False

def test_lipsync_generation():
    """口パク生成テスト"""
    print("🎬 口パク生成テスト...")
    
    # 入力ファイル確認
    video_path = "/home/adama/gpts/super wav2lip/input/videos/source_video.mp4"
    audio_path = "/home/adama/gpts/super wav2lip/input/audio/target_audio.wav"
    
    if not os.path.exists(video_path):
        print(f"❌ 動画ファイルが見つかりません: {video_path}")
        return False
    
    if not os.path.exists(audio_path):
        print(f"❌ 音声ファイルが見つかりません: {audio_path}")
        return False
    
    try:
        # ファイルアップロード準備
        with open(video_path, 'rb') as video_file, open(audio_path, 'rb') as audio_file:
            files = {
                'video_file': ('source_video.mp4', video_file, 'video/mp4'),
                'audio_file': ('target_audio.wav', audio_file, 'audio/wav')
            }
            
            data = {
                'enhancer': 'none',  # 高速テストのため
                'hq_output': False,
                'face_mask': False,
                'pads': 4,
                'resize_factor': 1,
                'blending': 10.0
            }
            
            print("📤 リクエスト送信中...")
            start_time = time.time()
            
            response = requests.post(
                f"{BASE_URL}/generate-lipsync",
                files=files,
                data=data,
                timeout=120  # 2分タイムアウト
            )
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ 口パク生成成功！")
                print(f"⏱️  処理時間: {processing_time:.2f}秒")
                print(f"📊 結果: {result}")
                
                # ダウンロードテスト
                if 'download_url' in result:
                    download_url = f"{BASE_URL}{result['download_url']}"
                    print(f"💾 ダウンロードテスト: {download_url}")
                    
                    download_response = requests.get(download_url, timeout=30)
                    if download_response.status_code == 200:
                        print(f"✅ ダウンロード成功: {len(download_response.content)} bytes")
                        return True
                    else:
                        print(f"❌ ダウンロード失敗: {download_response.status_code}")
                        return False
                
                return True
            else:
                print(f"❌ 口パク生成失敗: {response.status_code}")
                print(f"📝 エラー詳細: {response.text}")
                return False
                
    except Exception as e:
        print(f"❌ 口パク生成エラー: {e}")
        return False

def test_model_change():
    """モデル変更テスト"""
    print("🔄 モデル変更テスト...")
    
    new_model_path = "/app/models/onnx/wav2lip_384.onnx"
    
    try:
        data = {'checkpoint_path': new_model_path}
        response = requests.post(f"{BASE_URL}/change-model", data=data, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ モデル変更成功: {result}")
            return True
        else:
            print(f"❌ モデル変更失敗: {response.status_code}")
            print(f"📝 エラー詳細: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ モデル変更エラー: {e}")
        return False

def main():
    """メインテスト実行"""
    print("🎭 Super Wav2Lip FastAPI テスト開始")
    print("=====================================")
    
    # テスト実行
    tests = [
        ("サーバーヘルスチェック", test_server_health),
        ("ルートエンドポイント", test_root_endpoint),
        ("口パク生成", test_lipsync_generation),
        ("モデル変更", test_model_change),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🧪 {test_name}テスト実行中...")
        try:
            result = test_func()
            results.append((test_name, result))
            print(f"{'✅' if result else '❌'} {test_name}: {'成功' if result else '失敗'}")
        except Exception as e:
            print(f"❌ {test_name}でエラー: {e}")
            results.append((test_name, False))
        
        time.sleep(1)  # テスト間の待機時間
    
    # 結果サマリー
    print("\n📊 テスト結果サマリー")
    print("=====================")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\n🎯 合計: {passed}/{total} テスト通過")
    
    if passed == total:
        print("🎉 全てのテストが成功しました！")
        print("ふん！当然よね。私が作ったんだから！(｡•̀ᴗ-)✧")
    else:
        print("⚠️ 一部のテストが失敗しました")
        print("まあ、完璧じゃないけど...頑張ったじゃない (￣▽￣)")

if __name__ == "__main__":
    main()