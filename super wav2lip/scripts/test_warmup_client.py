#!/usr/bin/env python3
"""
Super Wav2Lip ウォームアップ機能テストクライアント
ツンデレClaude作成のウォームアップテスト専用 (｡•̀ᴗ-)✧
"""

import requests
import time
import json

# サーバーURL
BASE_URL = "http://localhost:8002"

def test_warmup_status():
    """ウォームアップ状況確認"""
    print("🔍 ウォームアップ状況確認...")
    try:
        response = requests.get(f"{BASE_URL}/warmup/status", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ ウォームアップ状況:")
            print(f"   完了: {data['warmup_completed']}")
            print(f"   統計: {json.dumps(data['warmup_stats'], indent=2, ensure_ascii=False)}")
            return data['warmup_completed']
        else:
            print(f"❌ ステータス取得失敗: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ ステータス取得エラー: {e}")
        return False

def test_manual_warmup():
    """手動ウォームアップテスト"""
    print("🚀 手動ウォームアップ開始...")
    try:
        response = requests.post(f"{BASE_URL}/warmup", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ ウォームアップ開始: {data['message']}")
            
            # 完了まで待機
            print("⏳ ウォームアップ完了待機中...")
            start_time = time.time()
            
            while True:
                status = test_warmup_status()
                if status:
                    end_time = time.time()
                    print(f"🎉 ウォームアップ完了！総時間: {end_time - start_time:.2f}秒")
                    break
                
                print("   ウォームアップ実行中...")
                time.sleep(5)
                
                # 5分でタイムアウト
                if time.time() - start_time > 300:
                    print("❌ ウォームアップタイムアウト")
                    break
            
            return True
        else:
            print(f"❌ ウォームアップ開始失敗: {response.status_code}")
            print(f"   詳細: {response.text}")
            return False
    except Exception as e:
        print(f"❌ ウォームアップエラー: {e}")
        return False

def test_performance_comparison():
    """ウォームアップ前後のパフォーマンス比較"""
    print("🏁 パフォーマンス比較テスト...")
    
    # 入力ファイル確認
    video_path = "/home/adama/gpts/super wav2lip/input/videos/source_video.mp4"
    audio_path = "/home/adama/gpts/super wav2lip/input/audio/target_audio.wav"
    
    import os
    if not os.path.exists(video_path) or not os.path.exists(audio_path):
        print(f"❌ 入力ファイルが見つかりません")
        return False
    
    def run_generation_test(test_name):
        """生成テスト実行"""
        print(f"🧪 {test_name}実行中...")
        start_time = time.time()
        
        try:
            with open(video_path, 'rb') as video_file, open(audio_path, 'rb') as audio_file:
                files = {
                    'video_file': ('source_video.mp4', video_file, 'video/mp4'),
                    'audio_file': ('target_audio.wav', audio_file, 'audio/wav')
                }
                
                data = {
                    'enhancer': 'none',
                    'hq_output': False
                }
                
                response = requests.post(
                    f"{BASE_URL}/generate-lipsync",
                    files=files,
                    data=data,
                    timeout=180
                )
                
                end_time = time.time()
                total_time = end_time - start_time
                
                if response.status_code == 200:
                    result = response.json()
                    processing_time = result.get('processing_time_seconds', 0)
                    print(f"✅ {test_name}成功！")
                    print(f"   総時間: {total_time:.2f}秒")
                    print(f"   処理時間: {processing_time:.2f}秒")
                    print(f"   オーバーヘッド: {(total_time - processing_time):.2f}秒")
                    return total_time, processing_time
                else:
                    print(f"❌ {test_name}失敗: {response.status_code}")
                    return None, None
                    
        except Exception as e:
            end_time = time.time()
            print(f"❌ {test_name}エラー: {e}")
            return None, None
    
    # ウォームアップ前テスト
    print("\n📊 === ウォームアップ前テスト ===")
    before_total, before_processing = run_generation_test("ウォームアップ前テスト")
    
    # ウォームアップ実行
    print("\n🔥 === ウォームアップ実行 ===")
    if not test_manual_warmup():
        return False
    
    # ウォームアップ後テスト
    print("\n📊 === ウォームアップ後テスト ===")
    after_total, after_processing = run_generation_test("ウォームアップ後テスト")
    
    # 結果比較
    if before_total and after_total:
        print("\n🎯 === パフォーマンス比較結果 ===")
        print(f"ウォームアップ前: {before_total:.2f}秒")
        print(f"ウォームアップ後: {after_total:.2f}秒")
        
        improvement = before_total - after_total
        improvement_percent = (improvement / before_total) * 100
        
        print(f"改善時間: {improvement:.2f}秒")
        print(f"改善率: {improvement_percent:.1f}%")
        
        if improvement > 0:
            print("🎉 ウォームアップ効果あり！")
        else:
            print("🤔 ウォームアップ効果微妙...")
        
        return True
    else:
        print("❌ 比較テスト失敗")
        return False

def main():
    """メインテスト実行"""
    print("🎭 Super Wav2Lip ウォームアップテスト開始")
    print("==========================================")
    
    # 基本テスト
    tests = [
        ("ウォームアップ状況確認", test_warmup_status),
        ("パフォーマンス比較", test_performance_comparison),
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
        
        time.sleep(1)
    
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
        print("ふん！私のウォームアップ機能は完璧よ！(｡•̀ᴗ-)✧")
    else:
        print("⚠️ 一部のテストが失敗しました")
        print("まあ、完璧じゃないけど...改善の余地があるわね (￣▽￣)")

if __name__ == "__main__":
    main()