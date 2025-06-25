#!/usr/bin/env python3
"""
Llama.cpp Model Test Script
モデルの基本動作とGPU認識をテストする
"""

import os
import sys
import time
from pathlib import Path

try:
    from llama_cpp import Llama
except ImportError:
    print("❌ llama-cpp-python not installed!")
    sys.exit(1)

def test_gpu_availability():
    """GPU使用可能性をテスト"""
    print("🔍 Checking GPU availability...")
    
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi'], capture_output=True)
        if result.returncode == 0:
            print("✅ NVIDIA GPU detected")
            
            # GPU詳細情報
            gpu_info = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'], 
                                    capture_output=True, text=True)
            if gpu_info.returncode == 0:
                print(f"   GPU Info: {gpu_info.stdout.strip()}")
            return True
        else:
            print("❌ nvidia-smi failed")
            return False
    except FileNotFoundError:
        print("❌ nvidia-smi not found")
        return False
    except Exception as e:
        print(f"❌ GPU check error: {e}")
        return False

def test_model_loading():
    """モデルのロードテスト"""
    print("\n🦙 Testing model loading...")
    
    model_path = "/app/models/Berghof-NSFW-7B.i1-Q4_K_S.gguf"
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return False
    
    print(f"📁 Model file found: {Path(model_path).name}")
    print(f"   Size: {os.path.getsize(model_path) / (1024**3):.2f} GB")
    
    try:
        print("⏳ Loading model...")
        start_time = time.time()
        
        llm = Llama(
            model_path=model_path,
            n_gpu_layers=-1,  # 全レイヤーをGPUに
            n_ctx=1024,       # テスト用に小さめ
            n_batch=256,      # 低VRAM最適化
            verbose=False,
            # FP16 + 低VRAM最適化設定
            f16_kv=True,      # FP16キーバリューキャッシュ
            use_mmap=True,    # メモリマッピング有効
            use_mlock=False,  # 低VRAM時は無効
            low_vram=True,    # 低VRAM最適化
            n_threads=8,      # CPU補助スレッド
        )
        
        load_time = time.time() - start_time
        print(f"✅ Model loaded successfully in {load_time:.2f} seconds")
        
        return llm
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return None

def test_inference(llm):
    """推論テスト"""
    print("\n🧠 Testing inference...")
    
    test_prompt = "Hello! How are you today?"
    print(f"📝 Prompt: {test_prompt}")
    
    try:
        start_time = time.time()
        
        response = llm(
            test_prompt,
            max_tokens=50,
            temperature=0.7,
            stop=["\n"]
        )
        
        inference_time = time.time() - start_time
        response_text = response["choices"][0]["text"].strip()
        
        print(f"🤖 Response: {response_text}")
        print(f"⏱️ Inference time: {inference_time:.2f} seconds")
        print(f"📊 Tokens: {len(response_text.split())} words")
        
        return True
        
    except Exception as e:
        print(f"❌ Inference failed: {e}")
        return False

def test_gpu_memory():
    """GPU メモリ使用量をチェック"""
    print("\n💾 Checking GPU memory usage...")
    
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            memory_info = result.stdout.strip().split(', ')
            used_memory = int(memory_info[0])
            total_memory = int(memory_info[1])
            usage_percent = (used_memory / total_memory) * 100
            
            print(f"   Used: {used_memory} MB / {total_memory} MB ({usage_percent:.1f}%)")
            
            if usage_percent > 90:
                print("⚠️ High GPU memory usage!")
            elif usage_percent > 50:
                print("✅ Moderate GPU memory usage")
            else:
                print("✅ Low GPU memory usage")
                
        else:
            print("❌ Could not get GPU memory info")
            
    except Exception as e:
        print(f"❌ GPU memory check failed: {e}")

def main():
    """メインテスト関数"""
    print("🚀 Llama.cpp Python Model Test")
    print("=" * 50)
    
    # テスト実行
    tests_passed = 0
    total_tests = 3
    
    # 1. GPU テスト
    if test_gpu_availability():
        tests_passed += 1
    
    # 2. モデルロードテスト
    llm = test_model_loading()
    if llm:
        tests_passed += 1
        
        # 3. 推論テスト
        if test_inference(llm):
            tests_passed += 1
    
    # GPU メモリチェック
    test_gpu_memory()
    
    # 結果表示
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! System is ready for chat.")
        return True
    else:
        print("❌ Some tests failed. Please check the configuration.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)