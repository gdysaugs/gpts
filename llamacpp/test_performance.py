#!/usr/bin/env python3
"""Performance test script for the optimized llama-cpp API"""

import requests
import time
import json
import statistics

def test_chat_api(message, endpoint="http://localhost:8001/chat"):
    """Test a single chat request and return timing"""
    start_time = time.time()
    
    response = requests.post(endpoint, json={
        "message": message,
        "use_history": False,
        "stream": False
    })
    
    total_time = time.time() - start_time
    
    if response.status_code == 200:
        data = response.json()
        return {
            "success": True,
            "total_time": total_time,
            "inference_time": data.get("inference_time", 0),
            "response": data.get("response", ""),
            "message": message
        }
    else:
        return {
            "success": False,
            "total_time": total_time,
            "error": response.text,
            "message": message
        }

def run_performance_tests():
    """Run multiple tests and display statistics"""
    test_messages = [
        "こんにちは！",
        "今日の天気はどう？",
        "プログラミングについて教えて",
        "ツンデレってなに？",
        "好きな食べ物は？",
        "Python vs JavaScript どっちがいい？",
        "機械学習について説明して",
        "おやすみなさい"
    ]
    
    print("🚀 Performance Test for Optimized llama-cpp API")
    print("=" * 60)
    
    results = []
    
    for i, message in enumerate(test_messages, 1):
        print(f"\n[Test {i}/{len(test_messages)}] '{message}'")
        result = test_chat_api(message)
        
        if result["success"]:
            print(f"✅ Success in {result['total_time']:.2f}s (inference: {result['inference_time']:.2f}s)")
            print(f"   Response: {result['response'][:100]}...")
            results.append(result)
        else:
            print(f"❌ Failed: {result['error']}")
    
    # Calculate statistics
    if results:
        print("\n" + "=" * 60)
        print("📊 Performance Statistics:")
        
        total_times = [r["total_time"] for r in results]
        inference_times = [r["inference_time"] for r in results]
        
        print(f"\nTotal Request Time:")
        print(f"  • Average: {statistics.mean(total_times):.2f}s")
        print(f"  • Min: {min(total_times):.2f}s")
        print(f"  • Max: {max(total_times):.2f}s")
        print(f"  • Std Dev: {statistics.stdev(total_times):.2f}s" if len(total_times) > 1 else "")
        
        print(f"\nInference Time:")
        print(f"  • Average: {statistics.mean(inference_times):.2f}s")
        print(f"  • Min: {min(inference_times):.2f}s")
        print(f"  • Max: {max(inference_times):.2f}s")
        print(f"  • Std Dev: {statistics.stdev(inference_times):.2f}s" if len(inference_times) > 1 else "")
        
        print(f"\nThroughput: {len(results) / sum(total_times):.2f} requests/second")

if __name__ == "__main__":
    run_performance_tests()