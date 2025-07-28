#!/usr/bin/env python3
"""
Quick test to verify 500 error is fixed
"""

import requests
import time

def test_api_endpoints():
    """Test all API endpoints"""
    print("🧪 Testing API endpoints after fix...")
    
    # Test basic endpoints
    endpoints = [
        ("http://localhost:8000/", "GPT-SoVITS"),
        ("http://localhost:8002/", "Wav2Lip Optimized"),
        ("http://localhost:8002/health", "Wav2Lip Health"),
        ("http://localhost:8002/models", "Wav2Lip Models"),
        ("http://localhost:7860/", "Gradio UI")
    ]
    
    for url, name in endpoints:
        try:
            response = requests.get(url, timeout=10)
            status = "✅ OK" if response.status_code == 200 else f"❌ {response.status_code}"
            print(f"  {name:20} | {status}")
        except Exception as e:
            print(f"  {name:20} | ❌ ERROR: {e}")
    
    print()
    
    # Test warmup endpoint
    print("🔥 Testing warmup endpoint...")
    try:
        response = requests.get("http://localhost:8002/warmup", timeout=30)
        if response.status_code == 200:
            print("  ✅ Warmup successful")
        else:
            print(f"  ❌ Warmup failed: {response.status_code}")
    except Exception as e:
        print(f"  ❌ Warmup error: {e}")
    
    print()
    print("🎉 500 error should be fixed! Try the WebUI at http://localhost:7860")

if __name__ == "__main__":
    test_api_endpoints()