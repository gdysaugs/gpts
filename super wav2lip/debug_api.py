#!/usr/bin/env python3
"""
Debug script to reproduce the exact API call that the integrated frontend makes
"""

import requests
import os

# Test with the same parameters as the frontend
SOVITS_API_URL = "http://localhost:8000"
ref_audio_path = "/home/adama/gpts/super wav2lip/input/audio/test_audio.mp3"
ref_text = "おはようございます"
target_text = "テスト"
temperature = 1.0

print(f"🔍 Testing API call to {SOVITS_API_URL}/clone-voice-simple")
print(f"📁 Reference audio: {ref_audio_path}")
print(f"📝 Reference text: {ref_text}")
print(f"🎯 Target text: {target_text}")
print(f"🌡️ Temperature: {temperature}")

# Check if file exists
if not os.path.exists(ref_audio_path):
    print(f"❌ Audio file not found: {ref_audio_path}")
    exit(1)

print(f"✅ Audio file exists ({os.path.getsize(ref_audio_path)} bytes)")

# Prepare the exact same request as the frontend
files = {
    "ref_audio": open(ref_audio_path, "rb")
}

data = {
    "ref_text": ref_text,
    "target_text": target_text,
    "temperature": temperature
}

try:
    print("\n🚀 Making POST request...")
    response = requests.post(
        f"{SOVITS_API_URL}/clone-voice-simple",
        files=files,
        data=data,
        timeout=60
    )
    
    print(f"📊 Response status: {response.status_code}")
    print(f"📋 Response headers: {dict(response.headers)}")
    
    if response.status_code == 200:
        print(f"✅ Success! Response size: {len(response.content)} bytes")
        # Save the response for testing
        with open("/tmp/debug_response.wav", "wb") as f:
            f.write(response.content)
        print("💾 Audio saved to /tmp/debug_response.wav")
    else:
        print(f"❌ Error! Status: {response.status_code}")
        print(f"📄 Response text: {response.text}")
        
finally:
    # Close the file handle
    files["ref_audio"].close()