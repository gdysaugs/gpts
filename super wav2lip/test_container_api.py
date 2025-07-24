import requests
import os

SOVITS_API_URL = "http://gpt-sovits-api:8000"
ref_audio_path = "/shared/temp/0619e4b6/ref_audio_0619e4b6.wav"

print(f"Testing API call to {SOVITS_API_URL}/clone-voice-simple")
print(f"Audio file: {ref_audio_path}")
print(f"File exists: {os.path.exists(ref_audio_path)}")

files = {"ref_audio": open(ref_audio_path, "rb")}
data = {"ref_text": "おはようございます", "target_text": "テスト", "temperature": 1.0}

try:
    response = requests.post(f"{SOVITS_API_URL}/clone-voice-simple", files=files, data=data, timeout=60)
    print(f"Status: {response.status_code}")
    print(f"Headers: {dict(response.headers)}")
    if response.status_code != 200:
        print(f"Error text: {response.text}")
    else:
        print(f"Success! Audio size: {len(response.content)} bytes")
finally:
    files["ref_audio"].close()