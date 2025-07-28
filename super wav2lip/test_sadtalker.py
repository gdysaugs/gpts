#!/usr/bin/env python3
"""
Test script to debug SadTalker API integration
"""

import requests
import os
import tempfile

def test_sadtalker_api():
    """Test SadTalker API with sample files"""
    
    # API endpoint
    sadtalker_url = "http://localhost:8001"
    
    # Test files
    image_path = "/home/adama/project/gpts/super wav2lip/input/test_image.jpg"
    audio_path = "/home/adama/project/gpts/super wav2lip/input/audio/test_audio.mp3"
    
    print(f"🎭 Testing SadTalker API...")
    print(f"📡 API URL: {sadtalker_url}")
    print(f"🖼️ Image: {image_path}")
    print(f"🎵 Audio: {audio_path}")
    
    # Check if files exist
    if not os.path.exists(image_path):
        print(f"❌ Image file not found: {image_path}")
        return
    
    if not os.path.exists(audio_path):
        print(f"❌ Audio file not found: {audio_path}")
        return
    
    print(f"✅ Files exist, starting API test...")
    
    try:
        # First test status endpoint
        print(f"📊 Testing status endpoint...")
        status_response = requests.get(f"{sadtalker_url}/status")
        print(f"Status Response: {status_response.status_code}")
        if status_response.status_code == 200:
            print(f"Status: {status_response.json()}")
        
        # Test generation endpoint
        print(f"🎬 Testing generation endpoint...")
        
        with open(image_path, 'rb') as img_file, open(audio_path, 'rb') as aud_file:
            files = {
                'image': (os.path.basename(image_path), img_file, 'image/jpeg'),
                'audio': (os.path.basename(audio_path), aud_file, 'audio/mp3')
            }
            data = {
                'quality': 'high',
                'fp16': 'true'
            }
            
            print(f"📡 Sending POST request to {sadtalker_url}/generate")
            response = requests.post(
                f"{sadtalker_url}/generate", 
                files=files, 
                data=data, 
                timeout=180
            )
        
        print(f"🔍 Response Status: {response.status_code}")
        print(f"🔍 Response Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Success: {result}")
            
            if result.get('success'):
                download_url = result.get('download_url')
                if download_url:
                    print(f"📥 Download URL: {download_url}")
                    # Try to download the video
                    video_response = requests.get(f"{sadtalker_url}{download_url}")
                    if video_response.status_code == 200:
                        print(f"✅ Video download successful! Size: {len(video_response.content)} bytes")
                    else:
                        print(f"❌ Video download failed: {video_response.status_code}")
            else:
                print(f"❌ API returned success=false: {result}")
        else:
            print(f"❌ Request failed with status {response.status_code}")
            try:
                error_data = response.json()
                print(f"Error details: {error_data}")
            except:
                print(f"Error response text: {response.text}")
                
    except Exception as e:
        print(f"❌ Exception occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_sadtalker_api()