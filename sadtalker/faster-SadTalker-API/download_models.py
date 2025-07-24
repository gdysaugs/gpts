#!/usr/bin/env python3
"""
SadTalker Model Download Script
Downloads all required models for SadTalker API to avoid downloading during each test
"""

import os
import requests
from pathlib import Path
from tqdm import tqdm
import hashlib

# Model URLs and their expected paths
MODELS_CONFIG = {
    # SadTalker checkpoints
    "checkpoints/auido2exp_00300-model.pth": {
        "url": "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/auido2exp_00300-model.pth",
        "sha256": None  # Add checksums if available
    },
    "checkpoints/auido2pose_00140-model.pth": {
        "url": "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/auido2pose_00140-model.pth",
        "sha256": None
    },
    "checkpoints/epoch_20.pth": {
        "url": "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/epoch_20.pth",
        "sha256": None
    },
    "checkpoints/facevid2vid_00189-model.pth.tar": {
        "url": "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/facevid2vid_00189-model.pth.tar",
        "sha256": None
    },
    "checkpoints/mapping_00109-model.pth.tar": {
        "url": "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/mapping_00109-model.pth.tar",
        "sha256": None
    },
    "checkpoints/mapping_00229-model.pth.tar": {
        "url": "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/mapping_00229-model.pth.tar",
        "sha256": None
    },
    "checkpoints/s3fd-619a316812.pth": {
        "url": "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/s3fd-619a316812.pth",
        "sha256": None
    },
    "checkpoints/shape_predictor_68_face_landmarks.dat": {
        "url": "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/shape_predictor_68_face_landmarks.dat",
        "sha256": None
    },
    "checkpoints/wav2lip.pth": {
        "url": "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/wav2lip.pth",
        "sha256": None
    },
    "checkpoints/wav2lip_gan.pth": {
        "url": "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/wav2lip_gan.pth",
        "sha256": None
    },
    
    # GFPGAN weights
    "gfpgan/weights/alignment_WFLW_4HG.pth": {
        "url": "https://github.com/xinntao/facexlib/releases/download/v0.1.0/alignment_WFLW_4HG.pth",
        "sha256": None
    },
    "gfpgan/weights/detection_Resnet50_Final.pth": {
        "url": "https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth",
        "sha256": None
    },
    "gfpgan/weights/parsing_parsenet.pth": {
        "url": "https://github.com/xinntao/facexlib/releases/download/v0.1.0/parsing_parsenet.pth",
        "sha256": None
    },
    "gfpgan/weights/GFPGANv1.4.onnx": {
        "url": "https://huggingface.co/akhaliq/GFPGAN/resolve/main/GFPGANv1.4.onnx",
        "sha256": None
    }
}

def download_file(url: str, filepath: Path, expected_sha256: str = None) -> bool:
    """Download a file with progress bar and optional checksum verification"""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filepath, 'wb') as f, tqdm(
            desc=filepath.name,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        # Verify checksum if provided
        if expected_sha256:
            actual_sha256 = calculate_sha256(filepath)
            if actual_sha256 != expected_sha256:
                print(f"❌ Checksum mismatch for {filepath.name}")
                return False
            else:
                print(f"✅ Checksum verified for {filepath.name}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to download {url}: {e}")
        return False

def calculate_sha256(filepath: Path) -> str:
    """Calculate SHA256 hash of a file"""
    hash_sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()

def main():
    """Download all required models"""
    base_dir = Path(__file__).parent
    
    print("🚀 Starting SadTalker model download...")
    print(f"📁 Base directory: {base_dir}")
    
    success_count = 0
    total_count = len(MODELS_CONFIG)
    
    for relative_path, config in MODELS_CONFIG.items():
        file_path = base_dir / relative_path
        
        # Create directory if it doesn't exist
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Skip if file already exists
        if file_path.exists():
            print(f"⏭️  {relative_path} already exists, skipping...")
            success_count += 1
            continue
        
        print(f"📥 Downloading {relative_path}...")
        
        if download_file(config["url"], file_path, config["sha256"]):
            print(f"✅ Successfully downloaded {relative_path}")
            success_count += 1
        else:
            print(f"❌ Failed to download {relative_path}")
    
    print(f"\n🎉 Download complete: {success_count}/{total_count} models downloaded")
    
    if success_count == total_count:
        print("✅ All models downloaded successfully!")
        return 0
    else:
        print(f"⚠️  {total_count - success_count} models failed to download")
        return 1

if __name__ == "__main__":
    exit(main())