#!/usr/bin/env python3
"""
FaceFusion 3.3.0 Test Script
Tests face swapping functionality with skip download option
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def test_face_swap():
    """Test face swapping with pre-downloaded models"""
    
    print("FaceFusion 3.3.0 Face Swap Test")
    print("=" * 50)
    
    # Set environment variable to skip download
    os.environ['FACEFUSION_SKIP_DOWNLOAD'] = '1'
    
    # Check if source and target images exist
    source_image = "/app/input/source.jpg"
    target_image = "/app/input/target.jpg"
    output_path = "/app/output/swapped_result.jpg"
    
    # Basic face swap command
    cmd = [
        "python", "facefusion.py",
        "headless-run",
        # Input/Output
        "-s", source_image,
        "-t", target_image,
        "-o", output_path,
        # Processing options
        "--processors", "face_swapper",
        "--face-detector-model", "yolo_face",
        "--face-recognizer-model", "arcface_simswap",
        "--face-swapper-model", "inswapper_128",
        # Performance options
        "--execution-providers", "cuda",
        "--execution-thread-count", "4",
        "--execution-queue-count", "1",
        # Skip options
        "--skip-download",
        "--skip-audio"
    ]
    
    print(f"Command: {' '.join(cmd)}")
    print("=" * 50)
    
    # Execute face swap
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"Processing completed in {processing_time:.2f} seconds")
        print("=" * 50)
        
        if result.returncode == 0:
            print("✓ Face swap successful!")
            print(f"Output saved to: {output_path}")
            
            # Check if output file was created
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path) / 1024  # KB
                print(f"Output file size: {file_size:.2f} KB")
            else:
                print("⚠ Warning: Output file not found")
        else:
            print("✗ Face swap failed!")
            print(f"Return code: {result.returncode}")
            print(f"Error output:\n{result.stderr}")
            
        # Print stdout for debugging
        if result.stdout:
            print("\nStdout:")
            print(result.stdout)
            
    except Exception as e:
        print(f"✗ Error running face swap: {str(e)}")
        return 1
    
    return 0

def test_video_face_swap():
    """Test video face swapping"""
    
    print("\nFaceFusion 3.3.0 Video Face Swap Test")
    print("=" * 50)
    
    source_image = "/app/input/source.jpg"
    target_video = "/app/input/target.mp4"
    output_path = "/app/output/swapped_video.mp4"
    
    cmd = [
        "python", "facefusion.py",
        "headless-run",
        # Input/Output
        "-s", source_image,
        "-t", target_video,
        "-o", output_path,
        # Processing options
        "--processors", "face_swapper",
        "--face-detector-model", "yolo_face",
        "--face-recognizer-model", "arcface_simswap", 
        "--face-swapper-model", "inswapper_128",
        # Video options
        "--output-video-encoder", "libx264",
        "--output-video-quality", "80",
        "--output-video-fps", "25",
        # Performance options
        "--execution-providers", "cuda",
        "--execution-thread-count", "4",
        "--execution-queue-count", "1",
        # Memory optimization
        "--video-memory-strategy", "moderate",
        # Skip options
        "--skip-download",
        "--skip-audio"
    ]
    
    print(f"Command: {' '.join(cmd)}")
    print("=" * 50)
    
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"Processing completed in {processing_time:.2f} seconds")
        print("=" * 50)
        
        if result.returncode == 0:
            print("✓ Video face swap successful!")
            print(f"Output saved to: {output_path}")
            
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
                print(f"Output file size: {file_size:.2f} MB")
        else:
            print("✗ Video face swap failed!")
            print(f"Return code: {result.returncode}")
            print(f"Error output:\n{result.stderr}")
            
        if result.stdout:
            print("\nStdout:")
            print(result.stdout)
            
    except Exception as e:
        print(f"✗ Error running video face swap: {str(e)}")
        return 1
    
    return 0

def main():
    """Main test function"""
    
    # Check if running in container
    if not os.path.exists("/app/facefusion.py"):
        print("Error: This script must be run inside the FaceFusion container")
        return 1
    
    # Check for test files
    required_files = [
        "/app/input/source.jpg",
        "/app/input/target.jpg"
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print("Warning: Missing test files:")
        for f in missing_files:
            print(f"  - {f}")
        print("\nPlease add test images to the input directory")
        print("  - source.jpg: Face to swap from")
        print("  - target.jpg: Image to swap face into")
        print("  - target.mp4: (Optional) Video to swap face into")
        return 1
    
    # Run image face swap test
    result = test_face_swap()
    
    # Run video test if video exists
    if os.path.exists("/app/input/target.mp4"):
        test_video_face_swap()
    
    return result

if __name__ == "__main__":
    sys.exit(main())