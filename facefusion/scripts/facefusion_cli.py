#!/usr/bin/env python3
"""
FaceFusion CLI Script for face swapping
Optimized for Docker WSL2 + RTX GPU environment

Usage examples:
    python facefusion_cli.py --source source_face.jpg --target target_video.mp4
    python facefusion_cli.py --source source_face.jpg --target target_video.mp4 --output swapped_result.mp4
"""

import os
import sys
import argparse
import time
import logging
import subprocess
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/app/logs/facefusion_cli.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def check_gpu():
    """Check GPU availability and CUDA setup"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"✅ GPU detected: {gpu_name} ({gpu_memory:.1f}GB)")
            return True
        else:
            logger.error("❌ CUDA not available")
            return False
    except ImportError:
        logger.error("❌ PyTorch not available")
        return False
    except Exception as e:
        logger.error(f"❌ GPU check failed: {e}")
        return False

def validate_inputs(source_path, target_path):
    """Validate input files exist and are accessible"""
    source_file = Path(source_path)
    target_file = Path(target_path)
    
    if not source_file.exists():
        raise FileNotFoundError(f"Source image not found: {source_path}")
    
    if not target_file.exists():
        raise FileNotFoundError(f"Target video not found: {target_path}")
    
    # Check file extensions
    valid_image_ext = {'.jpg', '.jpeg', '.png', '.bmp'}
    valid_video_ext = {'.mp4', '.avi', '.mov', '.mkv'}
    
    if source_file.suffix.lower() not in valid_image_ext:
        raise ValueError(f"Unsupported image format: {source_file.suffix}")
    
    if target_file.suffix.lower() not in valid_video_ext:
        raise ValueError(f"Unsupported video format: {target_file.suffix}")
    
    logger.info(f"✅ Input validation passed")
    logger.info(f"   Source: {source_path}")
    logger.info(f"   Target: {target_path}")

def run_facefusion(source_path, target_path, output_path, headless=True):
    """Execute FaceFusion face swapping"""
    try:
        # Generate timestamp for output if not provided
        if not output_path:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_name = f"faceswap_result_{timestamp}.mp4"
            output_path = f"/app/output/{output_name}"
        
        # Prepare FaceFusion command
        cmd = [
            "python", "/app/facefusion.py"
        ]
        
        if headless:
            cmd.append("headless-run")
        else:
            cmd.append("run")
        
        # Add source and target
        cmd.extend([
            "--source", source_path,
            "--target", target_path,
            "--output", output_path
        ])
        
        # GPU acceleration options
        cmd.extend([
            "--execution-providers", "cuda",  # Use CUDA
            "--execution-device-id", "0"      # GPU 0
        ])
        
        logger.info("🚀 Starting FaceFusion face swapping...")
        logger.info(f"Command: {' '.join(cmd)}")
        
        start_time = time.time()
        
        # Execute FaceFusion
        result = subprocess.run(
            cmd,
            cwd="/app",
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Check result
        if result.returncode == 0:
            logger.info(f"✅ Face swapping completed in {processing_time:.2f}s")
            logger.info(f"   Output saved to: {output_path}")
            if result.stdout:
                logger.info(f"   Output: {result.stdout}")
        else:
            logger.error(f"❌ Face swapping failed")
            logger.error(f"   Error: {result.stderr}")
            raise RuntimeError(f"FaceFusion failed: {result.stderr}")
        
        return output_path
        
    except subprocess.TimeoutExpired:
        logger.error("❌ FaceFusion timed out (>10 minutes)")
        raise
    except Exception as e:
        logger.error(f"❌ Face swapping failed: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description="FaceFusion CLI for face swapping")
    
    # Required arguments
    parser.add_argument("--source", required=True, help="Path to source face image")
    parser.add_argument("--target", required=True, help="Path to target video")
    
    # Optional arguments
    parser.add_argument("--output", help="Output video path (auto-generated if not specified)")
    parser.add_argument("--gui", action="store_true", help="Enable GUI mode (default: headless)")
    
    args = parser.parse_args()
    
    try:
        logger.info("🎭 FaceFusion CLI Starting...")
        
        # Check GPU
        if not check_gpu():
            logger.warning("GPU not available, performance may be degraded")
        
        # Validate inputs
        validate_inputs(args.source, args.target)
        
        # Run face swapping
        output_path = run_facefusion(
            args.source, 
            args.target, 
            args.output,
            headless=not args.gui
        )
        
        logger.info("🎉 FaceFusion completed successfully!")
        print(f"Output: {output_path}")
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()