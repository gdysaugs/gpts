#!/usr/bin/env python3
"""
Patch inference_onnxModel.py to support Anime4K enhancer
"""

def patch_inference_file():
    """Add Anime4K support to inference_onnxModel.py"""
    
    # Read the original file
    with open('/app/original_source/inference_onnxModel.py', 'r') as f:
        content = f.read()
    
    # 1. Add anime4k to choices
    content = content.replace(
        "choices=['none', 'gpen', 'gfpgan', 'codeformer', 'restoreformer']",
        "choices=['none', 'gpen', 'gfpgan', 'codeformer', 'restoreformer', 'anime4k']"
    )
    
    # 2. Add anime4k enhancer implementation
    gfpgan_block = """if args.enhancer == 'gfpgan':
\t\tfrom enhancers.GFPGAN.GFPGAN import GFPGAN
\t\tenhancer = GFPGAN(model_path="enhancers/GFPGAN/GFPGANv1.4.onnx", device=device)"""
    
    anime4k_block = """if args.enhancer == 'anime4k':
\t\timport sys
\t\tsys.path.append('/app/models/enhancers/Anime4K')
\t\tfrom anime4k_enhancer import Anime4KEnhancer
\t\tenhancer = Anime4KEnhancer(model_path="/app/models/enhancers/Anime4K/Restore_Soft_M.onnx", device=device)

if args.enhancer == 'gfpgan':
\t\tfrom enhancers.GFPGAN.GFPGAN import GFPGAN
\t\tenhancer = GFPGAN(model_path="enhancers/GFPGAN/GFPGANv1.4.onnx", device=device)"""
    
    content = content.replace(gfpgan_block, anime4k_block)
    
    # Write the patched file
    with open('/app/original_source/inference_onnxModel.py', 'w') as f:
        f.write(content)
    
    print("✅ Anime4K patch applied successfully!")

if __name__ == "__main__":
    patch_inference_file()