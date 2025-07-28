#!/bin/bash

# FaceFusion 3.3.0 model downloader script
# Downloads all required models from facefusion-assets repository

set -e

echo "Starting FaceFusion 3.3.0 model download..."

# Create model directory
MODEL_DIR=".assets/models"
mkdir -p $MODEL_DIR
cd $MODEL_DIR

# Base URL for model downloads
BASE_URL="https://github.com/facefusion/facefusion-assets/releases/download"

# Download face detector models
echo "Downloading face detector models..."
wget -q --show-progress "${BASE_URL}/models/yoloface_8n.onnx" -O yoloface_8n.onnx
wget -q --show-progress "${BASE_URL}/models-2.6.0/retinaface_10g.onnx" -O retinaface_10g.onnx
wget -q --show-progress "${BASE_URL}/models-2.6.0/scrfd_2.5g.onnx" -O scrfd_2.5g.onnx

# Download face recognizer models
echo "Downloading face recognizer models..."
wget -q --show-progress "${BASE_URL}/models-2.6.0/arcface_w600k_r50.onnx" -O arcface_w600k_r50.onnx
wget -q --show-progress "${BASE_URL}/models-2.5.0/arcface_simswap.onnx" -O arcface_simswap.onnx

# Download face swapper models (including 3.3.0 hyperswap)
echo "Downloading face swapper models..."
wget -q --show-progress "${BASE_URL}/models/inswapper_128.onnx" -O inswapper_128.onnx
wget -q --show-progress "${BASE_URL}/models/inswapper_128_fp16.onnx" -O inswapper_128_fp16.onnx
wget -q --show-progress "${BASE_URL}/models-2.3.0/simswap_256.onnx" -O simswap_256.onnx
wget -q --show-progress "${BASE_URL}/models-2.5.0/simswap_512_unofficial.onnx" -O simswap_512_unofficial.onnx
# 3.3.0 models
wget -q --show-progress "${BASE_URL}/models-3.3.0/hyperswap_1a_256.onnx" -O hyperswap_1a_256.onnx
wget -q --show-progress "${BASE_URL}/models-3.3.0/hyperswap_1b_256.onnx" -O hyperswap_1b_256.onnx

# Download face enhancer models
echo "Downloading face enhancer models..."
wget -q --show-progress "${BASE_URL}/models-2.3.0/codeformer.onnx" -O codeformer.onnx
wget -q --show-progress "${BASE_URL}/models/gfpgan_1.2.onnx" -O gfpgan_1.2.onnx
wget -q --show-progress "${BASE_URL}/models/gfpgan_1.3.onnx" -O gfpgan_1.3.onnx
wget -q --show-progress "${BASE_URL}/models/gfpgan_1.4.onnx" -O gfpgan_1.4.onnx

# Download frame enhancer models
echo "Downloading frame enhancer models..."
wget -q --show-progress "${BASE_URL}/models/real_esrgan_2x.onnx" -O real_esrgan_2x.onnx
wget -q --show-progress "${BASE_URL}/models/real_esrgan_4x.onnx" -O real_esrgan_4x.onnx
wget -q --show-progress "${BASE_URL}/models/real_esrgan_8x.onnx" -O real_esrgan_8x.onnx

# Download face mask models
echo "Downloading face mask models..."
wget -q --show-progress "${BASE_URL}/models/face_parser.onnx" -O face_parser.onnx
wget -q --show-progress "${BASE_URL}/models-2.5.0/face_occluder.onnx" -O face_occluder.onnx

# Download age modifier models
echo "Downloading age modifier models..."
wget -q --show-progress "${BASE_URL}/models-2.6.0/styleganex_age.onnx" -O styleganex_age.onnx

# Download expression restorer models
echo "Downloading expression restorer models..."
wget -q --show-progress "${BASE_URL}/models-2.6.0/liveportrait.onnx" -O liveportrait.onnx

# Download frame colorizer models
echo "Downloading frame colorizer models..."
wget -q --show-progress "${BASE_URL}/models-2.4.0/ddcolor.onnx" -O ddcolor.onnx
wget -q --show-progress "${BASE_URL}/models-2.4.0/ddcolor_artistic.onnx" -O ddcolor_artistic.onnx
wget -q --show-progress "${BASE_URL}/models-2.4.0/deoldify.onnx" -O deoldify.onnx
wget -q --show-progress "${BASE_URL}/models-2.4.0/deoldify_artistic.onnx" -O deoldify_artistic.onnx
wget -q --show-progress "${BASE_URL}/models-2.4.0/deoldify_stable.onnx" -O deoldify_stable.onnx

# Download lip syncer models
echo "Downloading lip syncer models..."
wget -q --show-progress "${BASE_URL}/models-2.3.0/wav2lip_gan.onnx" -O wav2lip_gan.onnx

# Download 3.3.0 specific models
echo "Downloading FaceFusion 3.3.0 specific models..."
wget -q --show-progress "${BASE_URL}/models-3.3.0/edtalk_256.onnx" -O edtalk_256.onnx

echo "Model download completed!"
echo "Models downloaded to: $(pwd)"

# List downloaded models
echo ""
echo "Downloaded models:"
ls -lh *.onnx 2>/dev/null || echo "No models found"