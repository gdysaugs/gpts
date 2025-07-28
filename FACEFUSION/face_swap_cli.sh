#!/bin/bash

# FaceFusion 3.3.0 CLI Face Swap Script
# Direct CLI execution with all options

# Check arguments
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <source_image> <target_image/video> [output_path]"
    echo ""
    echo "Examples:"
    echo "  $0 source.jpg target.jpg"
    echo "  $0 source.jpg target.jpg output.jpg"
    echo "  $0 source.jpg target.mp4 output.mp4"
    exit 1
fi

SOURCE=$1
TARGET=$2
OUTPUT=${3:-"output/result_$(date +%Y%m%d_%H%M%S).${TARGET##*.}"}

# Detect if target is video
if [[ "$TARGET" == *.mp4 ]] || [[ "$TARGET" == *.avi ]] || [[ "$TARGET" == *.mov ]]; then
    IS_VIDEO=true
else
    IS_VIDEO=false
fi

echo "FaceFusion 3.3.0 Face Swap"
echo "========================="
echo "Source: $SOURCE"
echo "Target: $TARGET"
echo "Output: $OUTPUT"
echo "Type: $([ "$IS_VIDEO" = true ] && echo "Video" || echo "Image")"
echo ""

# Build command
if [ "$IS_VIDEO" = true ]; then
    # Video face swap
    docker run --gpus all --rm \
        --privileged \
        -v /usr/lib/wsl:/usr/lib/wsl \
        -e LD_LIBRARY_PATH=/usr/lib/wsl/lib \
        -e FACEFUSION_SKIP_DOWNLOAD=1 \
        -v $(pwd):/workspace \
        --shm-size=8gb \
        facefusion:3.3.0 \
        python facefusion.py headless-run \
        --source "/workspace/$SOURCE" \
        --target "/workspace/$TARGET" \
        --output "/workspace/$OUTPUT" \
        --processors face_swapper face_enhancer \
        --face-detector-model yoloface_8n \
        --face-recognizer-model arcface_simswap \
        --face-swapper-model inswapper_128 \
        --face-enhancer-model gfpgan_1.4 \
        --face-enhancer-blend 80 \
        --face-detector-score 0.5 \
        --face-landmarker-score 0.5 \
        --output-video-encoder libx264 \
        --output-video-quality 80 \
        --output-video-fps 30 \
        --video-memory-strategy moderate \
        --execution-providers cuda \
        --execution-thread-count 4 \
        --execution-queue-count 1 \
        --skip-download \
        --skip-audio
else
    # Image face swap
    docker run --gpus all --rm \
        --privileged \
        -v /usr/lib/wsl:/usr/lib/wsl \
        -e LD_LIBRARY_PATH=/usr/lib/wsl/lib \
        -e FACEFUSION_SKIP_DOWNLOAD=1 \
        -v $(pwd):/workspace \
        --shm-size=8gb \
        facefusion:3.3.0 \
        python facefusion.py headless-run \
        --source "/workspace/$SOURCE" \
        --target "/workspace/$TARGET" \
        --output "/workspace/$OUTPUT" \
        --processors face_swapper face_enhancer \
        --face-detector-model yoloface_8n \
        --face-recognizer-model arcface_simswap \
        --face-swapper-model inswapper_128 \
        --face-enhancer-model gfpgan_1.4 \
        --face-enhancer-blend 80 \
        --face-detector-score 0.5 \
        --face-landmarker-score 0.5 \
        --output-image-quality 90 \
        --execution-providers cuda \
        --execution-thread-count 4 \
        --execution-queue-count 1 \
        --skip-download \
        --skip-audio
fi

echo ""
echo "Face swap completed! Output saved to: $OUTPUT"