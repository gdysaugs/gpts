#!/bin/bash

# FaceFusion 3.3.0 Test Runner Script
# Builds and runs face swap tests with WSL2 GPU support

set -e

echo "FaceFusion 3.3.0 Test Runner"
echo "=========================="

# Build Docker image
echo "Building Docker image..."
docker build -t facefusion:3.3.0 .

# Check if test images exist
if [ ! -f "./input/source.jpg" ] || [ ! -f "./input/target.jpg" ]; then
    echo ""
    echo "⚠ Warning: Test images not found!"
    echo "Please add the following files:"
    echo "  - ./input/source.jpg (face to swap from)"
    echo "  - ./input/target.jpg (image to swap face into)"
    echo "  - ./input/target.mp4 (optional: video to swap face into)"
    echo ""
    echo "You can use any portrait images for testing."
    exit 1
fi

# Run face swap test
echo ""
echo "Running face swap test..."
docker run --device=/dev/nvidiactl --device=/dev/nvidia-uvm --device=/dev/nvidia0 --rm \
    --privileged \
    -v /usr/lib/wsl:/usr/lib/wsl \
    -e LD_LIBRARY_PATH=/usr/lib/wsl/lib \
    -e FACEFUSION_SKIP_DOWNLOAD=1 \
    -v $(pwd)/input:/app/input \
    -v $(pwd)/output:/app/output \
    -v $(pwd)/temp:/app/temp \
    -v $(pwd)/scripts:/app/scripts \
    -v $(pwd)/models:/app/.assets/models \
    --shm-size=8gb \
    facefusion:3.3.0 \
    python /app/scripts/test_face_swap.py

echo ""
echo "Test completed! Check the output directory for results."