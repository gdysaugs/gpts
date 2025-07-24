#!/bin/bash

# FaceFusion Face Swap Test Script
# Kanna Hashimoto face + Screen recording video
# Optimized for WSL2 Ubuntu + Docker + RTX GPU

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }
print_face() { echo -e "${PURPLE}[FACE]${NC} $1"; }

# Check if Docker is running
check_docker() {
    if ! docker info >/dev/null 2>&1; then
        print_error "Docker is not running. Please start Docker."
        exit 1
    fi
    print_info "Docker is running ✅"
}

# Check if NVIDIA Container Toolkit is available
check_nvidia() {
    if ! docker run --rm --gpus all nvidia/cuda:12.2.2-runtime-ubuntu22.04 nvidia-smi >/dev/null 2>&1; then
        print_error "NVIDIA Container Toolkit not available"
        print_info "Install with: sudo apt install nvidia-container-toolkit"
        exit 1
    fi
    print_info "NVIDIA GPU support detected ✅"
}

# Validate input files
validate_inputs() {
    local source_file="input/source_face.jpg"
    local target_file="input/target_video.mp4"
    
    if [ ! -f "$source_file" ]; then
        print_error "Source face image not found: $source_file"
        exit 1
    fi
    
    if [ ! -f "$target_file" ]; then
        print_error "Target video not found: $target_file"
        exit 1
    fi
    
    # Check file sizes
    local src_size=$(stat -c%s "$source_file" 2>/dev/null || echo 0)
    local tgt_size=$(stat -c%s "$target_file" 2>/dev/null || echo 0)
    local src_mb=$(echo "scale=1; $src_size/1048576" | bc 2>/dev/null || echo "unknown")
    local tgt_mb=$(echo "scale=1; $tgt_size/1048576" | bc 2>/dev/null || echo "unknown")
    
    print_success "✓ Kanna Hashimoto face: ${src_mb}MB"
    print_success "✓ Screen recording video: ${tgt_mb}MB"
}

# Build Docker image if not exists
build_image() {
    if ! docker image inspect facefusion:v1 >/dev/null 2>&1; then
        print_info "Building FaceFusion Docker image..."
        print_warning "This may take 10-20 minutes for first build..."
        docker build -t facefusion:v1 .
        print_success "Docker image built successfully"
    else
        print_info "Docker image facefusion:v1 already exists ✅"
    fi
}

# Run face swap test
run_face_swap() {
    print_face "🎭 Starting Kanna Hashimoto face swap test..."
    print_info "Swapping Kanna Hashimoto's face into screen recording video"
    
    # Create output timestamp
    local start_time=$(date +%s)
    print_info "Start time: $(date)"
    
    # WSL2 optimized Docker command with proper GPU access
    docker run --gpus all --rm \
        --privileged \
        -v $(pwd)/input:/app/input \
        -v $(pwd)/output:/app/output \
        -v $(pwd)/logs:/app/logs \
        -v $(pwd)/scripts:/app/scripts \
        -v /usr/lib/wsl:/usr/lib/wsl \
        -e LD_LIBRARY_PATH=/usr/lib/wsl/lib \
        -e NVIDIA_VISIBLE_DEVICES=all \
        -e CUDA_VISIBLE_DEVICES=0 \
        facefusion:v1 \
        python /app/scripts/facefusion_cli.py \
        --source /app/input/source_face.jpg \
        --target /app/input/target_video.mp4
    
    local exit_code=$?
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    if [ $exit_code -eq 0 ]; then
        print_success "🎉 Face swap completed successfully!"
        print_info "Processing time: ${duration}s"
        
        # Show output files
        print_info "Generated files:"
        ls -la output/ | grep -E "\.(mp4|mov|avi)$" | while read line; do
            print_success "  🎥 $line"
        done
        
    else
        print_error "❌ Face swap failed (exit code: $exit_code)"
        print_info "Check logs for details:"
        print_info "  tail -n 50 logs/facefusion_cli.log"
        exit $exit_code
    fi
}

# Show results
show_results() {
    print_face "📊 Face Swap Results:"
    
    # Find the most recent output file
    local latest_file=$(ls -t output/*.mp4 2>/dev/null | head -n1)
    
    if [ -n "$latest_file" ]; then
        local file_size=$(stat -c%s "$latest_file" 2>/dev/null || echo 0)
        local file_mb=$(echo "scale=2; $file_size/1048576" | bc 2>/dev/null || echo "unknown")
        
        print_success "✅ Output: $(basename "$latest_file")"
        print_success "✅ Size: ${file_mb}MB"
        
        # Try to get video info if ffprobe is available
        if command -v ffprobe >/dev/null 2>&1; then
            local duration=$(ffprobe -v quiet -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$latest_file" 2>/dev/null || echo "unknown")
            if [ "$duration" != "unknown" ]; then
                print_success "✅ Duration: $(printf "%.1f" $duration)s"
            fi
        fi
        
        print_face "🎥 Generated Kanna Hashimoto face-swapped video!"
        print_info "You can now play: $latest_file"
    else
        print_warning "No output video files found"
    fi
}

# Main execution
main() {
    print_face "🎭 FaceFusion Face Swap Test"
    print_face "Kanna Hashimoto face → Screen recording video"
    echo
    
    # Change to FaceFusion directory
    cd "$(dirname "$0")/.." || exit 1
    
    # Create necessary directories
    mkdir -p input output logs scripts
    
    # Verification steps
    check_docker
    check_nvidia
    validate_inputs
    build_image
    
    # Run face swap test
    run_face_swap
    
    # Show results
    show_results
    
    print_face "🎌 Test completed! Enjoy your face-swapped video!"
}

# Execute main function
main "$@"