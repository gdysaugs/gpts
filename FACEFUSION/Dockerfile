# Multi-stage build for FaceFusion 3.3.0 with Conda environment
# Stage 1: Base CUDA environment with Conda
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04 as base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=/opt/conda/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/wsl/lib:$LD_LIBRARY_PATH

# Install system dependencies including CUDA libraries
RUN apt-get update && apt-get install -y \
    wget \
    git \
    git-lfs \
    build-essential \
    cmake \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1-mesa-glx \
    libglu1-mesa \
    libxml2-dev \
    libxslt1-dev \
    libtbb-dev \
    ffmpeg \
    curl \
    unzip \
    ca-certificates \
    libcublas11 \
    libcusparse11 \
    libcurand10 \
    libcufft10 \
    libcudart11.0 \
    && rm -rf /var/lib/apt/lists/*
# Install cuDNN separately to avoid hold package conflicts
RUN apt-get update && apt-get install -y --allow-change-held-packages libcudnn8 && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -afy

# Stage 2: Download and setup FaceFusion
FROM base as downloader

WORKDIR /app

# Clone FaceFusion 3.3.0
RUN git clone https://github.com/facefusion/facefusion.git . && \
    git checkout tags/3.3.0 -b v3.3.0

# Create conda environment with Python 3.11
# Use environment variable to bypass license prompt
ENV CONDA_AUTO_UPDATE_CONDA=false
ENV CONDA_CHANNEL_PRIORITY=flexible
RUN /opt/conda/bin/conda init bash && \
    /opt/conda/bin/conda create -n facefusion python=3.11 -y --override-channels -c conda-forge && \
    echo "source /opt/conda/bin/activate facefusion" > ~/.bashrc

# Activate conda environment and install dependencies
SHELL ["/bin/bash", "-c"]
RUN source /opt/conda/bin/activate facefusion && \
    pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt && \
    pip install onnxruntime-gpu==1.19.2

# Create CUDA library compatibility links for ONNXRuntime
RUN ln -sf /usr/local/cuda/lib64/libcudart.so.12 /usr/local/cuda/lib64/libcudart.so.11.0 || true
RUN ln -sf /usr/lib/x86_64-linux-gnu/libcudart.so.12 /usr/lib/x86_64-linux-gnu/libcudart.so.11.0 || true
# Create cuDNN 9 compatibility links 
RUN ln -sf /usr/lib/x86_64-linux-gnu/libcudnn.so.8 /usr/lib/x86_64-linux-gnu/libcudnn.so.9 || true
RUN ln -sf /usr/local/cuda/lib64/libcudnn.so.8 /usr/local/cuda/lib64/libcudnn.so.9 || true

# Stage 3: Model downloader
FROM downloader as model-downloader

WORKDIR /app

# Create model directories
RUN mkdir -p .assets/models

# Download models from facefusion-assets releases
# Note: These URLs should be updated based on actual 3.3.0 model releases
RUN cd .assets/models && \
    # Face detector models
    wget -q https://github.com/facefusion/facefusion-assets/releases/download/models-3.3.0/yoloface_8n.onnx && \
    wget -q https://github.com/facefusion/facefusion-assets/releases/download/models-3.3.0/retinaface_10g.onnx && \
    # Face recognizer models
    wget -q https://github.com/facefusion/facefusion-assets/releases/download/models-3.3.0/arcface_w600k_r50.onnx && \
    wget -q https://github.com/facefusion/facefusion-assets/releases/download/models-3.3.0/arcface_simswap.onnx && \
    # Face swapper models
    wget -q https://github.com/facefusion/facefusion-assets/releases/download/models-3.3.0/inswapper_128.onnx && \
    wget -q https://github.com/facefusion/facefusion-assets/releases/download/models-3.3.0/inswapper_128_fp16.onnx && \
    # Face enhancer models
    wget -q https://github.com/facefusion/facefusion-assets/releases/download/models-3.3.0/codeformer.onnx && \
    wget -q https://github.com/facefusion/facefusion-assets/releases/download/models-3.3.0/gfpgan_1.2.onnx && \
    wget -q https://github.com/facefusion/facefusion-assets/releases/download/models-3.3.0/gfpgan_1.3.onnx && \
    wget -q https://github.com/facefusion/facefusion-assets/releases/download/models-3.3.0/gfpgan_1.4.onnx && \
    # Frame enhancer models
    wget -q https://github.com/facefusion/facefusion-assets/releases/download/models-3.3.0/real_esrgan_2x.onnx && \
    wget -q https://github.com/facefusion/facefusion-assets/releases/download/models-3.3.0/real_esrgan_4x.onnx && \
    wget -q https://github.com/facefusion/facefusion-assets/releases/download/models-3.3.0/real_esrgan_8x.onnx && \
    # Face mask models
    wget -q https://github.com/facefusion/facefusion-assets/releases/download/models-3.3.0/face_parser.onnx && \
    wget -q https://github.com/facefusion/facefusion-assets/releases/download/models-3.3.0/face_occluder.onnx && \
    # Age modifier models
    wget -q https://github.com/facefusion/facefusion-assets/releases/download/models-3.3.0/styleganex_age.onnx || true

# Stage 4: Runtime environment
FROM base as runtime

# Copy from previous stages
COPY --from=model-downloader /app /app
COPY --from=model-downloader /opt/conda /opt/conda

WORKDIR /app

# Create directories for input/output
RUN mkdir -p /app/input /app/output /app/temp

# Set conda environment
SHELL ["/bin/bash", "-c"]

# Create entrypoint script
RUN echo '#!/bin/bash' > /entrypoint.sh && \
    echo 'source /opt/conda/bin/activate facefusion' >> /entrypoint.sh && \
    echo 'exec "$@"' >> /entrypoint.sh && \
    chmod +x /entrypoint.sh

# Set entrypoint
ENTRYPOINT ["/entrypoint.sh"]

# Default command
CMD ["python", "facefusion.py", "--help"]