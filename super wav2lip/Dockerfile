# Super Wav2Lip Dockerfile  
# wav2lip-onnx-HQベース + 既存wav2lipsプロジェクトのノウハウ適用

FROM ubuntu:22.04

# 環境変数設定
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# システムパッケージインストール (wav2lipsプロジェクトを参考)
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    ffmpeg \
    cmake \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgstreamer1.0-0 \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    # dlib用追加依存関係
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    && rm -rf /var/lib/apt/lists/*

# 作業ディレクトリ設定
WORKDIR /app

# Python symlink作成
RUN ln -sf /usr/bin/python3 /usr/bin/python

# CMakeバージョン確認
RUN cmake --version

# Python dependencies (wav2lipsスタイルの段階的インストール)
RUN pip3 install --no-cache-dir --upgrade pip

# PyTorch GPU版 (CUDA 12.1対応)
RUN pip3 install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# wav2lip-onnx-HQ準拠 - 元の設定に復元

# 基本ライブラリ (NumPy互換性修正)
RUN pip3 install --no-cache-dir \
    "numpy<2.0" \
    opencv-python==4.8.0.76 \
    scipy \
    tqdm \
    librosa \
    numba \
    imutils \
    # ONNX関連 (TensorRT対応版)
    onnxruntime-gpu>=1.16.0 \
    # 画像処理
    Pillow>=9.0.0 \
    scikit-image \
    imageio \
    imageio-ffmpeg \
    # 音声処理追加
    resampy \
    python_speech_features \
    pydub \
    # ユーティリティ
    yacs \
    ffmpeg-python \
    PyYAML \
    # YOLO関連
    ultralytics>=8.3.0

# TensorRT高速化ライブラリ（ホスト競合回避、pip版のみ）
RUN pip3 install --no-cache-dir \
    tensorrt \
    pycuda \
    cuda-python

# dlibを最後にインストール (個別でエラーハンドリング)
RUN pip3 install --no-cache-dir dlib>=19.24.0 || echo "dlib installation failed, continuing..."

# wav2lip-onnx-HQソースをダウンロード
RUN git clone https://github.com/instant-high/wav2lip-onnx-HQ.git /app/original_source

# アプリケーションソースをコピー  
COPY src/ /app/src/
COPY config/ /app/config/

# 必要なディレクトリ作成
RUN mkdir -p /app/input /app/output /app/models /app/temp /app/logs /app/checkpoints

# 実行権限設定
RUN chmod +x /app/src/*.py

# CUDA ライブラリパス設定
ENV CUDA_LIB_PATH="/usr/local/lib/python3.10/dist-packages/nvidia"
ENV LD_LIBRARY_PATH="${CUDA_LIB_PATH}/cublas/lib:${CUDA_LIB_PATH}/cudnn/lib:${CUDA_LIB_PATH}/cufft/lib:${CUDA_LIB_PATH}/curand/lib:${CUDA_LIB_PATH}/cusolver/lib:${CUDA_LIB_PATH}/cusparse/lib:${CUDA_LIB_PATH}/nccl/lib:${CUDA_LIB_PATH}/nvtx/lib:${LD_LIBRARY_PATH}"

# システム確認スクリプト
RUN echo '#!/bin/bash\necho "=== Super Wav2Lip システム確認 ==="\necho "CUDA Libraries:"; find /usr/local/lib/python3.10/dist-packages/nvidia -name "*.so*" | head -5\npython3 -c "import torch; print(f\"PyTorch version: {torch.__version__}\")"' > /app/check_system.sh && \
    chmod +x /app/check_system.sh

# デフォルトコマンド
CMD ["python3", "/app/src/lipsync_cli.py", "--help"]