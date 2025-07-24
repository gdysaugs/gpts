# GPTs Multi-Project AI System - Complete Setup Guide

このガイドは、新しいPCでGitHubリポジトリをクローンして、完全に動作するAIシステムを1から構築するための包括的な手順書です。

## 🎯 システム概要

3つの専用AIコンポーネントを統合したマルチプロジェクトシステム：
- **GPT-SoVITS**: 日本語音声クローニング（感情表現対応）
- **LlamaCPP**: GPU最適化ローカルLLMチャットシステム  
- **Wav2Lip**: リップシンク動画生成（YOLO11顔検出）

## 📋 前提条件

### ハードウェア要件
- **OS**: Windows 10/11 with WSL2 Ubuntu 22.04
- **GPU**: NVIDIA RTX 3050以上（4GB VRAM推奨）
- **RAM**: 16GB以上推奨
- **Storage**: 50GB以上の空き容量

### ソフトウェア要件
- WSL2 with Ubuntu 22.04
- Docker Desktop with WSL2 integration
- NVIDIA Container Toolkit
- Git

## 🚀 Step 1: 基本環境セットアップ

### 1.1 WSL2 Ubuntu 22.04のインストール
```powershell
# Windows PowerShell (管理者権限)
wsl --install -d Ubuntu-22.04
wsl --set-default Ubuntu-22.04
```

### 1.2 基本パッケージのインストール
```bash
# WSL2 Ubuntu内で実行
sudo apt update && sudo apt upgrade -y
sudo apt install -y curl wget git build-essential python3 python3-pip
```

### 1.3 Docker Desktop + NVIDIA Container Toolkitのセットアップ
```bash
# Docker GPG キーとリポジトリの追加
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Docker Engine インストール
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io

# ユーザーをdockerグループに追加
sudo usermod -aG docker $USER

# NVIDIA Container Toolkit インストール
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt update && sudo apt install -y nvidia-container-toolkit
sudo systemctl restart docker

# 再ログインまたは新しいシェルで確認
docker --version
nvidia-smi
```

## 🔽 Step 2: リポジトリクローンとディレクトリ準備

### 2.1 GitHubからクローン
```bash
cd /home/$USER
git clone https://github.com/gdysaugs/gpts.git
cd gpts
chmod +x setup.sh
```

### 2.2 必要ディレクトリの作成
```bash
# 各プロジェクトの必要ディレクトリを作成
mkdir -p Gptsovits/models/v4/GPT-SoVITS/gpt-sovits-ja-h
mkdir -p Gptsovits/input Gptsovits/output Gptsovits/logs
mkdir -p llamacpp/models llamacpp/logs/sessions
mkdir -p faster-sadtalker/checkpoints faster-sadtalker/gfpgan/weights
mkdir -p sadtalker/faster-SadTalker-API/checkpoints sadtalker/faster-SadTalker-API/gfpgan/weights
mkdir -p "super wav2lip/models" "super wav2lip/src" "super wav2lip/input" "super wav2lip/output"
mkdir -p "super wav2lip_backup_gradio3.50/models" "super wav2lip_backup_gradio3.50/src"
mkdir -p wav2lips/gfpgan/weights
```

## 📦 Step 3: 必須モデルファイルのダウンロード

### 3.1 GPT-SoVITS Japanese Emotional Model
```bash
# 148MB - 日本語感情表現モデル
cd Gptsovits/models/v4/GPT-SoVITS/gpt-sovits-ja-h/
wget https://huggingface.co/spaces/keisuke/hscene-voice-clone/resolve/main/hscene-e17.ckpt
# または手動ダウンロード後にコピー:
# cp /mnt/c/Users/YourName/Downloads/hscene-e17.ckpt ./
```

### 3.2 LlamaCPP Quantized Model
```bash
# 4.26GB - 量子化LLMモデル
cd ../../../../../../llamacpp/models/
wget https://huggingface.co/TheBloke/Berghof-NSFW-7B-GGUF/resolve/main/Berghof-NSFW-7B.i1-Q4_K_S.gguf
# 大きなファイルのため、手動ダウンロード推奨:
# cp /mnt/c/Users/YourName/Downloads/Berghof-NSFW-7B.i1-Q4_K_S.gguf ./
```

### 3.3 SadTalker Models
```bash
cd ../../faster-sadtalker/checkpoints/
# 基本モデル群 (合計約2GB)
wget https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/mapping_00109-model.pth.tar
wget https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/mapping_00229-model.pth.tar
wget https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/SadTalker_V0.0.2_256.safetensors
wget https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/SadTalker_V0.0.2_512.safetensors
wget https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/auido2exp_00300-model.pth
wget https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/auido2pose_00140-model.pth
wget https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/facevid2vid_00189-model.pth.tar
wget https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/epoch_20.pth
wget https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/wav2lip.pth
wget https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/wav2lip_gan.pth
wget https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/s3fd-619a316812.pth

# GFPGAN Enhancement Models
cd ../gfpgan/weights/
wget https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.onnx
wget https://github.com/xinntao/facexlib/releases/download/v0.1.0/alignment_WFLW_4HG.pth
wget https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth
wget https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth
```

### 3.4 Super Wav2Lip Models（大容量 - 必要に応じて）
```bash
cd "../../../super wav2lip/"

# ⚠️ 大容量ファイル - 手動ダウンロードまたはSkip推奨
# wav2lip_onnx_models.zip (258MB) - ONNXモデル集
# wav2lip_face_occluder.zip (62MB) - 顔遮蔽モデル  
# wav2lip_seg_mask.zip (50MB) - セグメンテーションマスク
# wav2lip_insightface_func.zip - InsightFace機能

# 基本のPyTorchモデルのみダウンロード
mkdir -p models
cd models
wget https://github.com/Rudrabha/Wav2Lip/releases/download/v1.0.0/wav2lip.pth
wget https://github.com/Rudrabha/Wav2Lip/releases/download/v1.0.0/wav2lip_gan.pth
```

### 3.5 その他の必須ファイル
```bash
# SadTalker API用
cd ../../sadtalker/faster-SadTalker-API/checkpoints/
# 上記3.3と同じファイルをコピー
cp ../../../faster-sadtalker/checkpoints/* ./

# GFPGAN weights
cd ../gfpgan/weights/
cp ../../../../faster-sadtalker/gfpgan/weights/* ./

# Wav2Lips GFPGAN
cd ../../../wav2lips/gfpgan/weights/
wget https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth
wget https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth
```

## 🐳 Step 4: Dockerイメージのビルド

### 4.1 GPT-SoVITS
```bash
cd /home/$USER/gpts/Gptsovits
DOCKER_BUILDKIT=1 docker build -t gpt-sovits:v4 .
```

### 4.2 LlamaCPP
```bash
cd ../llamacpp
DOCKER_BUILDKIT=1 docker build -t llama-cpp-python:cuda .
```

### 4.3 Super Wav2Lip
```bash
cd "../super wav2lip"
DOCKER_BUILDKIT=1 docker build -t wav2lip-optimized:v1 .
```

### 4.4 ビルド確認
```bash
docker images | grep -E "gpt-sovits|llama-cpp|wav2lip"
```

## ✅ Step 5: システム動作確認

### 5.1 GPU アクセステスト
```bash
# ホストGPU確認
nvidia-smi

# Docker GPU アクセス確認
docker run --gpus all --rm --privileged \
  -v /usr/lib/wsl:/usr/lib/wsl -e LD_LIBRARY_PATH=/usr/lib/wsl/lib \
  gpt-sovits:v4 nvidia-smi
```

### 5.2 GPT-SoVITS テスト
```bash
cd /home/$USER/gpts/Gptsovits

# FastAPI サーバー起動 (20秒初期化)
docker run --gpus all -d -p 8000:8000 --name gpt-sovits-test \
  --privileged -v /usr/lib/wsl:/usr/lib/wsl -e LD_LIBRARY_PATH=/usr/lib/wsl/lib \
  -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output \
  -v $(pwd)/scripts:/app/scripts \
  -v $(pwd)/models/v4/GPT-SoVITS/gpt-sovits-ja-h:/app/GPT_SoVITS/pretrained_models/gpt-sovits-ja-h \
  gpt-sovits:v4 bash -c "pip install fastapi uvicorn python-multipart && python /app/scripts/fastapi_voice_server.py"

# 20秒待機後、音声生成テスト
sleep 20
curl -G "http://localhost:8000/clone-voice-simple" \
  --data-urlencode "ref_text=おはようございます" \
  --data-urlencode "target_text=音声生成のテストです" > output/test.wav

# クリーンアップ
docker stop gpt-sovits-test && docker rm gpt-sovits-test
```

### 5.3 LlamaCPP テスト
```bash
cd ../llamacpp

# CLI チャットテスト
docker run --gpus all --rm -it --privileged \
  -v /usr/lib/wsl:/usr/lib/wsl -e LD_LIBRARY_PATH=/usr/lib/wsl/lib \
  -v $(pwd)/models:/app/models -v $(pwd)/config:/app/config \
  -v $(pwd)/logs:/app/logs \
  llama-cpp-python:cuda python /app/scripts/chat_cli.py
```

### 5.4 Wav2Lip テスト
```bash
cd "../super wav2lip"

# テスト用ファイル準備（適切な動画と音声ファイルを用意）
# input/target_video.mp4 と input/reference_audio.wav

# 720p リップシンク生成テスト（5秒処理）
docker run --gpus all --rm --privileged \
  -v /usr/lib/wsl:/usr/lib/wsl -e LD_LIBRARY_PATH=/usr/lib/wsl/lib \
  -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output \
  -v $(pwd)/checkpoints:/app/checkpoints -v $(pwd):/app/host \
  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  wav2lip-optimized:v1 bash -c "
  mkdir -p temp && cp /app/host/inference.py /app/inference.py 
  python /app/inference.py \
  --checkpoint_path /app/checkpoints/wav2lip_gan.pth \
  --face /app/input/target_video.mp4 \
  --audio /app/input/reference_audio.wav \
  --outfile /app/output/result_test.mp4 \
  --out_height 720 \
  --quality Fast"
```

## 🔧 Step 6: トラブルシューティング

### 6.1 一般的な問題と解決策

#### Docker GPU アクセスエラー
```bash
# NVIDIA Container Toolkit 再インストール
sudo apt remove nvidia-container-toolkit
sudo apt install -y nvidia-container-toolkit
sudo systemctl restart docker
```

#### WSL2 GPU ライブラリエラー
```bash
# WSL2 GPU ドライバー確認
ls /usr/lib/wsl/lib/
# 必要なライブラリが存在することを確認
```

#### モデルファイルサイズ確認
```bash
# 重要なファイルサイズを確認
ls -lh Gptsovits/models/v4/GPT-SoVITS/gpt-sovits-ja-h/hscene-e17.ckpt  # 148MB
ls -lh llamacpp/models/Berghof-NSFW-7B.i1-Q4_K_S.gguf  # 4.26GB
ls -lh faster-sadtalker/checkpoints/  # 各種モデルファイル
```

### 6.2 パフォーマンス最適化

#### メモリ不足対策
```bash
# Docker メモリ制限調整
docker run --memory=12g --gpus all ...
```

#### GPU VRAM 最適化
```bash
# 低VRAM モード（LlamaCPP）
# config/model_config.yaml で n_gpu_layers を調整
```

## 📁 Step 7: ファイル構造確認

セットアップ完了後、以下の構造になっていることを確認：

```
gpts/
├── Gptsovits/
│   ├── models/v4/GPT-SoVITS/gpt-sovits-ja-h/hscene-e17.ckpt (148MB)
│   ├── input/ output/ logs/
│   └── scripts/
├── llamacpp/
│   ├── models/Berghof-NSFW-7B.i1-Q4_K_S.gguf (4.26GB)
│   ├── config/ logs/
│   └── src/
├── faster-sadtalker/
│   ├── checkpoints/ (約2GB)
│   ├── gfpgan/weights/
│   └── src/
├── "super wav2lip"/
│   ├── models/ (基本モデル)
│   ├── src/ (ONNX models - optional)
│   └── input/ output/
└── wav2lips/
    ├── gfpgan/weights/
    └── models/
```

## 🎉 Step 8: 運用開始

システム全体が正常に動作することを確認後、以下のコマンドで各サービスを起動：

### 日常的な使用コマンド

#### GPT-SoVITS 音声生成
```bash
cd /home/$USER/gpts/Gptsovits
# FastAPI サーバー起動
docker run --gpus all -d -p 8000:8000 --name gpt-sovits-prod [上記と同じオプション]
# 音声生成
curl -G "http://localhost:8000/clone-voice-simple" --data-urlencode "ref_text=..." --data-urlencode "target_text=..." > output/voice.wav
```

#### LlamaCPP チャット
```bash
cd /home/$USER/gpts/llamacpp
docker run --gpus all --rm -it [上記と同じオプション] llama-cpp-python:cuda python /app/scripts/chat_cli.py
```

#### Wav2Lip 動画生成
```bash
cd "/home/$USER/gpts/super wav2lip"
docker run --gpus all --rm [上記と同じオプション] wav2lip-optimized:v1 [inference.py コマンド]
```

## 📋 必要なファイル一覧

### 手動ダウンロードが必要な大容量ファイル
| ファイル名 | サイズ | 用途 | ダウンロード先 |
|-----------|--------|------|---------------|
| hscene-e17.ckpt | 148MB | GPT-SoVITS日本語モデル | HuggingFace |
| Berghof-NSFW-7B.i1-Q4_K_S.gguf | 4.26GB | LlamaCPP量子化モデル | HuggingFace |
| SadTalker checkpoints | ~2GB | 顔生成モデル群 | GitHub Releases |
| wav2lip_onnx_models.zip | 258MB | Wav2Lip ONNXモデル | 元リポジトリ |
| wav2lip_face_occluder.zip | 62MB | 顔遮蔽機能 | 元リポジトリ |

### 自動ダウンロード可能ファイル
- 基本PyTorchモデル (wav2lip.pth, wav2lip_gan.pth)
- GFPGAN enhancement models
- Face detection models

---

💡 **Tips**: 
- 初回セットアップは2-3時間かかる場合があります
- モデルファイルのダウンロードは安定したネットワーク環境で実行してください
- GPU ドライバーとCUDA バージョンの互換性を事前に確認してください

🔗 **参考リンク**:
- [CLAUDE.md](./CLAUDE.md) - 詳細な技術仕様とコマンドリファレンス
- [GitHub Repository](https://github.com/gdysaugs/gpts) - 最新のソースコード