# 🤖 Complete Model Download Guide
## 完全モデルダウンロードガイド

このガイドに従って、全プロジェクトのモデルファイルを完璧にダウンロードできます。

## 🚨 重要：クローン後に必ず実行してください

```bash
git clone https://github.com/gdysaugs/gpts.git
cd gpts
chmod +x setup.sh
./setup.sh
```

## 📋 必要なモデルファイル一覧

### 1. GPT-SoVITS (音声クローニング)
- **標準v2モデル**: 自動ダウンロード（Gptsovits/scripts/download_models.sh）
- **日本語感情モデル**: **手動ダウンロード必須**

#### 日本語感情モデル（手動）
```bash
# WindowsからWSL2にコピーする場合
cp /mnt/c/Users/adama/Downloads/hscene-e17.ckpt Gptsovits/models/v4/GPT-SoVITS/gpt-sovits-ja-h/

# または直接ダウンロード（リンクが有効な場合）
cd Gptsovits/models/v4/GPT-SoVITS/gpt-sovits-ja-h/
# このファイルは通常Hugging FaceやGitHub Releasesからダウンロード
# 具体的なリンクは提供元に依存
```

**ファイルサイズ確認**:
- `hscene-e17.ckpt`: 148MB

### 2. LlamaCPP (ローカルLLM)
- **Berghof-NSFW-7B量子化モデル**: **手動ダウンロード必須**

#### LlamaCPP量子化モデル（手動）
```bash
# Hugging Faceからダウンロード
cd llamacpp/models/

# Method 1: Hugging Face CLI (推奨)
pip install huggingface_hub
huggingface-cli download TheBloke/Berghof-NSFW-7B-GGUF Berghof-NSFW-7B.i1-Q4_K_S.gguf --local-dir . --local-dir-use-symlinks False

# Method 2: 直接wget (リンクが有効な場合)
wget -O Berghof-NSFW-7B.i1-Q4_K_S.gguf "https://huggingface.co/TheBloke/Berghof-NSFW-7B-GGUF/resolve/main/Berghof-NSFW-7B.i1-Q4_K_S.gguf?download=true"

# Method 3: WindowsからWSL2にコピー
cp /mnt/c/Users/adama/Downloads/Berghof-NSFW-7B.i1-Q4_K_S.gguf .
```

**ファイルサイズ確認**:
- `Berghof-NSFW-7B.i1-Q4_K_S.gguf`: 3.9GB (4,140,374,464 bytes)

### 3. Wav2Lip (リップシンク)
- **基本モデル**: 自動ダウンロード（wav2lip/scripts/download_models.sh）
- **追加モデル**: checkpointsディレクトリに自動配置

#### Wav2Lip自動ダウンロード
```bash
cd wav2lip
chmod +x scripts/download_models.sh
./scripts/download_models.sh
```

**自動ダウンロードされるファイル**:
- `wav2lip_gan.pth`: 416MB
- `wav2lip.pth`: 416MB
- `detection_Resnet50_Final.pth`: 105MB
- `parsing_parsenet.pth`: 82MB
- `GFPGANv1.4.pth`: 333MB
- `yolo11n.pt`: 5.7MB

## 🔍 ダウンロード検証コマンド

### 全モデルファイル確認
```bash
# GPT-SoVITS確認
ls -lh Gptsovits/models/v4/GPT-SoVITS/gpt-sovits-ja-h/hscene-e17.ckpt

# LlamaCPP確認  
ls -lh llamacpp/models/Berghof-NSFW-7B.i1-Q4_K_S.gguf

# Wav2Lip確認
ls -lh wav2lip/checkpoints/wav2lip_gan.pth
ls -lh wav2lip/checkpoints/GFPGANv1.4.pth
```

### サイズ検証
```bash
# 期待されるファイルサイズ
echo "Expected file sizes:"
echo "hscene-e17.ckpt: 148MB"
echo "Berghof-NSFW-7B.i1-Q4_K_S.gguf: 3.9GB" 
echo "wav2lip_gan.pth: 416MB"
echo "GFPGANv1.4.pth: 333MB"

echo ""
echo "Actual file sizes:"
find . -name "hscene-e17.ckpt" -exec ls -lh {} \;
find . -name "Berghof-NSFW-7B.i1-Q4_K_S.gguf" -exec ls -lh {} \;
find . -name "wav2lip_gan.pth" -exec ls -lh {} \;
find . -name "GFPGANv1.4.pth" -exec ls -lh {} \;
```

## 📦 Docker ビルド（モデル配置後）

### 全プロジェクトビルド
```bash
# GPT-SoVITS
cd Gptsovits && DOCKER_BUILDKIT=1 docker build -t gpt-sovits:v4 .

# LlamaCPP  
cd ../llamacpp && DOCKER_BUILDKIT=1 docker build -t llama-cpp-python:cuda .

# Wav2Lip
cd ../wav2lip && DOCKER_BUILDKIT=1 docker build -f Dockerfile.optimized -t wav2lip-optimized:v1 .
```

### 実行テスト
```bash
# GPT-SoVITS音声生成テスト
cd Gptsovits
docker run --gpus all --rm --privileged \
  -v /usr/lib/wsl:/usr/lib/wsl -e LD_LIBRARY_PATH=/usr/lib/wsl/lib \
  -v $(pwd):/app gpt-sovits:v4 python /app/scripts/test_voice_clone_warmup.py

# LlamaCPPチャットテスト
cd ../llamacpp  
docker run --gpus all --rm -it --privileged \
  -v /usr/lib/wsl:/usr/lib/wsl -e LD_LIBRARY_PATH=/usr/lib/wsl/lib \
  -v $(pwd)/models:/app/models \
  llama-cpp-python:cuda python /app/scripts/chat_cli.py

# Wav2Lipリップシンクテスト
cd ../wav2lip
docker run --gpus all --rm --privileged \
  -v /usr/lib/wsl:/usr/lib/wsl -e LD_LIBRARY_PATH=/usr/lib/wsl/lib \
  -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output \
  -v $(pwd)/checkpoints:/app/checkpoints \
  wav2lip-optimized:v1 python /app/inference_fp16_yolo.py \
  --checkpoint_path /app/checkpoints/wav2lip_gan.pth \
  --face /app/input/target_video.mp4 \
  --audio /app/input/reference_audio.wav \
  --outfile /app/output/test_result.mp4 \
  --out_height 720 --quality Fast
```

## 🚨 トラブルシューティング

### モデルファイルが見つからない場合
1. パスが正しいか確認
2. ファイル権限を確認（chmod 755）
3. WSL2とWindows間のファイルコピーを確認

### ダウンロードに失敗する場合
1. インターネット接続を確認
2. Hugging Face認証が必要な場合は`huggingface-cli login`
3. 手動でブラウザからダウンロードしてWSL2にコピー

### GPU認識しない場合
1. `nvidia-smi`でGPU確認
2. `docker run --gpus all --rm nvidia/cuda:12.1-runtime-ubuntu20.04 nvidia-smi`でDocker GPU確認
3. NVIDIA Container Toolkitインストール確認

## ✅ 完全復元チェックリスト

- [ ] リポジトリクローン完了
- [ ] setup.sh実行完了
- [ ] GPT-SoVITS hscene-e17.ckpt (148MB) 配置完了
- [ ] LlamaCPP Berghof-NSFW-7B.i1-Q4_K_S.gguf (3.9GB) 配置完了
- [ ] Wav2Lip自動ダウンロード完了
- [ ] 全Dockerイメージビルド完了
- [ ] GPU認識テスト完了
- [ ] 各プロジェクト動作テスト完了

**✅ 全チェック完了後、完全にいつでも再開可能！**