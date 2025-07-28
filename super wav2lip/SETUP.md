# Super Wav2Lip セットアップ完全ガイド

## 🎯 WSL2再構築時の完全復旧手順

**このガイドは、WSL2を削除・再インストールした後に、Super Wav2Lipシステムを完全に復旧するための詳細手順書です。**

## 📋 前提条件確認

### **Step 1: WSL2 + Ubuntu 22.04インストール**
```bash
# Windows PowerShell（管理者権限）で実行
wsl --install -d Ubuntu-22.04

# WSL2が既にある場合の削除・再インストール
wsl --unregister Ubuntu-22.04
wsl --install -d Ubuntu-22.04
```

### **Step 2: Ubuntu初期設定**
```bash
# システム更新
sudo apt update && sudo apt upgrade -y

# 基本ツールインストール
sudo apt install -y curl wget git build-essential
```

## 🐳 Docker + NVIDIA Container Toolkit インストール

### **Step 3: Docker インストール**
```bash
# 公式Dockerリポジトリ追加
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

echo "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Docker インストール
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

# ユーザーをdockerグループに追加
sudo usermod -aG docker $USER

# Dockerサービス開始
sudo systemctl enable docker
sudo systemctl start docker
```

### **Step 4: NVIDIA Container Toolkit インストール（重要）**
```bash
# NVIDIA Container Toolkit リポジトリ追加
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# インストール
sudo apt update
sudo apt install -y nvidia-container-toolkit nvidia-docker2

# Docker再起動
sudo systemctl restart docker

# GPU動作確認
docker run --rm --gpus all nvidia/cuda:11.0.3-base-ubuntu20.04 nvidia-smi
```

### **Step 5: Docker Compose インストール**
```bash
# Docker Compose v2（プラグイン形式）は自動インストール済み
# 確認
docker compose version
```

## 📁 プロジェクトセットアップ

### **Step 6: プロジェクトディレクトリ作成**
```bash
# プロジェクトディレクトリ作成
mkdir -p /home/adama/project/gpts
cd /home/adama/project/gpts

# Super Wav2Lipプロジェクトをクローン（またはコピー）
# 既存のプロジェクトがある場合
cp -r "/mnt/c/path/to/backup/super wav2lip" ./

# プロジェクトディレクトリに移動
cd "super wav2lip"
```

### **Step 7: 必要ファイルの配置確認**
```bash
# 重要なファイルが存在するか確認
ls -la docker-compose.yml
ls -la scripts/setup.sh
ls -la gradio_wav2lip_ui.py

# setup.shに実行権限付与
chmod +x scripts/setup.sh
```

## 🚀 自動セットアップ実行

### **Step 8: 完全自動セットアップ**
```bash
# プロジェクトディレクトリで実行
cd "/home/adama/project/gpts/super wav2lip"

# セットアップスクリプト実行
bash scripts/setup.sh
```

**setup.shが実行すること:**
- ✅ ディレクトリ構造作成
- ✅ 入力ファイル（動画・音声）コピー
- ✅ ONNXモデルファイル配置
- ✅ GPU環境確認
- ✅ Docker環境確認

### **Step 9: Dockerコンテナビルド・起動**
```bash
# 統合システム起動
docker-compose up -d

# 起動ログ確認
docker-compose logs --follow
```

**起動確認ポイント:**
- GPT-SoVITS: 20秒で初期化完了
- Wav2Lip: 30秒で初期化完了
- Gradio UI: 10秒で起動完了

## 🔍 動作確認手順

### **Step 10: API動作確認**
```bash
# GPT-SoVITS API確認
curl http://localhost:8000/

# Wav2Lip API確認
curl http://localhost:8002/health

# Gradio UI確認
curl http://localhost:7860/
```

### **Step 11: 統合テスト実行**
```bash
# ブラウザでWebUI確認
# http://localhost:7860 にアクセス

# または CLI API テスト
curl -X POST "http://localhost:8002/generate-lipsync" \
  -F "video_file=@input/videos/source_video.mp4" \
  -F "audio_file=@input/audio/target_audio.wav" \
  -F "enhancer=gfpgan" \
  -F "batch_size=8" \
  --max-time 120 -o output/setup_test.mp4
```

## ⚠️ トラブルシューティング（よくある問題）

### **問題1: Docker GPU認識エラー**
```bash
# 症状: docker: Error response from daemon: could not select device driver
# 解決策1: NVIDIA Container Toolkit再インストール
sudo apt remove nvidia-container-toolkit nvidia-docker2
sudo apt install -y nvidia-container-toolkit nvidia-docker2
sudo systemctl restart docker

# 解決策2: nvidia-smiコマンド確認
nvidia-smi
# ホストでNVIDIA GPUが認識されているか確認
```

### **問題2: Permission Denied エラー**
```bash
# 症状: docker: permission denied while trying to connect
# 解決策: WSL再起動後にDockerグループ設定反映
# WSLを一度閉じて、Windows PowerShellで実行:
wsl --shutdown
# 再度WSLを開いて確認
groups | grep docker
```

### **問題3: ポート競合エラー**
```bash
# 症状: Port 7860 is already in use
# 解決策: ポート使用確認・停止
sudo lsof -i :7860
sudo lsof -i :8000
sudo lsof -i :8002

# 使用中のコンテナ停止
docker-compose down
```

### **問題4: モデルファイル不足**
```bash
# 症状: FileNotFoundError: model file not found
# 解決策: setup.sh再実行
bash scripts/setup.sh

# 手動でモデルファイル確認
ls -la models/onnx/
ls -la input/videos/
ls -la input/audio/
```

### **問題5: NLTK Data Missing**
```bash
# 症状: averaged_perceptron_tagger_eng not found
# 解決策: NLTKデータダウンロード
docker exec gpt-sovits-api python -c "
import nltk
nltk.download('averaged_perceptron_tagger_eng')
"
```

### **問題6: Web UI起動しない**
```bash
# 症状: localhost:7860 接続拒否
# 解決策1: コンテナログ確認
docker logs super-wav2lip-ui --tail 20

# 解決策2: UI コンテナ再起動
docker restart super-wav2lip-ui

# 解決策3: 全システム再起動
docker-compose restart
```

### **問題7: GPU メモリ不足**
```bash
# 症状: CUDA out of memory
# 解決策: batch_size調整
# gradio UIで batch_size を 8 → 4 → 2 に下げる

# または docker-compose.yml の環境変数で調整
environment:
  - BATCH_SIZE=4
```

## 🔧 高度なトラブルシューティング

### **完全リセット手順**
システムが完全に壊れた場合の復旧:
```bash
# 1. 全コンテナ・イメージ削除
docker-compose down
docker system prune -a --volumes

# 2. プロジェクト再配置
cd /home/adama/project/gpts
rm -rf "super wav2lip"
# バックアップからコピー
cp -r "/mnt/c/backup/super wav2lip" ./

# 3. セットアップ再実行
cd "super wav2lip"
bash scripts/setup.sh
docker-compose up -d --build
```

### **パフォーマンス最適化**
RTX 3050での最適設定:
```bash
# 1. GPU メモリ最適化
nvidia-smi  # VRAM使用量確認

# 2. batch_size最適化
# RTX 3050 (4GB): batch_size=8
# RTX 3060 (8GB): batch_size=16  
# RTX 3070 (8GB): batch_size=16

# 3. Docker メモリ制限調整
# docker-compose.yml で:
mem_limit: 8g  # システムRAMの50-75%
```

## 📝 チェックリスト

### **セットアップ完了確認**
- [ ] WSL2 Ubuntu 22.04インストール完了
- [ ] Docker + NVIDIA Container Toolkit インストール完了
- [ ] GPU認識確認（`nvidia-smi`成功）
- [ ] プロジェクトファイル配置完了
- [ ] `setup.sh` 実行完了
- [ ] `docker-compose up -d` 成功
- [ ] http://localhost:7860 アクセス成功
- [ ] テスト動画生成成功（30秒以内）

### **動作確認項目**
- [ ] GPT-SoVITS API (Port 8000) 動作
- [ ] Wav2Lip API (Port 8002) 動作  
- [ ] Gradio UI (Port 7860) 動作
- [ ] ファイルアップロード機能
- [ ] 音声生成（3秒以内）
- [ ] 口パク動画生成（27秒以内、GFPGAN込み）
- [ ] 動画ダウンロード機能

## 🎯 成功時の期待結果

**正常動作時の処理時間:**
- 音声生成: 3秒
- 口パク生成: 27秒（GFPGAN強化）
- 総処理時間: 30秒

**システム状態:**
- CPU使用率: 10-30%
- GPU使用率: 80-100%（処理中）
- VRAM使用量: 1.7GB/4GB
- Docker コンテナ: 3個すべて健全

**ファイル出力:**
- MP4形式、1-3MB程度
- H.264+AAC エンコーディング
- ブラウザ再生対応

---

## 🚀 まとめ

この手順に従うことで、WSL2を完全に削除・再構築した後でも、Super Wav2Lipシステムを確実に復旧できます。

**重要なポイント:**
1. **NVIDIA Container Toolkit**: GPU使用の必須要件
2. **setup.sh**: 自動セットアップの活用
3. **docker-compose**: ワンコマンド起動
4. **トラブルシューティング**: 7つの主要問題解決法

**困った時の最終手段:**
```bash
# 完全リセット + 再構築
docker system prune -a --volumes
bash scripts/setup.sh
docker-compose up -d --build
```

これで**いつでも30秒で口パク動画生成**できるシステムが復活します！🎭✨