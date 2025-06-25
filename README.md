# GPT-SoVITS & LlamaCPP Projects

日本語音声クローニングシステム(GPT-SoVITS)とローカルLLMチャットシステム(LlamaCPP)のDocker統合環境

## 🚀 Quick Start

### 1. リポジトリをクローン
```bash
git clone https://github.com/gdysaugs/gpts.git
cd gpts
```

### 2. 自動セットアップ実行
```bash
chmod +x setup.sh
./setup.sh
```

このスクリプトが以下を自動実行します：
- 環境自動検出（WSL2 vs ネイティブLinux）
- 必要なディレクトリ構造作成
- GPT-SoVITSモデルダウンロード
- LlamaCPPモデルダウンロード
- Docker環境確認
- 実行権限設定
- 環境別コマンド例の生成（docker_commands.txt）

### 3. Dockerイメージビルド

#### GPT-SoVITS
```bash
cd Gptsovits
DOCKER_BUILDKIT=1 docker build -t gpt-sovits:v4 .
```

#### LlamaCPP
```bash
cd llamacpp
DOCKER_BUILDKIT=1 docker build -t llama-cpp-python:cuda .
```

## 📁 プロジェクト構成

```
gpts/
├── setup.sh                    # 🔧 メインセットアップスクリプト
├── CLAUDE_HOME.md               # 📝 ホーム設定ファイル
├── Gptsovits/                   # 🎵 音声クローニングプロジェクト
│   ├── CLAUDE.md               # GPT-SoVITS設定
│   ├── Dockerfile              # Docker環境
│   ├── docker-compose.yml      # Compose設定
│   ├── scripts/                # スクリプト群
│   │   ├── download_models.sh  # モデルダウンロード
│   │   ├── test_voice_clone.py # 標準v2モデル
│   │   └── test_voice_clone_ja_complete.py # 日本語モデル
│   ├── models/                 # モデル配置 (gitignore済)
│   ├── input/                  # 参照音声ファイル
│   ├── output/                 # 生成音声
│   └── logs/                   # 実行ログ
└── llamacpp/                   # 🤖 LLMチャットプロジェクト
    ├── CLAUDE.md               # LlamaCPP設定
    ├── Dockerfile              # Docker環境
    ├── docker-compose.yml      # Compose設定
    ├── src/                    # コアモジュール
    │   ├── api_server.py       # FastAPI RESTサーバー
    │   └── llm_engine.py       # LLM推論エンジン
    ├── scripts/                # CLIアプリケーション
    │   ├── setup_model.sh      # モデルセットアップ
    │   ├── chat_cli.py         # 直接CLI会話
    │   ├── api_chat_cli.py     # API経由CLI会話
    │   └── test_api.py         # API総合テスト
    ├── models/                 # LLMモデル配置 (gitignore済)
    ├── config/                 # 設定ファイル
    └── logs/                   # 実行ログ
```

## 🎵 GPT-SoVITS 音声クローニング

### 特徴
- **日本語特化**: 感情表現豊かな音声生成
- **GPU加速**: WSL2 + Docker環境でRTX 3050最適化
- **2つのモデル**: 標準v2モデルと日本語専用モデル

### 基本的な使用方法

#### 標準v2モデル
```bash
cd Gptsovits
docker run --gpus all --rm \
  --privileged \
  -v $(pwd)/input:/app/input \
  -v $(pwd)/output:/app/output \
  -v /usr/lib/wsl:/usr/lib/wsl \
  -e LD_LIBRARY_PATH=/usr/lib/wsl/lib \
  gpt-sovits:v4 python /app/scripts/test_voice_clone.py \
  --ref-audio /app/input/reference_5sec.wav \
  --ref-text "おはようございます" \
  --target-text "こんにちは、これはテスト音声です" \
  --output /app/output/cloned_voice.wav
```

#### 日本語専用モデル（感情表現）
```bash
docker run --gpus all --rm \
  --privileged \
  -v $(pwd)/input:/app/input \
  -v $(pwd)/output:/app/output \
  -v $(pwd)/models/v4/GPT-SoVITS/gpt-sovits-ja-h:/app/GPT_SoVITS/pretrained_models/gpt-sovits-ja-h \
  -v /usr/lib/wsl:/usr/lib/wsl \
  -e LD_LIBRARY_PATH=/usr/lib/wsl/lib \
  gpt-sovits:v4 python /app/scripts/test_voice_clone_ja_complete.py \
  --ref-audio /app/input/reference_5sec.wav \
  --ref-text "おはようございます" \
  --target-text "わあああ！すごい！本当に素晴らしい結果です！" \
  --sovits-model "/app/GPT_SoVITS/pretrained_models/gpt-sovits-ja-h/hscene-e17.ckpt" \
  --output /app/output/emotional_result.wav
```

## 🤖 LlamaCPP ローカルLLM

### 特徴
- **高速推論**: FP16最適化で1.03秒/回答
- **デュアルインターフェース**: 直接CLI・FastAPI RESTサーバー
- **GPU最適化**: CUDA 12.1 + 量子化モデル(Q4_K_S)で4GB VRAM効率

### 基本的な使用方法

#### 直接CLI会話（最高速）
```bash
cd llamacpp
docker run --gpus all --rm -it \
  --privileged \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/config:/app/config \
  -v $(pwd)/logs:/app/logs \
  -v /usr/lib/wsl:/usr/lib/wsl \
  -e LD_LIBRARY_PATH=/usr/lib/wsl/lib \
  llama-cpp-python:cuda python /app/scripts/chat_cli.py
```

#### FastAPI RESTサーバー起動
```bash
docker run --gpus all --rm -it \
  --privileged \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/config:/app/config \
  -v $(pwd)/logs:/app/logs \
  -v /usr/lib/wsl:/usr/lib/wsl \
  -e LD_LIBRARY_PATH=/usr/lib/wsl/lib \
  -p 8000:8000 \
  llama-cpp-python:cuda python /app/src/api_server.py
```

API ドキュメント: http://localhost:8000/docs

#### API経由CLI会話
```bash
docker run --gpus all --rm -it \
  --network host \
  llama-cpp-python:cuda python /app/scripts/api_chat_cli.py
```

## ⚙️ システム要件

### 必須環境
- **OS**: Linux (Ubuntu 22.04推奨、WSL2対応)
- **GPU**: NVIDIA GPU + CUDA 12.1以上対応
- **VRAM**: 4GB以上（8GB推奨）
- **RAM**: 16GB以上推奨
- **Storage**: 50GB以上の空き容量

### 必要なソフトウェア
- Docker（WSL2内、**Docker Desktopではない**）
- NVIDIA Container Toolkit
- Git LFS（GPT-SoVITSモデル用）

## 🔧 トラブルシューティング

### GPU認識問題
```bash
# GPU確認
docker run --gpus all --rm nvidia/cuda:12.1-runtime-ubuntu20.04 nvidia-smi

# NVIDIA Container Toolkit再インストール
sudo apt update && sudo apt install nvidia-container-toolkit
sudo systemctl restart docker
```

### WSL2 GPU アクセス問題
必須フラグを確認：
- `--privileged`
- `-v /usr/lib/wsl:/usr/lib/wsl`  
- `-e LD_LIBRARY_PATH=/usr/lib/wsl/lib`

### モデルダウンロード失敗
```bash
# GPT-SoVITS
cd Gptsovits && ./scripts/download_models.sh

# LlamaCPP
cd llamacpp && ./scripts/setup_model.sh
```

## 📊 パフォーマンス

### GPT-SoVITS
- **生成速度**: 5秒音声→6秒生成（RTX 3050）
- **VRAM使用量**: 4-6GB（FP16で2-3GB可能）
- **音質**: RMS=25-35, 非無音率80%+

### LlamaCPP
- **応答速度**: 1.03秒/質問（直接CLI）
- **API応答**: ~2秒/質問（FastAPI経由）
- **VRAM使用量**: 3.86GB（96.1%使用率）
- **トークン生成**: 20-30 tokens/sec

## 🤝 コントリビューション

1. このリポジトリをフォーク
2. 機能ブランチを作成 (`git checkout -b feature/amazing-feature`)
3. 変更をコミット (`git commit -m 'Add amazing feature'`)
4. ブランチにプッシュ (`git push origin feature/amazing-feature`)
5. プルリクエストを作成

## 📄 ライセンス

このプロジェクトはMITライセンスの下で公開されています。

## 🆘 サポート

問題が発生した場合：
1. 各プロジェクトの`CLAUDE.md`で詳細な設定を確認
2. ログファイル(`logs/`)を確認
3. GPU状況を`nvidia-smi`で確認
4. Issuesで報告

---

**重要**: dockerビルドは必ずキャッシュを使用し（`--no-cache`禁止）、GPU推論は必須です！