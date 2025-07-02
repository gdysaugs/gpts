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

### 推奨使用方法

#### ⭐ FastAPIサーバー（推奨・本格利用）
**特徴**: 初期化1回のみ、以降3秒/回の高速応答
```bash
cd Gptsovits

# 1. サーバー起動（初期化20秒、1回のみ）
docker run --gpus all -d -p 8000:8000 --privileged --name gpt-sovits-api \
  -v $(pwd)/input:/app/input \
  -v $(pwd)/output:/app/output \
  -v $(pwd)/scripts:/app/scripts \
  -v $(pwd)/models/v4/GPT-SoVITS/gpt-sovits-ja-h:/app/GPT_SoVITS/pretrained_models/gpt-sovits-ja-h \
  -v /usr/lib/wsl:/usr/lib/wsl \
  -e LD_LIBRARY_PATH=/usr/lib/wsl/lib \
  gpt-sovits:v4 bash -c "pip install fastapi uvicorn python-multipart && python /app/scripts/fastapi_voice_server.py"

# 2. API呼び出し（3秒で完了）
curl -G "http://localhost:8000/clone-voice-simple" \
  --data-urlencode "ref_text=おはようございます" \
  --data-urlencode "target_text=FastAPIで高速音声生成テストです" > output/fastapi_result.wav

# 3. サーバー停止
docker stop gpt-sovits-api && docker rm gpt-sovits-api
```

#### CLI版（開発・テスト用）

##### 標準v2モデル
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

##### 日本語専用モデル（感情表現）
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

##### Warm-up最適化版（連続処理用）
```bash
docker run --gpus all --rm \
  --privileged \
  -v $(pwd)/input:/app/input \
  -v $(pwd)/output:/app/output \
  -v $(pwd)/scripts:/app/scripts \
  -v $(pwd)/models/v4/GPT-SoVITS/gpt-sovits-ja-h:/app/GPT_SoVITS/pretrained_models/gpt-sovits-ja-h \
  -v /usr/lib/wsl:/usr/lib/wsl \
  -e LD_LIBRARY_PATH=/usr/lib/wsl/lib \
  gpt-sovits:v4 python /app/scripts/test_voice_clone_warmup.py \
  --ref-audio /app/input/reference_5sec.wav \
  --ref-text "おはようございます" \
  --target-text "Warm-up最適化版のテストです" \
  --output /app/output/warmup_result.wav
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

### 必要なソフトウェアとセットアップ

#### 1. Docker環境
- **Docker（WSL2内、Docker Desktopではない）**
- **NVIDIA Container Toolkit**（GPUアクセス必須）
- **Git LFS**（GPT-SoVITSモデル用）

#### 2. NVIDIA Container Toolkit インストール手順
```bash
# 1. GPGキーとリポジトリ追加
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# 2. インストール
sudo apt update
sudo apt install -y nvidia-container-toolkit

# 3. Docker再起動
sudo systemctl restart docker

# 4. 動作確認
docker run --gpus all --rm --privileged \
  -v /usr/lib/wsl:/usr/lib/wsl \
  -e LD_LIBRARY_PATH=/usr/lib/wsl/lib \
  gpt-sovits:v4 nvidia-smi
```

## 🔧 トラブルシューティング

### 🚨 今回の詰まったポイントと解決策

#### ❌ NVIDIA Container Toolkit未インストール
**症状**: `docker: Error response from daemon: could not select device driver with capabilities: [[gpu]]`
```bash
# 解決策: 上記インストール手順を実行
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
# 以下、上記手順通り実行
```

#### ❌ 不正なCUDAイメージ名
**症状**: `manifest for nvidia/cuda:12.1-runtime-ubuntu20.04 not found`
```bash
# ❌ 失敗例
docker run --gpus all --rm nvidia/cuda:12.1-runtime-ubuntu20.04 nvidia-smi

# ✅ 正解: ビルド済みイメージで確認
docker run --gpus all --rm --privileged \
  -v /usr/lib/wsl:/usr/lib/wsl \
  -e LD_LIBRARY_PATH=/usr/lib/wsl/lib \
  gpt-sovits:v4 nvidia-smi
```

#### ❌ WSL2 GPU アクセス設定不備
**症状**: `RuntimeError: Unexpected error from cudaGetDeviceCount()`
```bash
# 必須フラグ（全て必要）
--privileged                          # WSL2権限
-v /usr/lib/wsl:/usr/lib/wsl         # WSL2ライブラリマウント
-e LD_LIBRARY_PATH=/usr/lib/wsl/lib  # WSL2ライブラリパス
```

#### ❌ スクリプトファイル見つからない
**症状**: `can't open file '/app/scripts/test_voice_clone_ja_complete.py'`
```bash
# 解決策: scriptsディレクトリもマウントする
-v $(pwd)/scripts:/app/scripts  # この行を追加
```

### GPU認識問題（基本）
```bash
# GPU確認（RTX 3050の場合）
nvidia-smi  # ホストで確認

# Docker内でGPU確認
docker run --gpus all --rm --privileged \
  -v /usr/lib/wsl:/usr/lib/wsl \
  -e LD_LIBRARY_PATH=/usr/lib/wsl/lib \
  gpt-sovits:v4 nvidia-smi
```

### パフォーマンス問題
**症状**: 音声生成が遅い（20秒以上）
```bash
# ✅ 解決策: FastAPIサーバー使用（3秒/回）
# 上記「FastAPIサーバー（推奨・本格利用）」を参照
```

### モデルダウンロード失敗
```bash
# GPT-SoVITS
cd Gptsovits && ./scripts/download_models.sh

# LlamaCPP
cd llamacpp && ./scripts/setup_model.sh
```

## 📊 パフォーマンス（RTX 3050基準）

### GPT-SoVITS 実測値

| 方式 | 初期化時間 | 生成時間/回 | 総時間 | 使用場面 |
|------|------------|-------------|--------|----------|
| **FastAPI** | 20秒（1回のみ） | **3秒** | **3秒** | 🥇 **本格利用推奨** |
| Warm-up版 | 21秒（毎回） | 2.7秒 | 35秒 | 🥈 連続処理用 |
| 通常版 | 25秒（毎回） | 20秒 | 45秒 | 🥉 開発・テスト用 |

#### FastAPIサーバーの圧倒的優位性
- **初期化**: 20秒（起動時1回のみ）
- **レスポンス**: 3秒/リクエスト（7.5倍高速化）
- **並列処理**: 複数リクエスト同時処理可能
- **RESTful API**: プログラムから簡単呼び出し

#### 技術仕様
- **VRAM使用量**: 4-6GB（FP16で2-3GB可能）
- **音質**: RMS=25-35, 非無音率60-80%
- **対応モデル**: 標準v2、日本語専用（hscene-e17.ckpt）
- **最適化**: TensorCore + Torch.compile + FP16

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