# LlamaCPP FastAPI 使用ガイド

## 概要
事前ロード最適化による高速LLMチャットサーバーです。7つのキャラクタープリセットでインタラクティブな会話が可能です。

## 🚀 クイックスタート

### 1. FastAPIサーバーの起動
```bash
cd /home/adama/gpts/llamacpp

# 改良版FastAPIサーバーを起動（初回は7-10秒程度かかります）
docker run --gpus all --rm -it \
  --privileged \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/config:/app/config \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/src:/app/src \
  -v /usr/lib/wsl:/usr/lib/wsl \
  -e LD_LIBRARY_PATH=/usr/lib/wsl/lib \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e CUDA_VISIBLE_DEVICES=0 \
  -p 8001:8001 \
  llama-cpp-python:cuda python /app/src/fastapi_chat_server.py
```

### 2. サーバー状態確認
```bash
# 基本ヘルスチェック
curl http://localhost:8001/

# 詳細ヘルスチェック
curl http://localhost:8001/health | python3 -m json.tool

# 利用可能なプリセット
curl http://localhost:8001/presets
```

### 3. インタラクティブチャット開始

#### 方法1: インタラクティブCLI使用（推奨）
```bash
# 基本的な使用（ツンデレキャラクター）
python3 scripts/interactive_chat_cli.py

# 異なるキャラクターで開始
python3 scripts/interactive_chat_cli.py --character friendly

# 設定を指定して開始
python3 scripts/interactive_chat_cli.py --character technical --temperature 0.5 --max-tokens 256
```

#### 方法2: 一括テスト実行
```bash
# 全機能の自動テスト
./run_fastapi_tests.sh
```

#### 方法3: API直接呼び出し
```bash
# インタラクティブチャット
curl -X POST http://localhost:8001/interactive \
  -H "Content-Type: application/json" \
  -d '{
    "message": "こんにちは！",
    "character": "tsundere",
    "temperature": 0.7,
    "max_tokens": 512
  }'

# 基本チャット
curl -X POST http://localhost:8001/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "テストメッセージ",
    "use_history": true,
    "stream": false,
    "generation_config": {
      "max_tokens": 256,
      "temperature": 0.7
    }
  }'
```

## 🎭 キャラクタープリセット

### 利用可能なキャラクター
- **tsundere** 🎭 - ツンデレ女の子（「べ、別に〜」「〜なんだからね！」）
- **friendly** 😊 - フレンドリー（明るく親しみやすい）
- **technical** 🔧 - 技術的（プログラミング・技術特化）
- **casual** 😎 - カジュアル（友達感覚のタメ口）
- **polite** 🙏 - 丁寧（非常に礼儀正しい敬語）
- **creative** 🎨 - クリエイティブ（詩的で創造的）
- **academic** 📚 - 学術的（論理的で専門的）

### キャラクター変更
```bash
# インタラクティブCLI内で
/character friendly
/character technical
/character casual
```

## 🛠️ APIエンドポイント

### GET /health
詳細なヘルスチェック

**レスポンス例:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "gpu_available": true,
  "uptime": "0:05:32.123456",
  "preload_status": {
    "basic_deps": true,
    "llm_deps": true,
    "config": true,
    "presets": true,
    "cache_size": 7
  }
}
```

### POST /interactive
インタラクティブチャット（キャラクター対応）

**リクエスト:**
```json
{
  "message": "こんにちは！",
  "character": "tsundere",
  "temperature": 0.7,
  "max_tokens": 512
}
```

**レスポンス:**
```json
{
  "response": "ふん！こ、こんにちは...べ、別に挨拶したかったわけじゃないんだからね！",
  "character": "tsundere",
  "inference_time": 1.23,
  "tokens_per_second": 25.4
}
```

### POST /chat
基本チャット

**リクエスト:**
```json
{
  "message": "テストメッセージ",
  "use_history": true,
  "stream": false,
  "generation_config": {
    "max_tokens": 256,
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 40,
    "repeat_penalty": 1.1
  }
}
```

### GET /presets
利用可能なプリセット一覧

### DELETE /history
会話履歴クリア

### GET /status
詳細なサーバーステータス

## 🎮 インタラクティブCLI使用方法

### 基本コマンド
```bash
/help              # ヘルプ表示
/character <name>  # キャラクター変更
/temp <value>      # 温度変更 (0.1-2.0)
/tokens <value>    # 最大トークン数変更
/presets           # プリセット一覧
/status            # サーバーステータス
/clear             # 履歴クリア
/stats             # セッション統計
/quit              # 終了
```

### 使用例
```bash
# キャラクター変更
/character friendly

# 温度調整（創造性を高める）
/temp 0.9

# トークン数調整
/tokens 256

# 履歴クリア
/clear
```

## 📊 パフォーマンス特性

### 事前ロード最適化
- **初期化時間**: 7-10秒（依存関係事前ロード込み）
- **応答速度**: 1-2秒/リクエスト
- **スループット**: 30-60リクエスト/分
- **GPU使用量**: 3.86GB VRAM（RTX 3050）

### 最適化機能
- **基本依存関係**: JSON、YAML、datetime事前ロード
- **LLM依存関係**: llama-cpp-python、設定ファイル事前ロード
- **キャラクタープリセット**: 7種類のプリセット事前ロード
- **Warm-up推論**: 初期化時にテスト推論実行
- **GPU最適化**: TensorCore、CUDA最適化

## 🔧 サーバー管理

### ログ確認
```bash
# Dockerコンテナのログ
docker logs <container_id>

# アプリケーションログ
ls -la logs/llm_engine_*.log
```

### サーバー停止
```bash
# Ctrl+C でサーバー停止
# または
docker stop <container_id>
```

### 設定変更
```bash
# 設定ファイル編集
vim config/model_config.yaml

# サーバー再起動が必要
```

## 🚨 トラブルシューティング

### サーバーに接続できない
```bash
# ポート使用状況確認
sudo lsof -i :8001

# コンテナ状態確認
docker ps -a
```

### GPU認識エラー
```bash
# GPU状態確認
nvidia-smi
docker run --gpus all --rm nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

### 応答が遅い
- 初回起動時は7-10秒かかります
- 事前ロード完了後は1-2秒で応答
- `/status`で事前ロード状態を確認

### キャラクターが反映されない
- `/presets`で利用可能なプリセットを確認
- 正確なスペルでキャラクター名を指定
- `/character <name>`で変更

## 📈 パフォーマンス比較

| 機能 | 旧Gradio | 新FastAPI | 改善率 |
|------|---------|-----------|--------|
| 初期化時間 | 21秒 | **7-10秒** | **50%向上** |
| 応答速度 | 2.14-2.80秒 | **1-2秒** | **30%向上** |
| キャラクター数 | 7個 | **7個** | **同等** |
| API機能 | 限定的 | **充実** | **大幅向上** |
| 事前ロード | 基本的 | **高度** | **大幅向上** |

## 🎯 推奨使用パターン

### 日常会話
```bash
python3 scripts/interactive_chat_cli.py --character friendly
```

### 技術質問
```bash
python3 scripts/interactive_chat_cli.py --character technical --temperature 0.5
```

### 創造的作業
```bash
python3 scripts/interactive_chat_cli.py --character creative --temperature 0.9
```

### 学術的議論
```bash
python3 scripts/interactive_chat_cli.py --character academic --temperature 0.6
```

---

**バージョン**: 2.0.0 (FastAPI Enhanced)  
**作成日**: 2025-07-17  
**動作確認環境**: WSL2 Ubuntu 22.04 + RTX 3050 + CUDA 12.1  
**特記事項**: 事前ロード最適化、7キャラクター対応、GPU加速、インタラクティブCLI