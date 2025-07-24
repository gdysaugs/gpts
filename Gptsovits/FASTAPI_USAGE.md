# GPT-SoVITS FastAPI 使用ガイド

## 概要
GradioではなくFastAPIサーバーを使用した音声生成システムです。`reference_5sec.wav`の声質で任意のテキストを読み上げます。

## クイックスタート

### 1. FastAPIサーバーの起動
```bash
cd /home/adama/gpts/Gptsovits

# Dockerコンテナでサーバー起動（初回は20秒程度かかります）
docker run --gpus all -d -p 8000:8000 --name gpt-sovits-api \
  --privileged -v /usr/lib/wsl:/usr/lib/wsl -e LD_LIBRARY_PATH=/usr/lib/wsl/lib \
  -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output \
  -v $(pwd)/scripts:/app/scripts \
  -v $(pwd)/models/v4/GPT-SoVITS/gpt-sovits-ja-h:/app/GPT_SoVITS/pretrained_models/gpt-sovits-ja-h \
  gpt-sovits:v4 bash -c "pip install fastapi uvicorn python-multipart && python /app/scripts/fastapi_voice_server.py"
```

### 2. サーバー状態確認
```bash
# ヘルスチェック
curl http://localhost:8000/

# ログ確認
docker logs gpt-sovits-api --tail 20
```

### 3. 音声生成テスト

#### 方法1: テストCLI使用（推奨）
```bash
# 基本的な使用
python scripts/test_fastapi_cli.py "こんにちは、音声生成のテストです"

# 長文テスト
python scripts/test_fastapi_cli.py "今日のAI技術は素晴らしい進歩を遂げています。"

# 感情表現
python scripts/test_fastapi_cli.py "わあああ！すごい！本当に素晴らしい！"

# ファイル名指定
python scripts/test_fastapi_cli.py "テスト" --filename my_voice.wav
```

#### 方法2: 一括テスト実行
```bash
# 複数のテストを自動実行
./run_fastapi_tests.sh
```

#### 方法3: curl直接実行
```bash
# シンプルなGETリクエスト
curl -G "http://localhost:8000/clone-voice-simple" \
  --data-urlencode "ref_text=おはようございます" \
  --data-urlencode "target_text=音声生成のテストです" \
  > output/test.wav
```

## APIエンドポイント

### GET /clone-voice-simple
固定の参照音声（reference_5sec.wav）を使用した音声生成

**パラメータ:**
- `ref_text`: 参照音声のテキスト（デフォルト: "おはようございます"）
- `target_text`: 生成したいテキスト（必須）
- `temperature`: 生成の創造性（デフォルト: 1.0）

**レスポンス:** WAVファイル

### GET /
ヘルスチェックエンドポイント

**レスポンス例:**
```json
{
  "message": "GPT-SoVITS Voice Cloning API",
  "models_loaded": true,
  "gpu_available": true,
  "gpu_name": "NVIDIA GeForce RTX 3050 Laptop GPU"
}
```

## パフォーマンス特性

- **初期化時間**: 約20秒（初回のみ）
- **生成速度**: 2-3秒/リクエスト
- **スループット**: 20-30リクエスト/分
- **GPU使用量**: 4-6GB VRAM

## サーバー管理

### ログ確認
```bash
docker logs gpt-sovits-api --tail 50 -f
```

### サーバー停止
```bash
docker stop gpt-sovits-api && docker rm gpt-sovits-api
```

### サーバー再起動
```bash
docker restart gpt-sovits-api
```

## トラブルシューティング

### サーバーに接続できない
```bash
# ポート使用状況確認
sudo lsof -i :8000

# コンテナ状態確認
docker ps -a | grep gpt-sovits-api
```

### GPU認識エラー
```bash
# GPU状態確認
nvidia-smi
docker run --gpus all --rm nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

### 音声が短い・切れる
- 20文字未満のテキストは自動的に延長されます
- 長文の場合は句読点で適切に区切ってください

## 出力ファイル

生成された音声は以下の場所に保存されます：
- CLIテスト: `output/cli_test_YYYYMMDD_HHMMSS_[テキスト].wav`
- サーバー内部: `output/fastapi_YYYYMMDD_HHMMSS_[テキスト].wav`

## 注意事項

- 参照音声は`input/reference_5sec.wav`を使用します
- 日本語特化モデル（hscene-e17.ckpt）が自動的に使用されます
- GPU（CUDA）が必須です
- WSL2環境での実行を前提としています