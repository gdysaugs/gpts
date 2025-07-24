# SadTalker Custom API Setup

## Quick Start

### 1. Pre-download Models (避免每次テスト時のモデルdl)

```bash
# モデルを事前ダウンロード
python download_models.py
```

### 2. Build Docker Image with Pre-downloaded Models

```bash
# Dockerイメージをビルド（モデル込み）
docker build -t sadtalker-custom:latest .
```

### 3. Run with Custom API

```bash
# カスタムAPIで実行
docker-compose -f docker-compose-custom.yml up -d
```

## API Endpoints

### Text to Video
```bash
curl -X POST "http://localhost:10364/generate/text" \
     -H "Content-Type: application/json" \
     -d '{"text": "Hello, this is a test", "enhance": true}' \
     --output result.mp4
```

### Audio to Video
```bash
curl -X POST "http://localhost:10364/generate/audio" \
     -F "audio=@input.wav" \
     -F "enhance=true" \
     --output result.mp4
```

### Audio + Image to Video
```bash
curl -X POST "http://localhost:10364/generate/video" \
     -F "audio=@input.wav" \
     -F "image=@face.jpg" \
     -F "enhance=true" \
     --output result.mp4
```

### Status Check
```bash
curl "http://localhost:10364/status"
```

## Features

### ✅ Model Pre-downloading
- 全てのモデルがDockerビルド時にダウンロード
- テスト実行時のモデルダウンロード不要
- 初回テスト時間大幅短縮

### ✅ Custom FastAPI
- オリジナルのFastAPIより柔軟
- 複数のエンドポイント対応
- ファイルアップロード対応
- Base64レスポンス対応
- CORS対応
- ヘルスチェック対応

### ✅ Enhanced Endpoints
- `/generate/text` - テキストから動画生成
- `/generate/audio` - 音声ファイルから動画生成  
- `/generate/video` - 音声+画像から動画生成
- `/status` - APIステータス確認
- `/health` - ヘルスチェック

## Performance Improvements

1. **Model Pre-loading**: 全モデルが起動時にロード済み
2. **No Re-download**: テスト毎のモデルダウンロード不要
3. **Async Processing**: 非同期処理対応
4. **File Management**: 効率的なファイル管理
5. **Error Handling**: 詳細なエラーハンドリング

## Directory Structure

```
faster-SadTalker-API/
├── download_models.py          # モデルダウンロードスクリプト
├── custom_api.py              # カスタムFastAPI実装
├── docker-compose-custom.yml   # カスタム用Docker Compose
├── Dockerfile                 # 修正済みDockerfile (モデル事前DL)
├── checkpoints/               # SadTalkerモデル (自動ダウンロード)
├── gfpgan/weights/           # GFPGANモデル (自動ダウンロード)
├── results/                  # 出力動画
└── input/                    # 入力ファイル
```

## Migration from Original API

### Before (Original)
```bash
# 毎回モデルダウンロードが発生
docker run --gpus all -p 10364:10364 paidax/faster-sadtalker-api:0.1.3
```

### After (Custom)
```bash
# モデル事前ダウンロード済み、高速起動
docker-compose -f docker-compose-custom.yml up -d
```

## Troubleshooting

### Models Not Found
```bash
# モデルダウンロードを再実行
python download_models.py
```

### GPU Issues
```bash
# GPU確認
docker run --gpus all --rm nvidia/cuda:11.6-base-ubuntu20.04 nvidia-smi
```

### API Health Check
```bash
# ヘルスチェック
curl http://localhost:10364/health
curl http://localhost:10364/status
```