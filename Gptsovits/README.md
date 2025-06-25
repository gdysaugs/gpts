# GPT-SoVITS 音声クローニングシステム

## プロジェクト概要
GPT-SoVITSを使用した日本語特化音声クローニングシステム。標準モデルと日本語特化モデル（hscene-e17.ckpt）両方に対応し、感情豊かで自然な音声生成が可能。WSL2環境でDockerを使用してGPU加速による高速生成を実現。

## システム要件
- WSL2 Ubuntu 22.04
- Docker（WSL2内で動作、**Docker Desktopではない**）
- NVIDIA RTX 3050 Laptop GPU with CUDA 12.1 (compatible with host CUDA 12.6)
- NVIDIA Container Toolkit（WSL2でのGPUアクセス用）
- Python 3.10（Docker内）
- 8GB+ VRAM推奨（RTX 3050で動作確認済み）

## ディレクトリ構造
```
/home/adama/.claude/projects/Gptsovits/
├── README.md                           # 本ファイル
├── CLAUDE.md                          # Claude Code用設定ファイル
├── Dockerfile                         # マルチステージビルド用Dockerfile
├── docker-compose.yml                # Docker Compose設定
├── scripts/                          # CLIテストスクリプト
│   ├── test_voice_clone.py           # 標準音声クローンスクリプト
│   ├── test_voice_clone_ja_complete.py # 日本語特化モデル対応スクリプト
│   └── download_models.sh            # モデルダウンロードスクリプト
├── models/v4/GPT-SoVITS/             # モデルファイル
│   ├── gsv-v2final-pretrained/      # 標準v2モデル
│   ├── gsv-v4-pretrained/           # v4モデル
│   └── gpt-sovits-ja-h/             # 日本語特化モデル
│       └── hscene-e17.ckpt          # 感情豊かな日本語モデル
├── input/                           # 入力音声ファイル
│   ├── reference_5sec.wav           # 参照音声（5秒程度推奨）
│   └── ohayougozaimasu_10.wav       # サンプル音声
├── output/                          # 出力音声ファイル
└── logs/                           # ログファイル
    └── test_voice_clone.log         # 実行ログ
```

## 主要機能

### 🎯 対応モデル
1. **標準v2モデル** (`gsv-v2final-pretrained`)
   - 汎用的な日本語音声生成
   - 安定した品質
   - 軽量で高速

2. **日本語特化モデル** (`hscene-e17.ckpt`)
   - 感情豊かな音声生成
   - より自然な日本語読み上げ
   - 650時間の日本語データで訓練済み

### 🎨 感情表現対応
- **平常**: 通常の読み上げ
- **怒り**: 「何ですって！絶対に許可できません！」
- **興奮**: 「わあああ！すごい！本当にすごいです！」
- **技術説明**: 英語混じりの専門用語も対応

### 📝 テキスト処理機能
- **長文対応**: 15秒を超える音声生成可能
- **英語混在**: "Machine Learning"、"Deep Learning"等の技術用語
- **句読点処理**: 自然な区切りで音声生成
- **全文生成**: セグメント分割問題を解決済み

## 🚀 クイックスタート

### 1. 事前準備
```bash
# プロジェクトディレクトリに移動
cd /home/adama/.claude/projects/Gptsovits

# 参照音声ファイルを配置（5秒程度推奨）
cp your_reference_audio.wav input/reference_5sec.wav
```

### 2. Dockerイメージビルド
```bash
# BuildKit使用でビルド（キャッシュ有効）
DOCKER_BUILDKIT=1 docker build -t gpt-sovits:v4 .
```

### 3. 🌟 **FastAPI 本格運用サーバー（推奨）**

#### 3.1 サーバー起動
```bash
# 高速音声生成サーバーを起動（初回初期化25秒、以降3秒/リクエスト）
docker run --gpus all -d -p 8000:8000 \
  --privileged --name gpt-sovits-api \
  -v $(pwd)/input:/app/input \
  -v $(pwd)/output:/app/output \
  -v $(pwd)/scripts:/app/scripts \
  -v $(pwd)/models/v4/GPT-SoVITS/gpt-sovits-ja-h:/app/GPT_SoVITS/pretrained_models/gpt-sovits-ja-h \
  -v /usr/lib/wsl:/usr/lib/wsl \
  -e LD_LIBRARY_PATH=/usr/lib/wsl/lib \
  -e IS_HALF=True \
  gpt-sovits:v4 bash -c "pip install fastapi uvicorn python-multipart && python /app/scripts/fastapi_voice_server.py"
```

#### 3.2 API使用例

**シンプル音声生成（固定参照音声）**:
```bash
# 基本的な音声生成（日本語は必ずURLエンコード）
curl "http://localhost:8000/clone-voice-simple?ref_text=おはよう&target_text=こんにちは、今日は良い天気ですね" -o result.wav

# 感情豊かな音声生成
curl "http://localhost:8000/clone-voice-simple?ref_text=%E3%81%8A%E3%81%AF%E3%82%88%E3%81%86&target_text=%E3%82%8F%E3%81%82%E3%81%82%E3%81%82%EF%BC%81%E3%81%99%E3%81%94%E3%81%84%EF%BC%81%E6%9C%AC%E5%BD%93%E3%81%AB%E7%B4%A0%E6%99%B4%E3%82%89%E3%81%97%E3%81%84%E7%B5%90%E6%9E%9C%E3%81%A7%E3%81%99%EF%BC%81" -o emotional_voice.wav

# 短いテキスト（自動で20文字以上に延長される）
curl "http://localhost:8000/clone-voice-simple?ref_text=%E3%81%8A%E3%81%AF%E3%82%88%E3%81%86&target_text=%E3%83%86%E3%82%B9%E3%83%88" -o test_voice.wav
```

**サーバー状態確認**:
```bash
# サーバーの状態とGPU情報を確認
curl "http://localhost:8000/" | python3 -m json.tool

# ログ確認
docker logs gpt-sovits-api --tail 10

# サーバー停止
docker stop gpt-sovits-api && docker rm gpt-sovits-api
```

#### 3.3 🎛️ **パラメータ詳細**

| パラメータ | 説明 | 例 |
|-----------|------|-----|
| `ref_text` | 参照音声のテキスト | `おはようございます` |
| `target_text` | 生成したいテキスト | `こんにちは` |
| `temperature` | 生成の創造性（0.5-2.0） | `1.0` |

**⚠️ 重要**: 日本語パラメータは必ずURLエンコードしてください
```bash
# URLエンコード例
echo "わあああ！" | python3 -c "import urllib.parse,sys; print(urllib.parse.quote(sys.stdin.read().strip()))"
# → %E3%82%8F%E3%81%82%E3%81%82%E3%81%82%EF%BC%81
```

### 4. 標準モデルで音声生成（従来方式）
```bash
docker run --gpus all --rm \
  --privileged \
  -v $(pwd)/input:/app/input \
  -v $(pwd)/output:/app/output \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/scripts:/app/scripts \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v /usr/lib/wsl:/usr/lib/wsl \
  -e LD_LIBRARY_PATH=/usr/lib/wsl/lib \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e CUDA_VISIBLE_DEVICES=0 \
  gpt-sovits:v4 python /app/scripts/test_voice_clone.py \
  --ref-audio /app/input/reference_5sec.wav \
  --ref-text "おはようございます" \
  --target-text "こんにちは、これはテスト音声です" \
  --output /app/output/result.wav
```

### 4. 日本語特化モデルで感情豊かな音声生成
```bash
# 日本語特化モデルhscene-e17.ckptを使用
docker run --gpus all --rm \
  --privileged \
  -v $(pwd)/input:/app/input \
  -v $(pwd)/output:/app/output \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/scripts:/app/scripts \
  -v $(pwd)/models/v4/GPT-SoVITS/gpt-sovits-ja-h:/app/GPT_SoVITS/pretrained_models/gpt-sovits-ja-h \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v /usr/lib/wsl:/usr/lib/wsl \
  -e LD_LIBRARY_PATH=/usr/lib/wsl/lib \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e CUDA_VISIBLE_DEVICES=0 \
  gpt-sovits:v4 python /app/scripts/test_voice_clone_ja_complete.py \
  --ref-audio /app/input/reference_5sec.wav \
  --ref-text "おはようございます" \
  --target-text "わあああ！すごい！本当に素晴らしい結果です！" \
  --sovits-model "/app/GPT_SoVITS/pretrained_models/gpt-sovits-ja-h/hscene-e17.ckpt" \
  --output /app/output/emotional_result.wav
```

## 🎭 実用サンプル

### 技術説明（英語混在）
```bash
docker run --gpus all --rm --privileged \
  -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output \
  -v $(pwd)/logs:/app/logs -v $(pwd)/scripts:/app/scripts \
  -v $(pwd)/models/v4/GPT-SoVITS/gpt-sovits-ja-h:/app/GPT_SoVITS/pretrained_models/gpt-sovits-ja-h \
  -v /tmp/.X11-unix:/tmp/.X11-unix -v /usr/lib/wsl:/usr/lib/wsl \
  -e LD_LIBRARY_PATH=/usr/lib/wsl/lib -e NVIDIA_VISIBLE_DEVICES=all -e CUDA_VISIBLE_DEVICES=0 \
  gpt-sovits:v4 python /app/scripts/test_voice_clone_ja_complete.py \
  --ref-audio /app/input/reference_5sec.wav --ref-text "おはようございます" \
  --target-text "今日のAI技術はMachine LearningやDeep Learningの進歩により、Natural Language ProcessingとComputer Visionの分野で革命的な変化をもたらしています。" \
  --sovits-model "/app/GPT_SoVITS/pretrained_models/gpt-sovits-ja-h/hscene-e17.ckpt" \
  --output /app/output/technical_explanation.wav
```

### 怒りの表現
```bash
docker run --gpus all --rm --privileged \
  -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output \
  -v $(pwd)/logs:/app/logs -v $(pwd)/scripts:/app/scripts \
  -v $(pwd)/models/v4/GPT-SoVITS/gpt-sovits-ja-h:/app/GPT_SoVITS/pretrained_models/gpt-sovits-ja-h \
  -v /tmp/.X11-unix:/tmp/.X11-unix -v /usr/lib/wsl:/usr/lib/wsl \
  -e LD_LIBRARY_PATH=/usr/lib/wsl/lib -e NVIDIA_VISIBLE_DEVICES=all -e CUDA_VISIBLE_DEVICES=0 \
  gpt-sovits:v4 python /app/scripts/test_voice_clone_ja_complete.py \
  --ref-audio /app/input/reference_5sec.wav --ref-text "おはようございます" \
  --target-text "何ですって！そんなことは絶対に許可できません！あなたは一体何を考えているのですか！" \
  --sovits-model "/app/GPT_SoVITS/pretrained_models/gpt-sovits-ja-h/hscene-e17.ckpt" \
  --output /app/output/angry_voice.wav
```

## 🛠️ トラブルシューティング

### GPU認識確認
```bash
# Docker内でGPUが認識されているか確認
docker run --gpus all --rm gpt-sovits:v4 nvidia-smi

# CUDAバージョン確認
docker run --gpus all --rm gpt-sovits:v4 nvcc --version
```

### ログ確認
```bash
# 実行ログを確認
tail -f logs/test_voice_clone.log

# GPUメモリ使用量確認
watch -n 1 nvidia-smi
```

### よくある問題

#### 🌟 **FastAPI関連**
1. **音声が短い・前半スキップ**: 20文字未満のテキストは自動延長される
2. **日本語でエラー**: 必ずURLエンコードを使用（上記例参照）
3. **サーバー応答なし**: 初期化中（25秒）の可能性、ログで確認
4. **ファイルが保存されない**: `/app/output`ボリュームマウント確認

#### 🔧 **従来スクリプト関連**
5. **音声が短い**: `ref_free=True`が設定されているか確認
6. **GPUエラー**: WSL2用フラグ`-v /usr/lib/wsl:/usr/lib/wsl`が必須
7. **モデルが読み込めない**: ファイルパスとマウントを確認
8. **日本語特化モデルが使われない**: `test_voice_clone_ja_complete.py`を使用

## 📋 スクリプト詳細

### 🌟 **`fastapi_voice_server.py` (推奨)**
- **本格運用サーバー**: RESTful API提供
- **高速生成**: 3秒/リクエスト（9倍高速）
- **自動最適化**: FP16 + Torch.compile + GPU加速
- **短文対応**: 20文字未満は自動延長
- **永続保存**: `output/`ディレクトリに自動保存
- **日本語特化**: hscene-e17.ckptモデル使用

### 🔧 **`test_voice_clone.py` (標準モデル)**
- 標準v2モデルを使用
- 安定した音声生成
- 汎用的な日本語読み上げ

### 🎭 **`test_voice_clone_ja_complete.py` (日本語特化)**
- モンキーパッチでカスタムモデル対応
- `hscene-e17.ckpt`モデルを使用
- 感情豊かな音声生成
- `--sovits-model`オプションでモデル指定

### パラメータ説明

#### FastAPI パラメータ
- `ref_text`: 参照音声のテキスト（URLエンコード必須）
- `target_text`: 生成したいテキスト（URLエンコード必須）
- `temperature`: 生成の創造性（0.5-2.0、デフォルト1.0）

#### 従来スクリプト パラメータ
- `--ref-audio`: 参照音声ファイル（5秒程度推奨）
- `--ref-text`: 参照音声のテキスト
- `--target-text`: 生成したいテキスト
- `--output`: 出力ファイルパス
- `--sovits-model`: カスタムモデルパス（日本語特化版のみ）

## 📊 性能指標

### 🚀 **FastAPI サーバー（推奨）**
- **初期化時間**: 25秒（1回のみ）
- **生成速度**: **3秒/リクエスト**（**従来の9倍高速**）
- **スループット**: 20リクエスト/分
- **自動最適化**: FP16 + Torch.compile + GPU加速
- **短文対応**: 20文字未満は自動延長で前半スキップ防止

### 🐌 従来スクリプト（参考）
- **初期化時間**: 25秒（毎回）
- **生成速度**: 27秒/リクエスト
- **スループット**: 2リクエスト/分

### 音声品質指標
- **RMS値**: 25-35 (良好な音圧)
- **非無音率**: 80%以上 (高品質)
- **サンプリングレート**: 32kHz
- **ファイル形式**: WAVE PCM 16bit mono

### リソース使用量
- **VRAM**: 4-6GB (RTX 3050で動作確認済み)
- **RAM**: 8GB推奨
- **ストレージ**: 5GB程度（モデルファイル含む）
- **CPU**: FP16有効で30-40%削減

## 📝 重要な注意事項

### WSL2 GPUアクセス
- **必須フラグ**: `--privileged`, `-v /usr/lib/wsl:/usr/lib/wsl`, `-e LD_LIBRARY_PATH=/usr/lib/wsl/lib`
- **非Docker Desktop**: WSL2内のネイティブDockerを使用
- **CUDA互換性**: ホストCUDA 12.6、コンテナ12.1で動作確認済み

### モデルファイル
- **hscene-e17.ckpt**: 手動で`models/v4/GPT-SoVITS/gpt-sovits-ja-h/`に配置
- **マウント**: カスタムモデル使用時はボリュームマウント必須

### 文字化け問題解決済み
- **前半の欠落**: `ref_free=True`で解決
- **セグメント分割**: `how_to_cut="不切"`で全文生成
- **モンキーパッチ**: カスタムモデル対応完全解決

## 🔗 参考情報

### 公式リポジトリ
- [GPT-SoVITS GitHub](https://github.com/RVC-Boss/GPT-SoVITS)
- [DeepWiki GPT-SoVITS](https://deepwiki.com/RVC-Boss/GPT-SoVITS)

### 日本語特化モデル
- [AkitoP/GPT-SoVITS-JA-H](https://huggingface.co/AkitoP/GPT-SoVITS-JA-H)
- 650時間の日本語データで訓練
- Apache 2.0ライセンス

### 技術情報
- NVIDIA Container Toolkitドキュメント
- WSL2 GPUサポートガイド
- Docker BuildKitドキュメント

---

**作成日**: 2025-06-23  
**バージョン**: v1.0  
**動作確認環境**: WSL2 Ubuntu 22.04 + RTX 3050 + CUDA 12.1  
**特記事項**: 日本語特化モデル対応、感情表現対応、長文生成対応

