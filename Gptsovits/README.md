# GPT-SoVITS 日本語特化音声クローニングシステム

## プロジェクト概要
**AkitoP/GPT-SoVITS-JA-H + FastAPI + 事前ロード最適化**による次世代音声クローニングシステム。日本語特化モデル（650時間学習済み）とモデル常駐化により、初回から**3-5秒**の超高速音声生成を実現。REST APIで簡単に音声生成が可能。感情豊かで自然な日本語音声クローニングが可能。

### 🚀 **最新実装機能（2025年7月版）**
- ✅ **AkitoP/GPT-SoVITS-JA-H**: 650時間日本語データ学習済み特化モデル
- ✅ **FastAPI REST API**: 高速かつ軽量なAPIサーバー
- ✅ **事前ロード最適化**: 言語検出モデル・Open JTalk辞書を事前ダウンロード
- ✅ **モデル常駐化**: 初期化31秒後、永続的に3-5秒応答
- ✅ **Torch.compile最適化**: GPU + TensorCore最適化で高速推論
- ✅ **GPU排他制御**: 複数リクエストの安定処理
- ✅ **詳細統計表示**: 生成時間、音声品質、リアルタイム係数
- ✅ **感情表現**: より自然で豊かな感情表現対応

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

### 3. 🌟 **AkitoP/GPT-SoVITS-JA-H FastAPI サーバー（推奨・最新）**

#### 3.1 日本語特化モデルダウンロード
```bash
# AkitoP/GPT-SoVITS-JA-H 日本語特化モデル（650時間学習済み）をダウンロード
cd /home/adama && wget -O "hscene-e17.ckpt" "https://huggingface.co/AkitoP/GPT-SoVITS-JA-H/resolve/main/hscene-e17.ckpt"

# 正しい場所にコピー（要sudo）
sudo cp "/home/adama/hscene-e17.ckpt" "/home/adama/gpts/Gptsovits/models/v4/GPT-SoVITS/gpt-sovits-ja-h/"
```

#### 3.2 🚀 **FastAPI サーバー起動（2025年7月最新版）**
```bash
# プロジェクトディレクトリに移動
cd "/home/adama/gpts/Gptsovits"

# ⚡ FastAPI サーバー起動
# 事前ダウンロード機能: 言語検出モデル(125MB) + Open JTalk辞書(22.6MB) + その他依存関係
# 初期化31秒後、永続的に3-5秒の超高速応答（REST APIで簡単操作）
docker run --gpus all -d -p 8000:8000 --privileged --name gpt-sovits-api \
  -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output \
  -v $(pwd)/scripts:/app/scripts \
  -v $(pwd)/models/v4/GPT-SoVITS/gpt-sovits-ja-h:/app/GPT_SoVITS/pretrained_models/gpt-sovits-ja-h \
  -v /usr/lib/wsl:/usr/lib/wsl -e LD_LIBRARY_PATH=/usr/lib/wsl/lib \
  gpt-sovits:v4 bash -c "pip install fastapi uvicorn python-multipart && python /app/scripts/fastapi_voice_server.py"
```

**🎯 最適化の内容:**
- **言語検出モデル**: 125MBモデルを事前ダウンロード&キャッシュ
- **Open JTalk辞書**: 22.6MB辞書を事前ダウンロード&キャッシュ  
- **依存関係**: jieba, torchaudioなどを事前初期化
- **結果**: 初回リクエストから3-5秒で応答（従来30秒→最大10倍高速化）

#### 3.3 🌐 **API使用方法**

**ヘルスチェック**:
```bash
curl http://localhost:8000/
```

**📁 音声生成（シンプル）**:
```bash
# 基本的な音声生成（参照音声: reference_5sec.wav）
curl -G "http://localhost:8000/clone-voice-simple" \
  --data-urlencode "ref_text=おはようございます" \
  --data-urlencode "target_text=こんにちは、音声生成のテストです" \
  > output/test.wav
```

**⚡ テストCLI使用**:
```bash
# 基本的なテスト
python3 scripts/test_fastapi_cli.py "こんにちは、音声生成のテストです"

# 感情表現テスト
python3 scripts/test_fastapi_cli.py "わあああ！すごい！本当に素晴らしい！"

# 一括テスト実行
./run_fastapi_tests.sh
```

**📊 生成結果**:
- **音声ファイル**: 自動的に`output/`ディレクトリに保存
- **詳細統計**: 生成時間、音声長、リアルタイム係数、品質指標
- **ファイル命名**: `cli_test_YYYYMMDD_HHMMSS_[テキスト].wav`

**サーバー管理**:
```bash
# ログ確認
docker logs gpt-sovits-api --tail 20

# サーバー停止
docker stop gpt-sovits-api && docker rm gpt-sovits-api

# 再起動
docker restart gpt-sovits-api
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

#### 🌟 **FastAPI サーバー関連（2025年7月最新最適化版）**
1. **音声が短い・前半スキップ**: 20文字未満のテキストは自動延長される
2. **APIリクエストエラー**: 正しいエンドポイントとパラメータを確認
3. **サーバー応答なし**: 初期化中（31秒・事前ダウンロード込み）の可能性、ログで確認
4. **ファイルが保存されない**: `/app/output`ボリュームマウント確認
5. **🚀 初回リクエストが速くない**: 最新最適化版では初回から4.6秒で応答
6. **事前ダウンロード確認**: `docker logs gpt-sovits-api | grep "事前ダウンロード"`でログ確認
7. **APIアクセスエラー**: ポート確認 `curl http://localhost:8000`、ファイアウォール設定確認

#### 🔧 **従来スクリプト関連**
5. **音声が短い**: `ref_free=True`が設定されているか確認
6. **GPUエラー**: WSL2用フラグ`-v /usr/lib/wsl:/usr/lib/wsl`が必須
7. **モデルが読み込めない**: ファイルパスとマウントを確認
8. **日本語特化モデルが使われない**: `test_voice_clone_ja_complete.py`を使用

## 📋 スクリプト詳細

### 🌟 **`fastapi_voice_server.py` (推奨)**
- **本格運用サーバー**: FastAPI REST API提供
- **高速生成**: 3-5秒/リクエスト（10倍高速）
- **自動最適化**: FP16 + Torch.compile + GPU加速
- **短文対応**: 20文字未満は自動延長
- **永続保存**: `output/`ディレクトリに自動保存
- **日本語特化**: hscene-e17.ckptモデル使用
- **REST API**: 軽量でプログラムから使いやすい

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

#### Gradio Web UI パラメータ
- `参照音声`: WAV/MP3ファイル（音声アップロードタブ）
- `参照音声のテキスト`: アップロードした音声またはデフォルト音声の内容
- `生成したいテキスト`: この声質で読み上げたい文章
- `温度パラメータ`: 生成の創造性（0.5-2.0、デフォルト1.0）

#### 従来スクリプト パラメータ
- `--ref-audio`: 参照音声ファイル（5秒程度推奨）
- `--ref-text`: 参照音声のテキスト
- `--target-text`: 生成したいテキスト
- `--output`: 出力ファイルパス
- `--sovits-model`: カスタムモデルパス（日本語特化版のみ）

## 📊 性能指標

### 🚀 **FastAPI サーバー（2025年7月最新最適化版）**
- **初期化時間**: 31秒（1回のみ・事前ダウンロード込み）
- **生成速度**: **3-5秒/リクエスト**（**初回から超高速**）
- **実測パフォーマンス**:
  - 初回リクエスト: **4.6秒**（従来30秒→最大10倍高速化）
  - 2回目以降: **3-5秒**（一貫して高速）
  - 長文生成: **5.5秒**（感情表現等）
- **スループット**: 12-20リクエスト/分
- **🎯 最適化機能**: 
  - 言語検出モデル事前ダウンロード&キャッシュ
  - Open JTalk辞書事前ダウンロード&キャッシュ
  - 依存関係事前初期化（jieba, torchaudio）
  - FP16 + Torch.compile + TensorCore最適化
  - GPU排他制御で安定動作

### 📈 **性能比較（初回リクエスト）**
| バージョン | 初期化時間 | 初回リクエスト | 2回目以降 | 高速化倍率 | UIタイプ |
|-----------|-----------|---------------|-----------|-----------|--------|
| **FastAPI サーバー** | 31秒 | **4.6秒** | **3-5秒** | **10倍** | REST API |
| 従来スクリプト | 25秒 | 27秒 | 27秒 | 0.9倍 | CLI |

### 🌐 **FastAPI の特長**
- **REST API**: `http://localhost:8000`
- **軽量**: プログラムから簡単に呼び出し可能
- **高速**: 事前ロードによる初回からの高速応答
- **詳細統計**: 生成時間、品質指標、パフォーマンス情報
- **テストCLI**: 使いやすいコマンドラインインターフェース

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

**作成日**: 2025-07-17  
**バージョン**: v3.0 (FastAPI サーバー)  
**動作確認環境**: WSL2 Ubuntu 22.04 + RTX 3050 + CUDA 12.1  
**特記事項**: 日本語特化モデル対応、感情表現対応、長文生成対応、FastAPI REST API対応

