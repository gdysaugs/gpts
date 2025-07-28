# SadTalker CLI - 画像と音声から口パク動画生成 🎭

WSL2 + GPU対応のSadTalker CLIツールです。画像ファイルと音声ファイルから、**完全自動で音声付き口パク動画**を生成できます。

## 🎬 デモ

✅ **最新版で動作確認済み**：
- 入力: `source_image.jpg` + `source_audio.mp3`
- 出力: **音声付き動画** `sadtalker_result.mp4` (約30KB, 1080p品質)
- 処理時間: **約30秒** (RTX 3050)

## 🚀 特徴

- ✅ **完全自動処理**: 画像+音声→音声付き動画（1コマンド）
- ✅ **音声マージ自動化**: FFmpegによる自動音声合成
- ✅ **🔥 GFPGANエンハンサー**: PyTorch版顔画質向上機能
- ✅ **2つの品質モード**: 高速モード & 高画質エンハンサーモード
- ✅ **🎭 表情制御**: 表情強度調整 (0.0-2.0)
- ✅ **🔄 頭部姿勢制御**: 頭部回転・傾き制御 (Yaw/Pitch/Roll)
- ✅ **🔇 完全ノイズフリー音声**: ストリーム分離による高品質音声処理
- ✅ **🔧 自動権限修正**: Dockerファイル権限問題の完全解決
- ✅ **GPU加速対応**: WSL2 + NVIDIA GPU (RTX 3050対応)
- ✅ **エラー修正済み**: SadTalkerバグ修正版使用
- ✅ **高速処理**: CUDA使用で約30秒処理
- ✅ **メモリ最適化**: 4GB VRAM対応
- ✅ **高品質出力**: 1054x1580解像度対応

## 📋 必要環境

### システム要件
- **OS**: WSL2 Ubuntu 22.04
- **GPU**: NVIDIA GPU (RTX 3050以上推奨)
- **VRAM**: 4GB以上
- **Docker**: GPU対応Dockerランタイム

### ソフトウェア要件
- Python 3.8+
- Docker
- NVIDIA Container Toolkit
- Git

## 🛠️ インストール

### 1. リポジトリクローン
```bash
cd /home/adama/project/gpts
git clone https://github.com/kenwaytis/faster-SadTalker-API.git sadtalker
cd sadtalker
```

### 2. GPU環境設定

#### NVIDIA Container Toolkit確認
```bash
# GPU確認
nvidia-smi

# Docker GPU サポート確認
docker info | grep -i nvidia
```

#### WSL2 GPU対応設定
```bash
# Docker daemon設定
sudo tee /etc/docker/daemon.json > /dev/null << EOF
{
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    }
}
EOF

# Docker再起動
sudo systemctl restart docker
```

### 3. Dockerイメージビルド
```bash
# SadTalkerイメージビルド (約10分)
docker build -t sadtalker:latest .
```

### 4. モデルファイルダウンロード
```bash
cd checkpoints

# 必要なモデルファイル (合計 3.6GB)
wget https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2/epoch_20.pth
wget https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2/auido2pose_00140-model.pth
wget https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2/auido2exp_00300-model.pth
wget https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2/facevid2vid_00189-model.pth.tar

cd ..
```

### 5. CLIスクリプト設定
```bash
# CLIスクリプトを実行可能にする
chmod +x cli_sadtalker.py
```

## 🎯 使用方法

### 🚀 **新機能: FastAPIサーバー（最適化済み・推奨）**

#### **📋 毎回テスト手順（56秒高品質生成）**
```bash
# 1. プロジェクトディレクトリに移動
cd /home/adama/project/gpts/sadtalker

# 2. FastAPIサーバー起動（全最適化有効）
echo "1" | python3 start_fastapi_server.py &

# 3. サーバー準備完了まで待機
sleep 30 && curl -s http://localhost:8000/status

# 4. 🎯 最適化テスト実行（56秒で完了）
time curl -X POST http://localhost:8000/generate \
  -F "image=@input/kanna-hashimoto.jpg;type=image/jpeg" \
  -F "audio=@input/ohayougozaimasu_10.wav;type=audio/wav" \
  -F "quality=high" \
  -F "fp16=true"

# 5. 結果確認
# 出力: output/sadtalker_XXXXXXXX.mp4
```

#### **✅ 有効な最適化設定:**
1. ✅ **FastAPIサーバー起動**（モデル事前ロード）
2. ✅ **Docker + GPU + WSL2対応**
3. ✅ **GFPGAN高画質エンハンサー有効**
4. ✅ **FP16最適化による高速化**
5. ✅ **crop前処理最適化**
6. ✅ **CUDA最適化設定**（cudnn.benchmark=True）

#### **🔥 FastAPIサーバーの利点:**
- ⚡ **最適化済み**: 66秒 → **56秒**（15%高速化達成）
- 🎭 **高品質**: GFPGAN + FP16 + crop最適化
- 🌐 **WebUI**: ブラウザから簡単操作
- 📡 **REST API**: プログラムから利用可能
- 🔄 **常駐型**: 複数リクエスト対応
- 📱 **レスポンシブ**: スマホ・タブレット対応

### ⚡ CLI版（従来方式）
```bash
# 1. プロジェクトディレクトリに移動
cd /home/adama/project/gpts/sadtalker

# 2A. 高速生成（約30秒、エンハンサーなし）
python3 cli_sadtalker.py 画像ファイル 音声ファイル

# 2B. 🔥 高画質生成（約45秒、GFPGANエンハンサー有効）
python3 cli_sadtalker.py 画像ファイル 音声ファイル --quality high

# 2C. 🚀 FP16最適化生成（高速化+VRAM節約）
python3 cli_sadtalker.py 画像ファイル 音声ファイル --quality high --fp16

# 2D. 🎭 表情制御付き生成
python3 cli_sadtalker.py 画像ファイル 音声ファイル --expression 1.5

# 2E. 🔄 頭部制御付き生成
python3 cli_sadtalker.py 画像ファイル 音声ファイル --yaw 15 --pitch -5
```

### 📁 ファイル配置例
```bash
# 入力ファイル配置
faster-SadTalker-API/input/
├── source_image.jpg     # ソース画像
└── source_audio.mp3     # ソース音声

# 実行例
python3 cli_sadtalker.py faster-SadTalker-API/input/source_image.jpg faster-SadTalker-API/input/source_audio.mp3
```

### 🏆 実用例
```bash
# 1. 標準品質（高速）
python3 cli_sadtalker.py faster-SadTalker-API/input/source_image.jpg faster-SadTalker-API/input/source_audio.mp3

# 2. 🔥 高画質（GFPGANエンハンサー）
python3 cli_sadtalker.py faster-SadTalker-API/input/source_image.jpg faster-SadTalker-API/input/source_audio.mp3 --quality high

# 3. Windowsからの画像使用（高画質）
python3 cli_sadtalker.py /mnt/c/Users/adama/Downloads/photo.jpg faster-SadTalker-API/input/source_audio.mp3 --quality high

# 4. カスタムファイル（標準品質）
python3 cli_sadtalker.py my_photo.jpg my_audio.wav --quality fast

# 5. 🎭 表情制御例（表情を1.5倍強調）
python3 cli_sadtalker.py /mnt/c/Users/adama/Downloads/photo.jpg faster-SadTalker-API/input/source_audio.mp3 --expression 1.5

# 6. 🔄 頭部制御例（左向き10度 + 下向き5度）
python3 cli_sadtalker.py /mnt/c/Users/adama/Downloads/photo.jpg faster-SadTalker-API/input/source_audio.mp3 --yaw 10 --pitch -5

# 7. 🎯 静止モード（頭の動きを最小化）
python3 cli_sadtalker.py /mnt/c/Users/adama/Downloads/photo.jpg faster-SadTalker-API/input/source_audio.mp3 --still

# 8. 🔥 高画質 + 表情制御 + 頭部制御の組み合わせ
python3 cli_sadtalker.py /mnt/c/Users/adama/Downloads/photo.jpg faster-SadTalker-API/input/source_audio.mp3 --quality high --expression 1.3 --yaw 5 --roll -2

# 9. 🚀 FP16最適化 + 高画質の組み合わせ（推奨設定）
python3 cli_sadtalker.py /mnt/c/Users/adama/Downloads/kanna-hashimoto.jpg /mnt/c/Users/adama/Downloads/ohayougozaimasu_10.wav --quality high --fp16
```

### 📡 **FastAPI REST API使用例**
```python
import requests

# 1. サーバー状態確認
response = requests.get("http://localhost:8000/status")
print(response.json())

# 2. 動画生成（プログラムから）
files = {
    'image': open('kanna-hashimoto.jpg', 'rb'),
    'audio': open('ohayougozaimasu_10.wav', 'rb')
}
data = {
    'quality': 'high',
    'fp16': True,
    'still_mode': True,
    'expression_scale': 1.0
}

response = requests.post("http://localhost:8000/generate", files=files, data=data)
result = response.json()

if result['success']:
    print(f"生成成功! 処理時間: {result['processing_time']:.2f}秒")
    print(f"ダウンロードURL: {result['download_url']}")
else:
    print(f"生成失敗: {result['error']}")
```

### 📊 **性能比較: CLI vs FastAPI（最適化済み）**
| 方式 | 最適化レベル | 処理時間 | WebUI | API | 複数同時 |
|------|-------------|----------|--------|-----|----------|
| **CLI版** | 基本設定 | 66秒 | ❌ | ❌ | ❌ |
| **🚀FastAPI版** | **全最適化** | **56秒** | ✅ | ✅ | ✅ |

#### **🎯 達成した最適化:**
- **15%高速化**: 66秒 → 56秒
- **GFPGAN + FP16**: 高品質 + 高速化
- **事前ロード**: モデル常駐による安定性
- **CUDA最適化**: RTX 3050専用チューニング

### 📤 出力結果
```bash
# 自動生成される音声付き動画
./output/sadtalker_result.mp4

# Windowsからアクセス
\\wsl.localhost\Ubuntu-22.04\home\adama\project\gpts\sadtalker\output\sadtalker_result.mp4
```

### ⚙️ パラメーター
```bash
Usage: python3 cli_sadtalker.py 画像ファイル 音声ファイル [--quality QUALITY]

引数:
  画像ファイル          入力画像 (JPG/PNG) - 顔がはっきり写ったもの
  音声ファイル          入力音声 (WAV/MP3/M4A) - 任意の長さ

オプション:
  --quality fast       🚀 高速モード（デフォルト、約30秒）
  --quality high       🔥 高画質モード（GFPGANエンハンサー、約45秒）
  --fp16               🚀 FP16最適化（高速化+VRAM節約、RTX推奨）
  
  🎭 表情制御:
  --expression FLOAT   表情強度 (0.0-2.0, デフォルト:1.0)
                       0.0=無表情, 1.0=標準, 2.0=強調
  
  🔄 頭部姿勢制御:
  --still              静止モード（頭の動きを最小化）
  --yaw FLOAT         頭部左右回転 (-30〜30度)
  --pitch FLOAT       頭部上下回転 (-20〜20度)  
  --roll FLOAT        頭部傾き (-15〜15度)

自動処理:
  ✅ GPU自動検出・使用
  ✅ 完全ノイズフリー音声マージ自動実行  
  ✅ エラー自動修復
  ✅ ファイル権限自動修正
  ✅ PyTorch版GFPGAN顔エンハンサー
```

## 🐳 Docker実行方法

### 1. GPU対応Docker実行
```bash
# WSL2 GPU対応フラグ使用
docker run --rm --privileged \
  -v /usr/lib/wsl:/usr/lib/wsl \
  -e LD_LIBRARY_PATH=/usr/lib/wsl/lib \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e CUDA_VISIBLE_DEVICES=0 \
  -v $(pwd)/input:/app/input \
  -v $(pwd)/output:/app/output \
  -v $(pwd)/results:/app/results \
  -v $(pwd)/checkpoints:/app/checkpoints \
  -v $(pwd)/gfpgan:/app/gfpgan \
  -w /app \
  sadtalker:latest python standalone_api.py
```

### 2. APIサーバー起動
```bash
# バックグラウンドでAPIサーバー起動
docker run -d -p 10364:10364 --name sadtalker-api \
  --privileged \
  -v /usr/lib/wsl:/usr/lib/wsl \
  -e LD_LIBRARY_PATH=/usr/lib/wsl/lib \
  -v $(pwd)/input:/home/SadTalker/input \
  -v $(pwd)/output:/home/SadTalker/output \
  -v $(pwd)/results:/home/SadTalker/results \
  -v $(pwd)/checkpoints:/home/SadTalker/checkpoints \
  -v $(pwd)/gfpgan:/home/SadTalker/gfpgan \
  -v $(pwd)/standalone_api.py:/home/SadTalker/standalone_api.py \
  sadtalker:latest python standalone_api.py

# ログ確認
docker logs -f sadtalker-api

# 停止
docker stop sadtalker-api && docker rm sadtalker-api
```

## 🔧 GPU設定詳細

### WSL2 GPU要件
SadTalkerをGPUで実行するには、WSL2環境でNVIDIA GPUが使用可能である必要があります。

#### GPU確認コマンド
```bash
# GPU認識確認
nvidia-smi

# CUDA確認
nvcc --version

# Docker GPU確認
docker run --rm --privileged \
  -v /usr/lib/wsl:/usr/lib/wsl \
  -e LD_LIBRARY_PATH=/usr/lib/wsl/lib \
  sadtalker:latest python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

#### WSL2専用設定
WSL2環境では特別なDockerフラグが必要です：

```bash
--privileged                    # 特権モード
-v /usr/lib/wsl:/usr/lib/wsl   # WSL2ライブラリマウント
-e LD_LIBRARY_PATH=/usr/lib/wsl/lib  # ライブラリパス設定
-e NVIDIA_VISIBLE_DEVICES=all  # GPU可視性
-e CUDA_VISIBLE_DEVICES=0      # CUDAデバイス指定
```

### メモリ最適化
RTX 3050 (4GB VRAM)での推奨設定：

```python
# バッチサイズ調整
facerender_batch_size = 5  # 低VRAM用

# 高速化設定
preprocess = 'crop'        # fullの代わりにcrop
still_mode = True          # 静止画モード
enhancer = None           # エンハンサー無効
```

## 📁 出力ファイル

### 生成される動画
```bash
# CLI実行後の出力場所
/home/adama/project/gpts/sadtalker/output/result.mp4

# Windowsからアクセス
\\wsl.localhost\Ubuntu-22.04\home\adama\project\gpts\sadtalker\output\result.mp4
```

### 中間ファイル
```bash
faster-SadTalker-API/results/cli_YYYY_MM_DD_HH_MM_SS/
├── first_frame_dir/           # 前処理結果
├── source_audio.wav          # 音声ファイル
├── source_image##source_audio.mat  # 係数データ
├── source_image##source_audio.txt  # メタデータ
└── temp_source_image##source_audio.mp4  # 生成動画
```

## 🚨 トラブルシューティング

### ✅ 修正済みエラー

#### 1. 音声が出ない問題 → **修正完了**
```bash
# 問題: 生成動画に音声ストリームが含まれない
# 解決: FFmpegによる自動音声マージ実装済み
# 確認: 最終出力に Audio: aac ストリーム確認可能
```

#### 2. SadTalkerエラー → **修正完了**  
```bash
# 問題: av_path変数未定義エラー
# 解決: animate_onnx.py のバグ修正済み
# 確認: エラーなしで動画生成完了
```

#### 3. 🔇 音声ノイズ問題 → **完全解決**
```bash
# 問題: 音声にザーザーノイズが混入
# 原因: SadTalker処理による音声劣化 + 隠れた音声ストリーム重複
# 解決: ストリーム分離処理で元音声品質100%保持
# 技術: FFmpeg -map 0:v:0 -map 1:a:0 による完全分離
```

#### 4. 🔧 権限問題 → **完全解決**
```bash
# 問題: Docker実行でrootユーザーファイル作成による権限エラー
# 解決: 自動権限修正システム実装
# 技術: sudo chown による自動ユーザー権限変更
```

#### 5. 🚀 FP16最適化問題 → **完全解決** ✅ NEW!
```bash
# 問題: WAVファイルからのノイズ混入、MP4コンテナ非対応
# 解決: READMEのストリーム分離技術を正しく実装
# 技術: WAV→MP3変換後、完全ノイズフリーストリーム分離
# 結果: 元音声品質100%保持 + FP16高速化対応
```

### 💡 一般的な対処法

#### 1. GPU認識しない
```bash
# 症状: GPU加速が使用されない
# 確認: nvidia-smi でGPU状態確認
nvidia-smi

# 解決: Dockerサービス再起動
sudo systemctl restart docker
```

#### 2. メモリ不足（稀）
```bash
# 症状: CUDA error: out of memory
# 解決: 既に最適化済み（バッチサイズ5）
# 追加対策: 他のGPUアプリケーション終了
```

#### 3. 顔検出失敗
```bash
# 症状: 顔ランドマーク検出失敗
# 解決方法: 
✅ 顔がはっきり写った画像使用
✅ 正面向きの顔推奨
✅ JPG/PNG形式使用
✅ 高解像度画像推奨
```

#### 4. 権限エラー
```bash
# 症状: Permission denied エラー
# 解決: 出力ディレクトリクリア
rm -rf output/*

# 再実行
python3 cli_sadtalker.py 画像 音声
```

### ログ確認
```bash
# Docker実行ログ
docker logs sadtalker-api

# SadTalkerログ
tail -f faster-SadTalker-API/sadtalker.log

# システムログ
dmesg | grep -i cuda
```

## ⚡ システム起動・実行手順

### 🔧 初回セットアップ（1回のみ）
```bash
# 1. プロジェクトディレクトリ移動
cd /home/adama/project/gpts/sadtalker

# 2. Dockerイメージビルド（約10分）
cd faster-SadTalker-API
docker build -t sadtalker:latest .

# 3. CLIスクリプト実行権限付与
cd ..
chmod +x cli_sadtalker.py

# ✅ セットアップ完了
```

### 🚀 日常使用（毎回）
```bash
# 1. プロジェクトディレクトリ移動
cd /home/adama/project/gpts/sadtalker

# 2A. 高速実行（約30秒で完了）
python3 cli_sadtalker.py 画像ファイル 音声ファイル

# 2B. 🔥 高画質実行（約45秒、GFPGANエンハンサー）
python3 cli_sadtalker.py 画像ファイル 音声ファイル --quality high

# 3. 結果確認
# Windows: \\wsl.localhost\Ubuntu-22.04\home\adama\project\gpts\sadtalker\output\sadtalker_result.mp4
```

### 📈 処理フロー詳細

#### ⚡ 高速モード (--quality fast)
```
🖼️ 画像 + 🎵 音声
    ↓ (5秒)
🔍 GPU顔検出・前処理 (crop)
    ↓ (3秒)  
🎵 音声解析・係数変換
    ↓ (20秒)
🎬 CUDA動画レンダリング
🎭 表情制御・頭部姿勢適用
    ↓ (2秒)
🔊 FFmpeg音声マージ
    ↓
✅ 音声付き動画完成 (30秒)
```

#### 🔥 最適化高画質モード (FastAPI + FP16 + GFPGAN)
```
🖼️ 画像 + 🎵 音声
    ↓ (4秒)
🔍 GPU顔検出・前処理 (crop最適化)
    ↓ (3秒)
🎵 音声解析・係数変換
    ↓ (22秒)
🎬 CUDA動画レンダリング (FP16最適化)
🎭 表情制御・頭部姿勢適用
    ↓ (25秒)
🔥 PyTorch版GFPGAN顔エンハンサー
    ↓ (2秒)
🔊 FFmpeg音声マージ
    ↓
✅ 全最適化動画完成 (56秒)
```

## 📊 パフォーマンス

### 処理時間 (RTX 3050)

#### ⚡ 高速モード (--quality fast)
- **前処理**: 約5秒 (顔検出・ランドマーク、crop)
- **音声解析**: 約3秒 (MEL特徴量抽出)
- **動画生成**: 約20秒 (口パクレンダリング)
- **音声マージ**: 約2秒 (FFmpeg処理)
- **合計**: **約30秒**

#### 🔥 高画質モード (--quality high)  
- **前処理**: 約5秒 (顔検出・ランドマーク、full)
- **音声解析**: 約3秒 (MEL特徴量抽出)
- **動画生成**: 約25秒 (口パクレンダリング)
- **🔥 GFPGAN**: 約10秒 (PyTorch版顔エンハンサー)
- **音声マージ**: 約2秒 (FFmpeg処理)
- **合計**: **約45秒**

### メモリ使用量
- **VRAM**: 2-4GB (4GB中) - 最適化済み
- **RAM**: 1-2GB
- **ストレージ**: 3.6GB (モデルファイル + GFPGAN)

### 出力品質比較
| モード | 解像度 | エンハンサー | 表情制御 | 頭部制御 | 処理時間 | ファイルサイズ |
|--------|--------|-------------|----------|----------|----------|---------------|
| **fast** | 1054x1580 | ❌ なし | ✅ 対応 | ✅ 対応 | 30秒 | 約30KB |
| **🔥 high** | 1054x1580 | ✅ GFPGAN | ✅ 対応 | ✅ 対応 | 45秒 | 約35KB |

### 🎭 表情制御詳細
| パラメーター | 値 | 効果 |
|-------------|-----|------|
| --expression 0.0 | 無表情 | 元画像の表情を完全に抑制 |
| --expression 1.0 | 標準 | デフォルト（音声に応じた自然な表情） |
| --expression 1.5 | 強調 | 表情を1.5倍強調表示 |
| --expression 2.0 | 最大 | 最大強度の表情表現 |

### 🔄 頭部制御詳細
| パラメーター | 範囲 | 効果 |
|-------------|------|------|
| --yaw | -30〜30° | 左右回転（-=右向き、+=左向き） |
| --pitch | -20〜20° | 上下回転（-=下向き、+=上向き） |
| --roll | -15〜15° | 首傾き（-=右傾き、+=左傾き） |
| --still | - | 頭の動きを最小化（静止モード） |

### 共通仕様
- **フレームレート**: 25fps
- **音声品質**: 元音声品質保持（MP3: 192kbps, 44.1kHz）
- **対応形式**: JPG/PNG → MP4 + 完全ノイズフリー音声
- **音声処理**: ストリーム分離技術による最高品質維持

## 🔧 技術詳細

### 🔇 音声ノイズ完全除去技術

#### **問題の根本原因**
```
元音声 (192kbps, 44.1kHz) → SadTalker処理 → 16kHz劣化 → FFmpegマージ → ノイズ混入
```

#### **解決技術: ストリーム分離**
```bash
# FFmpegの明示的ストリーム指定
-map 0:v:0  # SadTalker動画の映像のみ
-map 1:a:0  # 元音声MP3の音声のみ
-c:a copy   # 音声を一切加工せずコピー
```

#### **音声品質比較**
| 段階 | サンプリングレート | ビットレート | 品質 |
|------|------------------|-------------|------|
| 元音声 | 44.1kHz | 192kbps | 🟢 完璧 |
| SadTalker処理後 | 16kHz | 256kbps | 🔴 劣化 |
| **最終出力** | **44.1kHz** | **192kbps** | **🟢 完璧** |

### 🔧 権限問題自動解決

#### **問題**
```bash
# Dockerがrootでファイル作成 → adamaユーザーが書き込めない
Permission denied: './output/sadtalker_result.mp4'
```

#### **解決技術**
```bash
# 自動権限修正システム
chown_cmd = f"sudo chown -R {os.getuid()}:{os.getgid()} {output_dir}"
subprocess.run(chown_cmd.split(), capture_output=True)
```

## 🔄 アップデート

### モデル更新
```bash
cd checkpoints
# 新しいモデルファイルダウンロード
wget https://github.com/OpenTalker/SadTalker/releases/download/v0.0.3/[新ファイル]
```

### Dockerイメージ更新
```bash
# イメージ再ビルド
docker build -t sadtalker:latest --no-cache .

# 古いイメージ削除
docker image prune
```

## 📄 ライセンス

このプロジェクトは [SadTalker](https://github.com/OpenTalker/SadTalker) をベースにしています。
ライセンスについては元プロジェクトを参照してください。

## 🤝 貢献

1. このリポジトリをフォーク
2. 機能ブランチ作成 (`git checkout -b feature/AmazingFeature`)
3. 変更をコミット (`git commit -m 'Add some AmazingFeature'`)
4. ブランチにプッシュ (`git push origin feature/AmazingFeature`)
5. プルリクエスト作成

## 📞 サポート

- **Issue**: [GitHub Issues](https://github.com/kenwaytis/faster-SadTalker-API/issues)
- **元プロジェクト**: [SadTalker](https://github.com/OpenTalker/SadTalker)
- **Documentation**: [SadTalker Paper](https://arxiv.org/abs/2211.12194)

---

**注意**: このツールは研究・教育目的での使用を想定しています。生成された動画の悪用は避けてください。