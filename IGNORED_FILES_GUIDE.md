# 除外ファイル完全ガイド - IGNORED_FILES_GUIDE.md

このドキュメントは、GitHubリポジトリから除外された大容量ファイルと、それらを別途ダウンロードする方法をまとめています。

## 📊 ファイル除外統計

- **ローカル総ファイル数**: 2,053個
- **Git追跡ファイル数**: 342個  
- **除外ファイル数**: 1,711個 (約83%が除外)

## 🚫 .gitignoreによる除外ファイル

### 大容量モデルファイル
```gitignore
*.safetensors
*.ckpt  
*.pth
*.tar
*.bin
*.model
*.onnx
*.trt
*.engine
*.gguf
```

### 大容量メディアファイル
```gitignore
*.mp4
*.avi
*.mkv
*.mov
*.wav
*.mp3
*.flac
```

### 出力・キャッシュディレクトリ
```gitignore
output/
results/
temp/
logs/
checkpoints/
models/
weights/
```

## 📁 除外された重要ファイル一覧

### 1. GPT-SoVITS (Gptsovits/)

#### 必須モデルファイル
| ファイル名 | サイズ | パス | ダウンロード方法 |
|-----------|--------|------|-----------------|
| hscene-e17.ckpt | 148MB | models/v4/GPT-SoVITS/gpt-sovits-ja-h/ | [詳細](#gpt-sovits-models) |

#### 除外された出力・ログファイル
- `output/*.wav` - 生成された音声ファイル（数百個）
- `logs/*.log` - システムログファイル

### 2. LlamaCPP (llamacpp/)

#### 必須モデルファイル
| ファイル名 | サイズ | パス | ダウンロード方法 |
|-----------|--------|------|-----------------|
| Berghof-NSFW-7B.i1-Q4_K_S.gguf | 4.26GB | models/ | [詳細](#llamacpp-models) |

#### 除外された関連ファイル
- `logs/*.log` - チャットセッションログ（80個以上）
- `logs/sessions/` - セッション履歴

### 3. SadTalker系プロジェクト

#### faster-sadtalker/checkpoints/
| ファイル名 | サイズ | 説明 |
|-----------|--------|------|
| SadTalker_V0.0.2_256.safetensors | ~500MB | 256px顔生成モデル |
| SadTalker_V0.0.2_512.safetensors | ~500MB | 512px顔生成モデル |
| SadTalker_V002.safetensors | ~300MB | メインモデル |
| facevid2vid_00189-model.pth.tar | ~200MB | 動画生成モデル |
| mapping_00109-model.pth.tar | ~100MB | マッピングモデル |
| mapping_00229-model.pth.tar | ~100MB | マッピングモデル |
| auido2exp_00300-model.pth | ~50MB | 音声→表情モデル |
| auido2pose_00140-model.pth | ~50MB | 音声→ポーズモデル |
| epoch_20.pth | ~150MB | 学習済みモデル |
| wav2lip.pth | ~44MB | Wav2Lipモデル |
| wav2lip_gan.pth | ~44MB | Wav2Lip GANモデル |
| s3fd-619a316812.pth | ~90MB | 顔検出モデル |

#### faster-sadtalker/gfpgan/weights/
| ファイル名 | サイズ | 説明 |
|-----------|--------|------|
| GFPGANv1.4.onnx | ~350MB | 顔画質向上モデル |
| alignment_WFLW_4HG.pth | ~270MB | 顔アライメント |
| detection_Resnet50_Final.pth | ~110MB | 顔検出 |
| parsing_parsenet.pth | ~85MB | 顔パーシング |

### 4. Super Wav2Lip プロジェクト

#### 除外された大容量zipファイル
| ファイル名 | サイズ | 内容 | 代替ダウンロード |
|-----------|--------|------|-----------------|
| wav2lip_onnx_models.zip | 258MB | ONNX最適化モデル集 | [詳細](#wav2lip-onnx) |
| wav2lip_face_occluder.zip | 62MB | 顔遮蔽機能モデル | [詳細](#face-occluder) |
| wav2lip_seg_mask.zip | 50MB | セグメンテーションマスク | [詳細](#seg-mask) |
| wav2lip_insightface_func.zip | ~45MB | InsightFace機能モジュール | [詳細](#insightface) |

#### 除外されたONNXモデル (147個)
```
super wav2lip/src/enhancers/*.onnx
super wav2lip/src/face_occluder/*.onnx  
super wav2lip/src/seg_mask/*.onnx
super wav2lip/models/onnx/*.onnx
```

### 5. 出力・テンポラリファイル

#### 動画出力ファイル
- `output/result_*.mp4` - 生成されたリップシンク動画（約40個）
- `results/*.mp4` - SadTalker出力動画

#### 音声出力ファイル
- `fastapi_*.wav` - GPT-SoVITS生成音声（200個以上）
- `cli_test_*.wav` - テスト音声ファイル

#### その他除外ファイル
- `.git/lfs/incomplete/` - Git LFS不完全ファイル
- `backup/` - バックアップディレクトリ（複数プロジェクト）
- `__pycache__/` - Python キャッシュファイル

## 📥 詳細ダウンロード手順

### <a id="gpt-sovits-models"></a>GPT-SoVITS Models

#### hscene-e17.ckpt (148MB)
```bash
cd Gptsovits/models/v4/GPT-SoVITS/gpt-sovits-ja-h/

# Method 1: HuggingFace Hub
wget https://huggingface.co/spaces/keisuke/hscene-voice-clone/resolve/main/hscene-e17.ckpt

# Method 2: 手動ダウンロード
# https://huggingface.co/spaces/keisuke/hscene-voice-clone/tree/main
# ダウンロード後: cp /mnt/c/Users/YourName/Downloads/hscene-e17.ckpt ./

# Method 3: Original source (if available)
# wget https://drive.google.com/uc?id=ORIGINAL_FILE_ID
```

### <a id="llamacpp-models"></a>LlamaCPP Models

#### Berghof-NSFW-7B.i1-Q4_K_S.gguf (4.26GB)
```bash
cd llamacpp/models/

# Method 1: HuggingFace (推奨)
wget https://huggingface.co/TheBloke/Berghof-NSFW-7B-GGUF/resolve/main/Berghof-NSFW-7B.i1-Q4_K_S.gguf

# Method 2: Git LFS (大容量のため時間がかかる)
git lfs clone https://huggingface.co/TheBloke/Berghof-NSFW-7B-GGUF
cp Berghof-NSFW-7B-GGUF/Berghof-NSFW-7B.i1-Q4_K_S.gguf ./
rm -rf Berghof-NSFW-7B-GGUF

# Method 3: 手動ダウンロード (推奨 - 最も確実)
# https://huggingface.co/TheBloke/Berghof-NSFW-7B-GGUF/tree/main
# ダウンロード後: cp /mnt/c/Users/YourName/Downloads/Berghof-NSFW-7B.i1-Q4_K_S.gguf ./
```

### SadTalker Models

#### 基本checkpointsダウンロード
```bash
cd faster-sadtalker/checkpoints/

# SadTalker official releases
REPO="https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc"
wget $REPO/mapping_00109-model.pth.tar
wget $REPO/mapping_00229-model.pth.tar  
wget $REPO/SadTalker_V0.0.2_256.safetensors
wget $REPO/SadTalker_V0.0.2_512.safetensors
wget $REPO/auido2exp_00300-model.pth
wget $REPO/auido2pose_00140-model.pth
wget $REPO/facevid2vid_00189-model.pth.tar
wget $REPO/epoch_20.pth
wget $REPO/wav2lip.pth
wget $REPO/wav2lip_gan.pth
wget $REPO/s3fd-619a316812.pth

# Alternative: Use included checkpoints.zip (if available)
# unzip checkpoints.zip
```

#### GFPGAN weightsダウンロード
```bash
cd ../gfpgan/weights/

# GFPGAN models
wget https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.onnx

# Face analysis models
FACE_REPO="https://github.com/xinntao/facexlib/releases/download"
wget $FACE_REPO/v0.1.0/alignment_WFLW_4HG.pth
wget $FACE_REPO/v0.1.0/detection_Resnet50_Final.pth  
wget $FACE_REPO/v0.2.2/parsing_parsenet.pth
```

### <a id="wav2lip-onnx"></a>Super Wav2Lip Models

#### 基本PyTorchモデル (必須)
```bash
cd "super wav2lip/models/"

# 基本モデル (44MB each)
wget https://github.com/Rudrabha/Wav2Lip/releases/download/v1.0.0/wav2lip.pth
wget https://github.com/Rudrabha/Wav2Lip/releases/download/v1.0.0/wav2lip_gan.pth
```

#### <a id="face-occluder"></a>顔遮蔽機能 (オプション)
```bash
# wav2lip_face_occluder.zip (62MB) に含まれるファイル:
# - src/face_occluder/face_occluder.onnx
# - その他関連ONNX models

# 必要に応じて元のSuper-Wav2Lipリポジトリから取得
# git clone https://github.com/SilentSwaps/Super-Wav2Lip
# cp Super-Wav2Lip/src/face_occluder/* "super wav2lip/src/face_occluder/"
```

#### <a id="seg-mask"></a>セグメンテーションマスク (オプション)
```bash
# wav2lip_seg_mask.zip (50MB) に含まれるファイル:
# - src/seg_mask/vox-5segments.onnx
# - その他セグメンテーション関連

# 高度な背景分離機能が必要な場合のみダウンロード
```

#### <a id="insightface"></a>InsightFace機能 (オプション)  
```bash
# wav2lip_insightface_func.zip に含まれるファイル:
# - src/insightface_func/models/antelope/scrfd_2.5g_bnkps.onnx
# - その他InsightFace関連モデル

# 高精度顔検出が必要な場合のみ
```

## 🔄 重複ファイルの最適化

### SadTalker関連の重複
以下のファイルは複数の場所に同じものが配置されています：

```bash
# faster-sadtalker/checkpoints/ と sadtalker/faster-SadTalker-API/checkpoints/
# 同じファイルが重複しているため、一箇所からコピー可能

cd sadtalker/faster-SadTalker-API/checkpoints/
cp ../../../faster-sadtalker/checkpoints/* ./

cd ../gfpgan/weights/  
cp ../../../../faster-sadtalker/gfpgan/weights/* ./
```

### バックアップディレクトリの管理
```bash
# super wav2lip/backup/ ディレクトリは開発履歴
# 基本的にスキップ可能だが、特定設定が必要な場合のみ使用
```

## 📋 最小構成セットアップ

基本機能のみで動作させる場合の最小必須ファイル：

### GPT-SoVITS (音声生成)
- `hscene-e17.ckpt` (148MB) - 日本語モデル

### LlamaCPP (チャット)  
- `Berghof-NSFW-7B.i1-Q4_K_S.gguf` (4.26GB) - LLMモデル

### Wav2Lip (基本リップシンク)
- `wav2lip_gan.pth` (44MB) - 基本モデル

**合計**: 約4.45GB

### 完全構成 (全機能)
上記 + SadTalker全モデル + GFPGAN + ONNX最適化モデル

**合計**: 約15-20GB

## ⚠️ 注意事項

1. **ダウンロード順序**: 大容量ファイルから順に、ネットワーク安定時に実行
2. **ディスク容量**: 完全セットアップには50GB以上の空き容量が必要
3. **バックアップ**: ダウンロード完了後、重要なモデルファイルはバックアップ推奨
4. **更新確認**: モデルファイルは定期的に更新される場合があります

## 🔗 関連リンク

- [SETUP_GUIDE.md](./SETUP_GUIDE.md) - 完全セットアップガイド
- [CLAUDE.md](./CLAUDE.md) - 技術仕様・運用ガイド  
- [GitHub Repository](https://github.com/gdysaugs/gpts) - 最新ソースコード

---

💡 **重要**: このガイドの情報は作成時点のものです。モデルファイルのURLや仕様は変更される可能性があるため、エラーが発生した場合は最新の公式ドキュメントを確認してください。