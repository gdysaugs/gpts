# 🎭 統合口パクシステム 使用ガイド

## 📋 システム概要

**テキストを入力するだけで、指定した声で口パクする動画を自動生成**

```
テキスト入力 → SoVITS音声生成 → Wav2Lip口パク → 最終動画
    ↓             ↓              ↓           ↓
 「こんにちは」   既存の声で      口パク      完成動画
                音声生成(3-5秒)  生成(41秒)  ダウンロード
```

## 🏗️ システム構成

### 4コンテナマイクロサービス
- **Redis**: GPU排他制御・分散ロック
- **SoVITS**: 音声クローニング (Port 8000)
- **Wav2Lip**: 口パク動画生成 (Port 8002)
- **Frontend**: Gradio Web UI (Port 7860)

## 🚀 システム起動

### 1. 簡単起動（推奨）
```bash
cd "/home/adama/gpts/super wav2lip"
./start_integrated_system.sh
```

### 2. 手動起動
```bash
# ディレクトリ準備
mkdir -p ./shared/{input,output,temp}
mkdir -p ./data/redis

# システム起動
docker-compose -f docker-compose-integrated.yml up -d

# ログ監視
docker-compose -f docker-compose-integrated.yml logs -f
```

## 🌐 Web UI 使用方法

### アクセス
**http://localhost:7860** にブラウザでアクセス

### 操作手順
1. **📝 テキスト入力**: 生成したい文章を入力
   - 例: "こんにちは！今日は素晴らしい天気ですね。"

2. **📹 動画ファイル**: 口パクさせたい人物の動画をアップロード
   - 形式: MP4, AVI, MOV
   - 推奨: 5-30秒程度

3. **🎵 参照音声**: クローンしたい声の音声ファイルをアップロード
   - 形式: WAV, MP3, M4A
   - 推奨: 5秒程度の明瞭な音声

4. **📄 参照テキスト**: 参照音声で話している内容を入力
   - デフォルト: "おはようございます"

5. **⚙️ 詳細設定**:
   - **顔強化**: gfpgan（高品質推奨）/ none（高速）
   - **感情レベル**: 1.0（標準）/ 1.5（感情豊か）/ 0.5（落ち着き）
   - **バッチサイズ**: 8（RTX 3050最適化）

6. **🚀 生成開始**: ボタンクリックで処理開始

7. **📺 結果確認**: 約50秒後に動画プレビュー＆ダウンロード

## ⏱️ 処理時間

| フェーズ | 時間 | 説明 |
|---------|------|------|
| API確認 | 1-2秒 | サービス健康状態チェック |
| 音声生成 | 3-5秒 | SoVITSでテキスト→音声変換 |
| GPU切替 | 1-2秒 | Redis Lock管理 |
| 口パク生成 | 41秒 | Wav2Lipで口パク動画作成 |
| **合計** | **47-50秒** | **完全自動処理** |

## 🔧 API エンドポイント

### SoVITS API (Port 8000)
- **ヘルスチェック**: `GET /health`
- **音声生成**: `GET /clone-voice-simple`
- **GPU統計**: `GET /gpu-stats`
- **Swagger UI**: http://localhost:8000/docs

### Wav2Lip API (Port 8002)
- **ヘルスチェック**: `GET /health`  
- **口パク生成**: `POST /generate-lipsync`
- **ファイルダウンロード**: `GET /download/{filename}`
- **Swagger UI**: http://localhost:8002/docs

## 🔍 監視・管理

### システム状態確認
```bash
# コンテナ状態
docker-compose -f docker-compose-integrated.yml ps

# リアルタイムログ
docker-compose -f docker-compose-integrated.yml logs -f

# 特定サービスログ
docker-compose -f docker-compose-integrated.yml logs sovits
docker-compose -f docker-compose-integrated.yml logs wav2lip
docker-compose -f docker-compose-integrated.yml logs frontend
```

### GPU使用状況
```bash
# ホストGPU監視
watch -n 1 nvidia-smi

# API経由GPU統計
curl http://localhost:8000/gpu-stats
curl http://localhost:8002/gpu-stats
```

### Redis監視
```bash
# Redis接続確認
docker-compose -f docker-compose-integrated.yml exec redis redis-cli ping

# Redis統計
docker-compose -f docker-compose-integrated.yml exec redis redis-cli info
```

## 🛠️ トラブルシューティング

### 📡 API接続エラー
**症状**: Web UIでAPI状態が🔴異常
**解決策**: 
```bash
# サービス再起動
docker-compose -f docker-compose-integrated.yml restart sovits wav2lip

# ログ確認
docker-compose -f docker-compose-integrated.yml logs sovits wav2lip
```

### 🎤 音声生成失敗
**症状**: "SoVITS APIエラー"
**原因**: 
- 参照音声ファイル形式
- テキスト内容（特殊文字等）
- GPU メモリ不足

**解決策**:
```bash
# SoVITS サービス確認
curl http://localhost:8000/health

# メモリ使用量確認
nvidia-smi
```

### 🎭 口パク生成失敗
**症状**: "Wav2Lip APIエラー"
**原因**:
- 動画ファイル形式
- 顔検出失敗
- GPU メモリ不足

**解決策**:
```bash
# Wav2Lip サービス確認  
curl http://localhost:8002/health

# 動画形式確認
ffprobe input_video.mp4
```

### 🔒 Redis Lock エラー
**症状**: "Redis Lock失敗"
**解決策**:
```bash
# Redis確認
docker-compose -f docker-compose-integrated.yml exec redis redis-cli ping

# Lock状態確認
docker-compose -f docker-compose-integrated.yml exec redis redis-cli keys "gpu_lock:*"
```

### 🌐 Frontend接続失敗
**症状**: Web UIにアクセスできない
**解決策**:
```bash
# Frontend再起動
docker-compose -f docker-compose-integrated.yml restart frontend

# ポート確認
netstat -tulpn | grep 7860
```

## 🛑 システム停止

### 通常停止
```bash
docker-compose -f docker-compose-integrated.yml down
```

### 完全削除（データ含む）
```bash
docker-compose -f docker-compose-integrated.yml down -v --rmi all
```

## 📁 ファイル構造

```
/home/adama/gpts/super wav2lip/
├── docker-compose-integrated.yml    # メイン設定
├── start_integrated_system.sh       # 起動スクリプト
├── gradio_wav2lip_ui.py             # 統合Frontend UI
├── src/
│   ├── fastapi_wav2lip_redis.py     # Wav2Lip Redis版
│   └── gpu_redis_lock.py            # GPU Lock管理
├── shared/                          # 共有データ
│   ├── input/                       # 入力ファイル
│   ├── output/                      # 出力ファイル
│   └── temp/                        # 一時ファイル
└── models/                          # 必須モデル
    └── onnx/
        └── wav2lip_gan.onnx
```

## 🎯 使用例

### 基本例
- **テキスト**: "こんにちは！今日は良い天気ですね。"
- **動画**: 人物の顔が明確な5-10秒の動画
- **参照音声**: "おはようございます"を話した5秒の音声
- **結果**: 指定した声で挨拶する口パク動画

### 感情表現例
- **テキスト**: "わあああ！すごい！本当に素晴らしい！"
- **感情レベル**: 1.5（感情豊か）
- **結果**: 興奮した感情での口パク動画

### 技術説明例
- **テキスト**: "Machine LearningとDeep Learningの発展により..."
- **結果**: 英語混在の技術解説口パク動画

## 🎉 完成！

**これで「文章を打ち込むとソース音声の声で口パクする動画が生成される」完全統合システムが使用可能です！**