#!/bin/bash
"""
統合口パクシステム起動スクリプト
テキスト→音声→口パク動画の完全システム
"""

set -e

echo "🚀 統合口パクシステム起動スクリプト"
echo "========================================"

# カラー定義
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 関数定義
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 1. 前提条件チェック
print_status "前提条件チェック中..."

# Docker確認
if ! command -v docker &> /dev/null; then
    print_error "Dockerがインストールされていません"
    exit 1
fi

# Docker Compose確認
if ! command -v docker-compose &> /dev/null; then
    print_error "Docker Composeがインストールされていません"
    exit 1
fi

# GPU確認
if ! nvidia-smi &> /dev/null; then
    print_warning "NVIDIA GPUまたはドライバーが検出されません"
    print_warning "CPU模式で続行しますが、処理が非常に遅くなります"
fi

print_success "前提条件チェック完了"

# 2. ディレクトリ構造作成
print_status "ディレクトリ構造作成中..."

mkdir -p ./shared/{input,output,temp}
mkdir -p ./data/redis
mkdir -p ./config
mkdir -p ./logs

print_success "ディレクトリ構造作成完了"

# 3. モデルファイル確認
print_status "必須モデルファイル確認中..."

# SoVITS日本語特化モデル
SOVITS_MODEL="../Gptsovits/models/v4/GPT-SoVITS/gpt-sovits-ja-h/hscene-e17.ckpt"
if [ ! -f "$SOVITS_MODEL" ]; then
    print_warning "SoVITS日本語特化モデルが見つかりません: $SOVITS_MODEL"
    print_warning "標準モデルで続行されます"
else
    print_success "SoVITS日本語特化モデル確認完了"
fi

# Wav2Lipモデル
WAV2LIP_MODEL="./models/onnx/wav2lip_gan.onnx"
if [ ! -f "$WAV2LIP_MODEL" ]; then
    print_error "Wav2Lipモデルが見つかりません: $WAV2LIP_MODEL"
    print_error "Google Driveからダウンロードしてください"
    exit 1
else
    print_success "Wav2Lipモデル確認完了"
fi

# 顔検出モデル
FACE_MODEL="./src/utils/scrfd_2.5g_bnkps.onnx"
if [ ! -f "$FACE_MODEL" ]; then
    print_error "顔検出モデルが見つかりません: $FACE_MODEL"
    exit 1
else
    print_success "顔検出モデル確認完了"
fi

# 4. テスト用ファイル確認
print_status "テスト用ファイル確認中..."

# テスト動画
if [ ! -f "./input/videos/test_video.mp4" ]; then
    print_warning "テスト動画がありません - ウォームアップがスキップされます"
    print_warning "任意の動画を './input/videos/test_video.mp4' に配置してください"
fi

# テスト音声
if [ ! -f "./input/audio/test_audio.mp3" ]; then
    print_warning "テスト音声がありません - ウォームアップがスキップされます"
    print_warning "任意の音声を './input/audio/test_audio.mp3' に配置してください"
fi

# 参照音声
if [ ! -f "../Gptsovits/input/reference_5sec.wav" ]; then
    print_warning "参照音声がありません"
    print_warning "5秒程度の音声を '../Gptsovits/input/reference_5sec.wav' に配置してください"
fi

# 5. Redisの設定確認
print_status "Redis設定作成中..."

cat > ./config/redis.conf << 'EOF'
# Redis統合システム用設定
maxmemory 512mb
maxmemory-policy allkeys-lru
save 900 1
save 300 10  
save 60 10000
appendonly yes
appendfsync everysec
EOF

print_success "Redis設定作成完了"

# 6. システム起動
print_status "Docker Compose システム起動中..."
print_status "これには数分かかる場合があります..."

# 既存コンテナ停止
docker-compose -f docker-compose-integrated.yml down > /dev/null 2>&1 || true

# システム起動
if docker-compose -f docker-compose-integrated.yml up -d; then
    print_success "Docker Compose起動完了"
else
    print_error "Docker Compose起動失敗"
    exit 1
fi

# 7. 起動状況監視
print_status "サービス起動状況監視中..."

# サービス起動待機
sleep 10

# Redis確認
print_status "Redis サービス確認中..."
if docker-compose -f docker-compose-integrated.yml exec -T redis redis-cli ping > /dev/null 2>&1; then
    print_success "Redis サービス正常"
else
    print_warning "Redis サービス異常"
fi

# SoVITS確認（タイムアウト付き）
print_status "SoVITS サービス確認中（最大60秒）..."
COUNTER=0
while [ $COUNTER -lt 12 ]; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        print_success "SoVITS サービス正常"
        break
    fi
    sleep 5
    COUNTER=$((COUNTER + 1))
    echo -n "."
done

if [ $COUNTER -eq 12 ]; then
    print_warning "SoVITS サービス応答なし（初期化中の可能性）"
fi

# Wav2Lip確認（タイムアウト付き）
print_status "Wav2Lip サービス確認中（最大90秒）..."
COUNTER=0
while [ $COUNTER -lt 18 ]; do
    if curl -s http://localhost:8002/health > /dev/null 2>&1; then
        print_success "Wav2Lip サービス正常"
        break
    fi
    sleep 5
    COUNTER=$((COUNTER + 1))
    echo -n "."
done

if [ $COUNTER -eq 18 ]; then
    print_warning "Wav2Lip サービス応答なし（初期化中の可能性）"
fi

# Frontend確認
print_status "Frontend サービス確認中..."
sleep 5
if curl -s http://localhost:7860 > /dev/null 2>&1; then
    print_success "Frontend サービス正常"
else
    print_warning "Frontend サービス異常"
fi

# 8. 起動完了メッセージ
echo ""
echo "=========================================="
print_success "🎉 統合口パクシステム起動完了！"
echo "=========================================="
echo ""
echo "📱 アクセス方法:"
echo "   Frontend UI:    http://localhost:7860"
echo "   SoVITS API:     http://localhost:8000/docs" 
echo "   Wav2Lip API:    http://localhost:8002/docs"
echo ""
echo "🎯 使用方法:"
echo "   1. ブラウザで http://localhost:7860 にアクセス"
echo "   2. テキストを入力"
echo "   3. 動画ファイルをアップロード（口パクさせたい人物）"
echo "   4. 参照音声をアップロード（クローンしたい声）"
echo "   5. 生成ボタンをクリック"
echo "   6. 50秒程度で口パク動画が完成"
echo ""
echo "📊 監視コマンド:"
echo "   ログ監視:    docker-compose -f docker-compose-integrated.yml logs -f"
echo "   状態確認:    docker-compose -f docker-compose-integrated.yml ps"
echo "   停止:        docker-compose -f docker-compose-integrated.yml down"
echo ""
echo "⚡ 処理時間目安:"
echo "   音声生成:    3-5秒"
echo "   口パク生成:  41秒"
echo "   合計:        50秒程度"
echo ""

# 9. ログ監視オプション
read -p "リアルタイムログ監視を開始しますか？ (y/n): " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_status "リアルタイムログ監視開始（Ctrl+Cで終了）"
    docker-compose -f docker-compose-integrated.yml logs -f
fi

print_success "起動スクリプト完了"