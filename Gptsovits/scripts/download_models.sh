#!/bin/bash
# GPT-SoVITS v4モデルダウンロードスクリプト

set -e

echo "===================================="
echo "GPT-SoVITS v4 モデルダウンロード"
echo "===================================="

# カラー出力用
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 作業ディレクトリ
WORK_DIR="/home/adama/.claude/projects/Gptsovits"
MODEL_DIR="${WORK_DIR}/models/v4"

# Git LFSの確認
if ! command -v git-lfs &> /dev/null; then
    echo -e "${RED}エラー: Git LFSがインストールされていません${NC}"
    echo "以下のコマンドでインストールしてください:"
    echo "sudo apt-get update && sudo apt-get install git-lfs"
    exit 1
fi

# ディレクトリ作成
echo -e "${YELLOW}モデルディレクトリを作成中...${NC}"
mkdir -p "${MODEL_DIR}"
cd "${MODEL_DIR}"

# Git LFS初期化
echo -e "${YELLOW}Git LFSを初期化中...${NC}"
git lfs install

# HuggingFaceからv4モデルをクローン
echo -e "${YELLOW}HuggingFaceからv4モデルをダウンロード中...${NC}"
echo "これには時間がかかる場合があります..."

if [ -d "GPT-SoVITS" ]; then
    echo -e "${YELLOW}既存のディレクトリを削除中...${NC}"
    rm -rf GPT-SoVITS
fi

# クローン実行
git clone https://huggingface.co/lj1995/GPT-SoVITS.git

cd GPT-SoVITS

# LFSファイルのフェッチ
echo -e "${YELLOW}LFSファイルをフェッチ中...${NC}"
git lfs fetch --all

# LFSファイルのチェックアウト
echo -e "${YELLOW}LFSファイルをチェックアウト中...${NC}"
git lfs checkout

# 必要なファイルの確認
echo -e "${YELLOW}ダウンロードしたファイルを確認中...${NC}"

# v4モデルファイルの存在確認
if [ -f "gsv-v4-pretrained/s2Gv4.pth" ] && [ -f "gsv-v4-pretrained/vocoder.pth" ]; then
    echo -e "${GREEN}✓ v4モデルファイルが正常にダウンロードされました${NC}"
    
    # ファイルサイズ確認
    echo "ファイルサイズ:"
    ls -lh gsv-v4-pretrained/s2Gv4.pth gsv-v4-pretrained/vocoder.pth
else
    echo -e "${RED}✗ v4モデルファイルのダウンロードに失敗しました${NC}"
    exit 1
fi

# その他の必要なモデルも確認
echo -e "${YELLOW}その他の必要なモデルを確認中...${NC}"

# chinese-hubert-baseの確認
if [ -d "chinese-hubert-base" ]; then
    echo -e "${GREEN}✓ chinese-hubert-base が見つかりました${NC}"
else
    echo -e "${YELLOW}! chinese-hubert-base が見つかりません${NC}"
fi

# chinese-roberta-wwm-ext-largeの確認
if [ -d "chinese-roberta-wwm-ext-large" ]; then
    echo -e "${GREEN}✓ chinese-roberta-wwm-ext-large が見つかりました${NC}"
else
    echo -e "${YELLOW}! chinese-roberta-wwm-ext-large が見つかりません${NC}"
fi

# ダウンロード完了
echo -e "${GREEN}===================================="
echo -e "モデルのダウンロードが完了しました！"
echo -e "====================================${NC}"

echo ""
echo "ダウンロードしたモデルの場所:"
echo "${MODEL_DIR}/GPT-SoVITS/"
echo ""
echo "次のステップ:"
echo "1. Dockerイメージをビルド: cd ${WORK_DIR} && DOCKER_BUILDKIT=1 docker build -t gpt-sovits:v4 ."
echo "2. Docker Composeで起動: docker-compose up -d"