#!/bin/bash
# GPT-SoVITS FastAPI テスト実行スクリプト

echo "🚀 GPT-SoVITS FastAPI テストスクリプト"
echo "=================================="

# カラー定義
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# FastAPIサーバーの状態確認
echo -e "\n${BLUE}📡 FastAPIサーバーの状態確認...${NC}"
if curl -s http://localhost:8000/ > /dev/null; then
    echo -e "${GREEN}✅ サーバーは起動しています${NC}"
else
    echo -e "${RED}❌ サーバーが起動していません${NC}"
    echo -e "\n${BLUE}以下のコマンドでサーバーを起動してください:${NC}"
    echo "cd /home/adama/gpts/Gptsovits"
    echo 'docker run --gpus all -d -p 8000:8000 --name gpt-sovits-api \'
    echo '  --privileged -v /usr/lib/wsl:/usr/lib/wsl -e LD_LIBRARY_PATH=/usr/lib/wsl/lib \'
    echo '  -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output \'
    echo '  -v $(pwd)/scripts:/app/scripts \'
    echo '  -v $(pwd)/models/v4/GPT-SoVITS/gpt-sovits-ja-h:/app/GPT_SoVITS/pretrained_models/gpt-sovits-ja-h \'
    echo '  gpt-sovits:v4 bash -c "pip install fastapi uvicorn python-multipart && python /app/scripts/fastapi_voice_server.py"'
    exit 1
fi

# テスト実行
echo -e "\n${BLUE}🎯 音声生成テストを開始します${NC}"

# テスト1: 基本的な挨拶
echo -e "\n${GREEN}[テスト1] 基本的な挨拶${NC}"
python scripts/test_fastapi_cli.py "こんにちは、音声生成のテストです。今日も良い一日をお過ごしください。"

# テスト2: 長文の技術説明
echo -e "\n${GREEN}[テスト2] 技術説明（英語混在）${NC}"
python scripts/test_fastapi_cli.py "今日のAI技術はMachine LearningやDeep Learningの進歩により、Natural Language ProcessingとComputer Visionの分野で革命的な変化をもたらしています。"

# テスト3: 感情豊かな表現
echo -e "\n${GREEN}[テスト3] 感情表現（興奮）${NC}"
python scripts/test_fastapi_cli.py "わあああ！すごい！本当に素晴らしい結果です！これは期待以上の品質です！"

# テスト4: 短文テスト（20文字未満、自動延長される）
echo -e "\n${GREEN}[テスト4] 短文テスト${NC}"
python scripts/test_fastapi_cli.py "はい、わかりました。"

# 結果確認
echo -e "\n${BLUE}📊 生成された音声ファイル:${NC}"
ls -la output/cli_test_*.wav 2>/dev/null | tail -5

echo -e "\n${GREEN}✅ テスト完了！${NC}"
echo -e "${BLUE}音声ファイルは output/ ディレクトリに保存されています${NC}"