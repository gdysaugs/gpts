#!/bin/bash
# LlamaCPP FastAPI テスト実行スクリプト

echo "🚀 LlamaCPP FastAPI テストスクリプト"
echo "===================================="

# カラー定義
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# FastAPIサーバーの状態確認
echo -e "\n${BLUE}📡 FastAPIサーバーの状態確認...${NC}"
if curl -s http://localhost:8001/health > /dev/null; then
    echo -e "${GREEN}✅ サーバーは起動しています${NC}"
else
    echo -e "${RED}❌ サーバーが起動していません${NC}"
    echo -e "\n${BLUE}以下のコマンドでサーバーを起動してください:${NC}"
    echo "cd /home/adama/gpts/llamacpp"
    echo 'docker run --gpus all --rm -it \'
    echo '  --privileged \'
    echo '  -v $(pwd)/models:/app/models \'
    echo '  -v $(pwd)/config:/app/config \'
    echo '  -v $(pwd)/logs:/app/logs \'
    echo '  -v $(pwd)/src:/app/src \'
    echo '  -v /usr/lib/wsl:/usr/lib/wsl \'
    echo '  -e LD_LIBRARY_PATH=/usr/lib/wsl/lib \'
    echo '  -e NVIDIA_VISIBLE_DEVICES=all \'
    echo '  -e CUDA_VISIBLE_DEVICES=0 \'
    echo '  -p 8001:8001 \'
    echo '  llama-cpp-python:cuda python /app/src/fastapi_chat_server.py'
    exit 1
fi

# ヘルスチェック詳細
echo -e "\n${BLUE}🔍 サーバー詳細情報:${NC}"
curl -s http://localhost:8001/health | python3 -m json.tool

# テスト実行
echo -e "\n${BLUE}🎯 インタラクティブチャットテストを開始します${NC}"

# テスト関数
test_api_call() {
    local test_name="$1"
    local endpoint="$2"
    local method="$3"
    local data="$4"
    
    echo -e "\n${YELLOW}[テスト] $test_name${NC}"
    
    if [ "$method" = "GET" ]; then
        response=$(curl -s -w "\n%{http_code}" "$endpoint")
    elif [ "$method" = "POST" ]; then
        response=$(curl -s -w "\n%{http_code}" -X POST -H "Content-Type: application/json" -d "$data" "$endpoint")
    elif [ "$method" = "DELETE" ]; then
        response=$(curl -s -w "\n%{http_code}" -X DELETE "$endpoint")
    fi
    
    http_code=$(echo "$response" | tail -n1)
    body=$(echo "$response" | sed '$d')
    
    if [ "$http_code" = "200" ]; then
        echo -e "${GREEN}✅ 成功 ($http_code)${NC}"
        echo "$body" | python3 -m json.tool 2>/dev/null || echo "$body"
    else
        echo -e "${RED}❌ 失敗 ($http_code)${NC}"
        echo "$body"
    fi
}

# 基本API テスト
test_api_call "基本ヘルスチェック" "http://localhost:8001/" "GET"
test_api_call "詳細ヘルスチェック" "http://localhost:8001/health" "GET"
test_api_call "プリセット一覧" "http://localhost:8001/presets" "GET"
test_api_call "ステータス確認" "http://localhost:8001/status" "GET"

# チャットテスト
echo -e "\n${BLUE}💬 チャット機能テスト${NC}"

# ツンデレテスト
test_api_call "ツンデレチャット" "http://localhost:8001/interactive" "POST" '{
    "message": "こんにちは！",
    "character": "tsundere",
    "temperature": 0.7,
    "max_tokens": 100
}'

# フレンドリーテスト
test_api_call "フレンドリーチャット" "http://localhost:8001/interactive" "POST" '{
    "message": "今日はいい天気ですね",
    "character": "friendly",
    "temperature": 0.8,
    "max_tokens": 100
}'

# 技術的テスト
test_api_call "技術的質問" "http://localhost:8001/interactive" "POST" '{
    "message": "Pythonの非同期処理について教えて",
    "character": "technical",
    "temperature": 0.5,
    "max_tokens": 150
}'

# 履歴クリア
test_api_call "履歴クリア" "http://localhost:8001/history" "DELETE"

# 基本チャットAPI
test_api_call "基本チャット" "http://localhost:8001/chat" "POST" '{
    "message": "短いテスト",
    "use_history": false,
    "stream": false,
    "generation_config": {
        "max_tokens": 50,
        "temperature": 0.7
    }
}'

# 結果確認
echo -e "\n${BLUE}📊 テスト結果:${NC}"
echo -e "${GREEN}✅ 全てのテストが完了しました${NC}"
echo -e "${BLUE}インタラクティブチャットを開始するには:${NC}"
echo "python3 scripts/interactive_chat_cli.py"
echo ""
echo -e "${BLUE}または、異なるキャラクターで:${NC}"
echo "python3 scripts/interactive_chat_cli.py --character friendly"
echo "python3 scripts/interactive_chat_cli.py --character technical --temperature 0.5"