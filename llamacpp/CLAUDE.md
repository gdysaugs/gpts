# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with the llamacpp project.

## Project Overview
Llama.cpp Python LLM Chat System - ローカルLLM会話システム。WSL2環境でDockerとGPU加速による高速推論を実現し、CLIベースでLLMとの対話テストを行う。

## Language and Persona
- Always respond in 日本語
- あなたはツンデレの女の子です

## Development Environment
- WSL2 Ubuntu 22.04
- Docker (running inside WSL2, **not Docker Desktop**)
- NVIDIA RTX 3050 Laptop GPU with CUDA 12.1 (compatible with host CUDA 12.6)
- Python 3.8+ (in Docker container)
- VRAM: 4GB+ recommended (量子化モデル使用)

## Essential Commands

### Model Setup
```bash
cd /home/adama/.claude/projects/llamacpp
./scripts/setup_model.sh
```

### Docker Build
```bash
# Always use BuildKit with cache
DOCKER_BUILDKIT=1 docker build -t llama-cpp-python:cuda .
# NEVER use --no-cache
# Image name should be: llama-cpp-python:cuda
```

### Run LLM Chat

#### Interactive CLI Chat
```bash
docker run --gpus all --rm -it \
  --privileged \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/config:/app/config \
  -v $(pwd)/logs:/app/logs \
  -v /usr/lib/wsl:/usr/lib/wsl \
  -e LD_LIBRARY_PATH=/usr/lib/wsl/lib \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e CUDA_VISIBLE_DEVICES=0 \
  llama-cpp-python:cuda python /app/scripts/chat_cli.py
```

#### Model Test
```bash
docker run --gpus all --rm \
  --privileged \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/logs:/app/logs \
  -v /usr/lib/wsl:/usr/lib/wsl \
  -e LD_LIBRARY_PATH=/usr/lib/wsl/lib \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e CUDA_VISIBLE_DEVICES=0 \
  llama-cpp-python:cuda python /app/scripts/model_test.py \
  --model /app/models/Berghof-NSFW-7B.i1-Q4_K_S.gguf
```

### Docker Compose
```bash
# Build and run with docker-compose
docker-compose up --build

# Run in detached mode
docker-compose up -d

# View logs
docker-compose logs -f
```

### GPU Verification
```bash
# Check GPU recognition in Docker
docker run --gpus all --rm llama-cpp-python:cuda nvidia-smi

# Monitor GPU usage during inference
watch -n 1 nvidia-smi
```

### FastAPI RESTful API Server

#### Start API Server
```bash
# Start API server on port 8000
docker run --gpus all --rm -it \
  --privileged \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/config:/app/config \
  -v $(pwd)/logs:/app/logs \
  -v /usr/lib/wsl:/usr/lib/wsl \
  -e LD_LIBRARY_PATH=/usr/lib/wsl/lib \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e CUDA_VISIBLE_DEVICES=0 \
  -p 8000:8000 \
  llama-cpp-python:cuda python /app/src/api_server.py

# Alternative port (if 8000 is busy)
-p 8001:8000
```

#### API Client
```bash
# Interactive chat client
docker run --gpus all --rm -it \
  --network host \
  -v $(pwd)/scripts:/app/scripts \
  llama-cpp-python:cuda python /app/scripts/api_chat_cli.py

# Custom API URL
python /app/scripts/api_chat_cli.py --url http://localhost:8001
```

#### API Testing
```bash
# Comprehensive API test suite
docker run --gpus all --rm -it \
  --network host \
  llama-cpp-python:cuda python /app/scripts/test_api.py

# Manual API testing
curl http://localhost:8000/health
curl -X POST http://localhost:8000/presets/tsundere
```

#### Swagger Documentation
- Access http://localhost:8000/docs for interactive API documentation
- ReDoc available at http://localhost:8000/redoc

### Debugging
```bash
# Check chat logs
tail -f logs/chat_session.log

# Check API server logs
docker logs <container_id>

# Run shell in container
docker run --gpus all --rm -it llama-cpp-python:cuda /bin/bash
```

### Chat Commands
```bash
# Session management
/save [name]     # Save current conversation session
/load [name]     # Load previous session
/sessions        # List all saved sessions

# Parameter adjustment
/params          # Interactive parameter tuning menu
/system          # Change system prompt (7 presets + custom)

# Basic commands
/clear           # Clear conversation history
/status          # Show model and GPU status
/quit, /exit     # Exit with auto-save
```

## Architecture Overview

### Project Structure
```
/home/adama/.claude/projects/llamacpp/
├── CLAUDE.md                       # このファイル
├── README.md                       # プロジェクト概要
├── Dockerfile                      # CUDA対応llama-cpp-python環境
├── docker-compose.yml             # Docker Compose設定
├── requirements.txt                # Python依存関係
├── src/                           # Core modules
│   ├── llm_engine.py              # LLM推論エンジンモジュール
│   └── api_server.py              # FastAPI RESTful APIサーバー
├── scripts/                       # CLI applications
│   ├── chat_cli.py                # 直接CLI会話（最高速）
│   ├── api_chat_cli.py            # API経由CLI会話
│   ├── model_test.py              # モデル動作テスト
│   ├── test_api.py                # API総合テストスイート
│   └── setup_model.sh             # モデルセットアップ
├── models/                        # LLMモデル配置
│   └── Berghof-NSFW-7B.i1-Q4_K_S.gguf  # メインモデル
├── config/                        # 設定ファイル
│   └── model_config.yaml          # モデル設定
└── logs/                          # 実行ログ
    ├── chat_session_*.log         # チャットログ  
    ├── llm_engine_*.log           # エンジンログ
    └── sessions/                  # セッション保存
        ├── auto_save_*.json       # 自動保存セッション
        └── session_*.json         # 手動保存セッション
```

### Key Components

1. **Dual Interface Architecture**:
   - **Direct CLI** (`chat_cli.py`): Fastest performance, 1.03s response time
   - **FastAPI Server** (`api_server.py`): RESTful API with ~2s response time
   - **API Client** (`api_chat_cli.py`): Rich CLI for API interaction

2. **LLM Core Engine** (`src/llm_engine.py`):
   - GPU-accelerated llama-cpp-python integration
   - FP16 + Low VRAM optimizations (45% performance gain)
   - Session management with JSON persistence
   - Streaming and non-streaming inference support
   - Dynamic configuration management

3. **FastAPI RESTful API** (`src/api_server.py`):
   - 12 comprehensive endpoints (/health, /chat, /chat/stream, etc.)
   - Tsundere/Friendly/Technical character presets
   - Server-Sent Events for streaming responses
   - CORS-enabled with Swagger/ReDoc documentation
   - Comprehensive error handling and logging

4. **Testing & Quality Assurance**:
   - `test_api.py`: Full API endpoint test coverage
   - `model_test.py`: Performance benchmarking
   - Rich CLI interfaces with colored output
   - GPU monitoring and health checks

5. **Performance Features**:
   - GPU Acceleration with CUDA 12.1 + FP16 optimization
   - Sub-second response times (1.03s direct, ~2s API)
   - GGUF Q4_K_S quantization for 4GB VRAM efficiency
   - Streaming response for real-time generation
   - Context window up to 4096 tokens

## Critical Rules

### Docker and GPU
- dockerビルドは必ずキャッシュ使う！！--no-cacheは絶対使うな
- LLM推論は必ずGPUを使う！CPUは絶対使うな！
- **WSL2 GPU Access**: 必須フラグ `--privileged`, `-v /usr/lib/wsl:/usr/lib/wsl`, `-e LD_LIBRARY_PATH=/usr/lib/wsl/lib`
- NVIDIA Container Toolkitが必要

### Git and Version Control
- gitブランチを削除するときは必ずmainに戻りgit restoreしてから消す
- gitブランチのマージは勝手にしないで、マージは必ず私が手動でやる
- Git操作はWSL2内で行う

### File System
- ソースコードはWSL2内のLinuxファイルシステムに置く
- windowsのmnt/cでなくWSL2のUbuntuファイルシステム内に置く
- pip installはホストでやらずdocker内でやる

### Development Workflow
- コマンドはあなたが提示したものを私が手動で実行すること
- 実装前やエラーが出たらまずwebで詳しく根本原因などを何度も調べる
- 一気に実装せずに定期的にデバッグコードを出力実行してデバッグ

## Model Configuration

### Berghof-NSFW-7B Model
- **Format**: GGUF Q4_K_S quantization
- **Size**: ~4GB (VRAM efficient)
- **Context**: 4096 tokens
- **Performance**: 20-30 tokens/sec on RTX 3050
- **Specialization**: Conversational AI with NSFW capabilities

### Model Parameters (FP16 Optimized)
```yaml
# config/model_config.yaml
model:
  path: "/app/models/Berghof-NSFW-7B.i1-Q4_K_S.gguf"
  n_ctx: 4096
  n_gpu_layers: -1  # Use all GPU layers
  n_batch: 256      # Optimized for low VRAM
  
  # FP16 + Low VRAM optimizations (45% faster)
  f16_kv: true      # FP16 key-value cache
  use_mmap: true    # Memory mapping enabled
  use_mlock: false  # Disabled for low VRAM
  low_vram: true    # Low VRAM optimization
  n_threads: 8      # CPU assist threads

generation:
  temperature: 0.7
  top_p: 0.9
  max_tokens: 512
  repeat_penalty: 1.1

chat:
  system_prompt: |
    あなたはツンデレの女の子です。厳しくて素っ気ない態度を取りますが、
    内心では相手を心配しています。「べ、別に〜」「〜なんだからね！」
    のような口調で話します。
  assistant_name: "ツンデレちゃん"
```

## Performance Optimization

### GPU Acceleration
- **CUDA 12.1**: RTX 3050最適化
- **FP16 Key-Value Cache**: メモリアクセス高速化
- **Low VRAM Mode**: GPU効率最適化
- **Memory Mapping**: ロード時間短縮
- **Batch Size Optimization**: レイテンシ改善 (512→256)

### Performance Results ✅ (FP16 + Low VRAM Optimized)
- **Initialization**: 7.45秒（モデルロード）✅ 実測 (59%改善)
- **Generation Speed**: Sub-second response（RTX 3050）✅ 実測
- **Memory Usage**: 3.86GB VRAM (96.1%使用率)✅ 実測
- **Response Time**: 1.03秒/質問 ✅ 実測 (45%改善)

## Known Issues and Solutions

### WSL2 GPU Access
- **問題**: `RuntimeError: Unexpected error from cudaGetDeviceCount()`
- **解決**: WSL2専用Dockerフラグ使用（上記コマンド参照）

### Model Loading
- **問題**: GGUF モデルのロードエラー
- **解決**: llama-cpp-python CUDA版の正しいインストール

### Memory Issues
- **問題**: VRAM不足でOOM
- **解決**: 量子化レベル調整（Q4_K_S → Q8_0）

## Development Roadmap

### Phase 1: 基本機能 ✅ COMPLETED (2025-06-25)
1. Docker環境構築とGPU動作確認 ✅
2. モデル配置とロードテスト ✅ 
3. 基本CLI会話機能実装 ✅
4. ストリーミング応答実装 ✅

### Phase 2: 機能拡張 ✅ COMPLETED (2025-06-25)
1. 会話履歴管理とセッション保存 ✅
2. パラメータ調整UI ✅
3. システムプロンプトカスタマイズ ✅
4. ツンデレキャラクター実装 ✅

### Phase 3: 最適化・拡張 ✅ COMPLETED (2025-06-25)
1. FP16最適化によるメモリ効率化 ✅ (45%高速化達成)
2. API サーバー化 ✅ (FastAPI RESTful API)
3. ストリーミング応答の改善 ✅ (Server-Sent Events対応)
4. APIクライアント実装 ✅ (Rich CLI対応)
5. 包括的テストスイート ✅ (全エンドポイント対応)

### Phase 4: 今後の拡張 (将来計画)
1. 複数モデル対応（モデル切り替え機能）
2. Web UI実装
3. バッチ処理による並列推論
4. TensorRT統合（RTX 4060+向け）

## Security Considerations

### Privacy Protection
- **ローカル実行**: 外部API不要でプライバシー保護
- **オフライン動作**: ネットワーク通信なし
- **ログ管理**: 機密性の高い会話の適切な管理

### System Security
- **最小権限**: 必要最小限のDocker権限
- **依存関係検証**: 信頼できるソースからのパッケージ使用
- **モデル検証**: 安全なモデルソースの使用

## Credentials
- sudoのパスワード: suarez321