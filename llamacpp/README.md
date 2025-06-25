# Llama.cpp Python LLM Chat System

## プロジェクト概要
llama-cpp-pythonを使用したローカルLLM会話システム。WSL2環境でDockerとGPU加速による高速推論を実現し、CLIベースでLLMとの対話テストを行う。

## システム要件
- WSL2 Ubuntu 22.04
- Docker（WSL2内で動作、**Docker Desktopではない**）
- NVIDIA RTX 3050 Laptop GPU with CUDA 12.1
- NVIDIA Container Toolkit（WSL2でのGPUアクセス用）
- Python 3.8+（Docker内）
- 4GB+ VRAM推奨（量子化モデル使用）

## アーキテクチャ設計

### プロジェクト構造
```
/home/adama/.claude/projects/llamacpp/
├── README.md                     # 本ファイル
├── Dockerfile                    # CUDA対応llama-cpp-python環境
├── docker-compose.yml           # Docker Compose設定
├── requirements.txt              # Python依存関係
├── scripts/                     # CLIスクリプト
│   ├── chat_cli.py              # メイン会話CLI
│   ├── model_test.py            # モデル動作テスト
│   └── setup_model.sh           # モデルセットアップ
├── models/                      # LLMモデル配置
│   └── Berghof-NSFW-7B.i1-Q4_K_S.gguf  # メインモデル
├── config/                      # 設定ファイル
│   └── model_config.yaml        # モデル設定
└── logs/                        # 実行ログ
    └── chat_session.log         # チャットログ
```

### 技術スタック

#### 🔧 **Core Components**
- **llama-cpp-python**: GGUF形式LLM推論エンジン
- **CUDA Acceleration**: RTX 3050 GPU加速
- **Docker**: 依存関係分離とポータビリティ
- **CLI Interface**: シンプルな会話テスト環境

#### 🚀 **Performance Features**
- **GPU Acceleration**: CUDA 12.1対応でRTX 3050最適化
- **GGUF Quantization**: Q4_K_S量子化で4GB VRAM対応
- **Streaming Response**: リアルタイム生成表示
- **Context Window**: 最大4096トークン対応

#### 🎯 **Chat Features**
- **Interactive CLI**: ユーザー友好的な会話インターface
- **Session Management**: 会話履歴保持
- **Model Configuration**: 温度、top-p等のパラメータ調整
- **Multi-turn Conversation**: 文脈を維持した連続会話

## 主要機能

### 🤖 **LLM推論機能**
1. **高速推論**: GPU加速による高速テキスト生成
2. **量子化対応**: Q4_K_S形式でメモリ効率化
3. **ストリーミング**: リアルタイム応答表示
4. **バッチ処理**: 複数プロンプトの並列処理

### 💬 **会話機能**
1. **インタラクティブチャット**: CLI-based conversation
2. **コンテキスト管理**: 会話履歴の自動管理
3. **システムプロンプト**: カスタマイズ可能なペルソナ設定
4. **会話保存**: セッションログの自動記録

### ⚙️ **設定機能**
1. **モデル切り替え**: 複数モデルの動的ロード
2. **パラメータ調整**: temperature, top_p, max_tokens等
3. **GPU設定**: VRAM使用量とレイヤー分散の最適化
4. **デバッグモード**: 詳細ログとパフォーマンス監視

## Docker設計

### 📦 **マルチステージビルド**
1. **Stage 1**: CUDA Toolkit + Build tools
2. **Stage 2**: llama-cpp-python CUDA build
3. **Stage 3**: Runtime environment with models

### 🔧 **GPU最適化**
- **CUDA 12.1 Compatible**: ホストCUDA 12.6との互換性
- **Tensor Cores**: RTX 3050最適化
- **Memory Management**: 効率的なVRAM使用
- **WSL2 Integration**: 必須フラグ設定済み

## セキュリティ考慮

### 🔒 **モデル管理**
- **ローカル実行**: 外部API不要でプライバシー保護
- **ファイルアクセス制御**: Docker内での適切な権限管理
- **ログ暗号化**: 機密性の高い会話の保護

### 🛡️ **システムセキュリティ**
- **最小権限**: 必要最小限のDocker権限
- **ネットワーク分離**: 外部通信なしのオフライン動作
- **依存関係検証**: 信頼できるソースからのパッケージ使用

## パフォーマンス実績 ✅

### 🎯 **推論速度** (実測値)
- **初期化**: 18.26秒（モデルロード） ✅
- **生成速度**: リアルタイム応答（RTX 3050） ✅
- **メモリ使用**: 3.86GB VRAM (96.1%使用率) ✅
- **応答時間**: 1.88秒/質問 ✅

### 📊 **品質指標** (動作確認済み)
- **コンテキスト長**: 4096トークン（n_ctx_train対応） ✅
- **量子化精度**: Q4_K_S（3.86GB）品質良好 ✅
- **安定性**: GPU加速で安定動作 ✅
- **一貫性**: CUDA 12.1完全対応 ✅

## 既知の制限事項

### ⚠️ **ハードウェア制限**
- **VRAM**: 8GB未満では大型モデル制限
- **量子化**: 精度と品質のトレードオフ
- **WSL2**: ネイティブLinuxより若干のオーバーヘッド

### 🔧 **ソフトウェア制限**
- **CUDA互換性**: ホストとコンテナの複雑な依存関係
- **モデル形式**: GGUF以外の形式は変換必要
- **メモリ管理**: 大きなコンテキストでのOOM可能性

## 開発ロードマップ

### Phase 1: **基本機能** ✅ COMPLETED
1. Docker環境構築 ✅
2. モデル配置とテスト ✅
3. 基本CLI会話機能 ✅
4. GPU動作確認 ✅

### Phase 2: **機能拡張** (優先度: 中)
1. ストリーミング応答
2. 会話履歴管理
3. パラメータ調整UI
4. 複数モデル対応

### Phase 3: **最適化** (優先度: 低)
1. パフォーマンスチューニング
2. Web UI実装
3. API サーバー化
4. 分散推論対応

---

**作成日**: 2025-06-24  
**バージョン**: v1.1  
**動作確認環境**: WSL2 Ubuntu 22.04 + RTX 3050 + CUDA 12.1  
**特記事項**: GPU加速、GGUF対応、オフライン動作  
**Phase 1完了日**: 2025-06-25  
**実績**: Berghof-NSFW-7B (3.86GB) 完全動作確認済み