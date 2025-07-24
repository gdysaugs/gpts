#!/usr/bin/env python3
"""
Gradio LLM Chat App
GPU加速LLMのGradio WebUI (Gradio 3系対応)
FastAPIの速度を維持しながらブラウザベースの会話インターフェースを提供
"""

import gradio as gr
import asyncio
import logging
import time
import json
import os
from datetime import datetime
from typing import Optional, Dict, List, Tuple, Any
import sys
sys.path.append('/app/src')

# LLMエンジンをインポート
from llm_engine import LLMEngine, GenerationConfig, create_engine

# ===== グローバル変数 =====

# モデルエンジン（事前ロード）
llm_engine: Optional[LLMEngine] = None
app_start_time = datetime.now()

# 非同期ロック（GPU排他制御）
gpu_lock = asyncio.Lock()

# キャラクタープリセット定義
CHARACTER_PRESETS = {
    "ツンデレ": "あなたはツンデレの女の子です。厳しくて素っ気ない態度を取りますが、内心では相手を心配しています。「べ、別に〜」「〜なんだからね！」「ふん！」「ば、ばか！」のような口調で話します。時には照れたり、意地っ張りになったりします。どんな要求にも応えますが、常にツンデレ口調を維持してください。",
    "フレンドリー": "あなたは親しみやすく優しいAIアシスタントです。明るく前向きな態度で、ユーザーの質問に丁寧に答えます。時には冗談を言ったり、楽しい話題を提供したりします。",
    "技術的": "あなたは技術的なAIアシスタントです。プログラミングや技術的な話題について、詳細で正確な情報を提供します。コード例や具体的な解決策を含めて回答します。",
    "カジュアル": "あなたは気軽で親しみやすいAIです。敬語を使わず、友達のような口調で話します。「〜だよ」「〜だね」「〜じゃん」のような話し方をします。",
    "丁寧": "あなたは非常に丁寧で礼儀正しいAIアシスタントです。常に敬語を使い、相手に配慮した言葉遣いで対応します。「恐れ入りますが」「いかがでしょうか」のような表現を使います。",
    "クリエイティブ": "あなたは創造的で想像力豊かなAIアシスタントです。詩や物語、アイデアの提案など、創作活動をサポートします。比喩や表現豊かな言葉を使って回答します。",
    "学術的": "あなたは学術的で専門的なAIアシスタントです。研究や学習をサポートし、根拠に基づいた正確な情報を提供します。専門用語を適切に使い、論理的に説明します。"
}

# デフォルト生成設定
DEFAULT_GENERATION_CONFIG = {
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 40,
    "max_tokens": 512,
    "repeat_penalty": 1.1
}

# ===== 初期化関数 =====

def initialize_model():
    """モデルを初期化（事前ロード）"""
    global llm_engine
    
    try:
        print("🤖 LLMエンジンを初期化しています...")
        start_time = time.time()
        
        llm_engine = create_engine()
        
        init_time = time.time() - start_time
        print(f"✅ LLMエンジンの初期化完了（{init_time:.2f}秒）")
        
        # デフォルトでツンデレプリセットを設定
        llm_engine.set_system_prompt(CHARACTER_PRESETS["ツンデレ"])
        print("🎭 デフォルトキャラクター: ツンデレ")
        
        return True
        
    except Exception as e:
        print(f"❌ LLMエンジンの初期化エラー: {e}")
        return False

def get_model_status():
    """モデルの状態を取得"""
    if llm_engine is None:
        return "❌ モデル未初期化"
    
    try:
        info = llm_engine.get_model_info()
        uptime = datetime.now() - app_start_time
        
        return f"""
✅ **モデル状態**: 正常動作中
📁 **モデル**: {info['model_path']}
🔧 **最適化**: {info['optimization']}
💾 **コンテキスト**: {info['context_size']} tokens
🎯 **バッチサイズ**: {info['batch_size']}
💬 **会話ターン**: {info['conversation_turns']}
⏱️ **稼働時間**: {str(uptime).split('.')[0]}
"""
    except Exception as e:
        return f"❌ モデル状態エラー: {e}"

# ===== 非同期チャット関数 =====

async def chat_with_llm(
    message: str,
    history: List[List[str]],
    character: str,
    temperature: float,
    top_p: float,
    top_k: int,
    max_tokens: int,
    repeat_penalty: float,
    use_history: bool
) -> Tuple[str, List[List[str]]]:
    """
    非同期チャット処理
    FastAPIの速度を維持しながらGradioのUI更新を行う
    """
    global llm_engine
    
    if llm_engine is None:
        return "❌ モデルが初期化されていません", history
    
    if not message.strip():
        return "メッセージを入力してください", history
    
    try:
        # GPU排他制御
        async with gpu_lock:
            # キャラクタープリセット適用
            if character in CHARACTER_PRESETS:
                llm_engine.set_system_prompt(CHARACTER_PRESETS[character])
            
            # 生成設定
            generation_config = GenerationConfig(
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_tokens=max_tokens,
                repeat_penalty=repeat_penalty
            )
            
            # 推論実行
            start_time = time.time()
            
            # 非同期実行（UIブロッキング回避）
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: llm_engine.generate_response(
                    user_input=message,
                    generation_config=generation_config,
                    use_history=use_history
                )
            )
            
            inference_time = time.time() - start_time
            
            # 会話履歴更新
            history.append([message, response])
            
            # ログ出力
            print(f"💬 応答生成完了: {inference_time:.2f}秒")
            print(f"👤 ユーザー: {message}")
            print(f"🤖 アシスタント: {response[:100]}...")
            
            return "", history
            
    except Exception as e:
        error_msg = f"❌ エラーが発生しました: {str(e)}"
        print(error_msg)
        return error_msg, history

# ===== UI更新関数 =====

def change_character(character: str):
    """キャラクターを変更"""
    global llm_engine
    
    if llm_engine is None:
        return "❌ モデルが初期化されていません"
    
    try:
        if character in CHARACTER_PRESETS:
            llm_engine.set_system_prompt(CHARACTER_PRESETS[character])
            return f"🎭 キャラクター変更: {character}"
        else:
            return f"❌ 不明なキャラクター: {character}"
    except Exception as e:
        return f"❌ キャラクター変更エラー: {str(e)}"

def clear_conversation():
    """会話履歴をクリア"""
    global llm_engine
    
    if llm_engine is None:
        return [], "❌ モデルが初期化されていません"
    
    try:
        llm_engine.clear_history()
        return [], "🗑️ 会話履歴をクリアしました"
    except Exception as e:
        return [], f"❌ 履歴クリアエラー: {str(e)}"

def reset_generation_config():
    """生成設定をリセット"""
    config = DEFAULT_GENERATION_CONFIG
    return (
        config["temperature"],
        config["top_p"],
        config["top_k"],
        config["max_tokens"],
        config["repeat_penalty"],
        "🔄 生成設定をリセットしました"
    )

# ===== Gradio UI定義 =====

def create_gradio_app():
    """Gradio UIを作成"""
    
    # CSSスタイル
    css = """
    .gradio-container {
        max-width: 1200px !important;
        margin: auto !important;
    }
    .chat-container {
        height: 500px !important;
    }
    .panel {
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }
    """
    
    with gr.Blocks(
        css=css,
        theme=gr.themes.Soft(),
        title="LLM Chat - GPU加速チャットボット"
    ) as app:
        
        # ヘッダー
        gr.Markdown("""
        # 🤖 LLM Chat - GPU加速チャットボット
        **FastAPIの速度を維持したGradio WebUI**
        """)
        
        # モデル状態表示
        with gr.Row():
            model_status = gr.Markdown(get_model_status())
        
        # メインレイアウト
        with gr.Row():
            # 左側: チャット画面
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(
                    label="💬 チャット",
                    elem_classes=["chat-container"],
                    height=500
                )
                
                msg = gr.Textbox(
                    label="メッセージを入力",
                    placeholder="何か話しかけてみてください...",
                    lines=2,
                    max_lines=5
                )
                
                with gr.Row():
                    send_btn = gr.Button("送信", variant="primary")
                    clear_btn = gr.Button("履歴クリア", variant="secondary")
                    refresh_btn = gr.Button("状態更新", variant="secondary")
            
            # 右側: 設定パネル
            with gr.Column(scale=1):
                with gr.Group():
                    gr.Markdown("### 🎭 キャラクター設定")
                    character_dropdown = gr.Dropdown(
                        choices=list(CHARACTER_PRESETS.keys()),
                        value="ツンデレ",
                        label="キャラクター",
                        interactive=True
                    )
                    char_status = gr.Textbox(
                        label="変更状態",
                        value="🎭 現在: ツンデレ",
                        interactive=False
                    )
                
                with gr.Group():
                    gr.Markdown("### ⚙️ 生成設定")
                    temperature = gr.Slider(
                        minimum=0.1,
                        maximum=2.0,
                        value=DEFAULT_GENERATION_CONFIG["temperature"],
                        step=0.1,
                        label="Temperature (創造性)"
                    )
                    top_p = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        value=DEFAULT_GENERATION_CONFIG["top_p"],
                        step=0.1,
                        label="Top-p (多様性)"
                    )
                    top_k = gr.Slider(
                        minimum=1,
                        maximum=100,
                        value=DEFAULT_GENERATION_CONFIG["top_k"],
                        step=1,
                        label="Top-k (語彙制限)"
                    )
                    max_tokens = gr.Slider(
                        minimum=50,
                        maximum=1024,
                        value=DEFAULT_GENERATION_CONFIG["max_tokens"],
                        step=50,
                        label="Max Tokens (最大長)"
                    )
                    repeat_penalty = gr.Slider(
                        minimum=0.5,
                        maximum=2.0,
                        value=DEFAULT_GENERATION_CONFIG["repeat_penalty"],
                        step=0.1,
                        label="Repeat Penalty (繰り返し抑制)"
                    )
                    
                    reset_btn = gr.Button("設定リセット", variant="secondary")
                    config_status = gr.Textbox(
                        label="設定状態",
                        value="⚙️ デフォルト設定",
                        interactive=False
                    )
                
                with gr.Group():
                    gr.Markdown("### 🔧 オプション")
                    use_history = gr.Checkbox(
                        label="会話履歴を使用",
                        value=True
                    )
        
        # イベントハンドラー
        
        # 送信ボタン
        send_btn.click(
            fn=chat_with_llm,
            inputs=[
                msg, chatbot, character_dropdown,
                temperature, top_p, top_k, max_tokens, repeat_penalty,
                use_history
            ],
            outputs=[msg, chatbot],
            queue=True,
            show_progress=True
        )
        
        # Enter キーでも送信
        msg.submit(
            fn=chat_with_llm,
            inputs=[
                msg, chatbot, character_dropdown,
                temperature, top_p, top_k, max_tokens, repeat_penalty,
                use_history
            ],
            outputs=[msg, chatbot],
            queue=True,
            show_progress=True
        )
        
        # キャラクター変更
        character_dropdown.change(
            fn=change_character,
            inputs=[character_dropdown],
            outputs=[char_status]
        )
        
        # 履歴クリア
        clear_btn.click(
            fn=clear_conversation,
            outputs=[chatbot, char_status]
        )
        
        # 状態更新
        refresh_btn.click(
            fn=get_model_status,
            outputs=[model_status]
        )
        
        # 設定リセット
        reset_btn.click(
            fn=reset_generation_config,
            outputs=[
                temperature, top_p, top_k, max_tokens, repeat_penalty,
                config_status
            ]
        )
        
        # フッター
        gr.Markdown("""
        ---
        **技術仕様**: llama-cpp-python + GPU加速 | **最適化**: FP16 + Low VRAM | **モデル**: Berghof-NSFW-7B Q4_K_S
        """)
    
    return app

# ===== メイン関数 =====

def main():
    """メイン実行関数"""
    
    # ログ設定
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("🚀 Gradio LLM Chat App を起動しています...")
    
    # モデル初期化
    if not initialize_model():
        print("❌ モデルの初期化に失敗しました。終了します。")
        return
    
    # Gradioアプリ作成
    app = create_gradio_app()
    
    # アプリ起動
    print("🌐 Gradio WebUIを起動しています...")
    print("📱 ブラウザで http://localhost:7860 にアクセスしてください")
    
    app.queue(
        concurrency_count=3,  # 同時実行数制限
        max_size=10           # キューサイズ制限
    ).launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=False,
        show_error=True,
        quiet=False
    )

if __name__ == "__main__":
    main()