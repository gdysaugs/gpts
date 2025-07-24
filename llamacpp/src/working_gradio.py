#!/usr/bin/env python3
"""
Working Gradio - 確実に動作する版
LLMエンジンを遅延初期化してサーバーを先に起動
"""

import gradio as gr
import threading
import time
import logging

# グローバル変数
engine = None
engine_ready = False

def init_engine_async():
    """バックグラウンドでLLMエンジンを初期化"""
    global engine, engine_ready
    try:
        print("🚀 Starting LLM Engine initialization in background...")
        from llm_engine import LLMEngine, GenerationConfig
        engine = LLMEngine()
        engine_ready = True
        print("✅ LLM Engine ready!")
    except Exception as e:
        print(f"❌ LLM Engine failed: {e}")
        engine_ready = False

def chat_response(message, history):
    """チャット応答"""
    if not message.strip():
        return history, ""
    
    if not engine_ready:
        response = "⏳ LLMエンジンを初期化中です...しばらくお待ちください"
        history.append([message, response])
        return history, ""
    
    if engine is None:
        response = "❌ LLMエンジンの初期化に失敗しました"
        history.append([message, response])
        return history, ""
    
    try:
        from llm_engine import GenerationConfig
        config = GenerationConfig(max_tokens=256, temperature=0.7)
        response = engine.generate_response(message, config)
        history.append([message, response])
        return history, ""
    except Exception as e:
        response = f"❌ エラー: {str(e)}"
        history.append([message, response])
        return history, ""

def get_status():
    """システム状態取得"""
    if engine_ready and engine:
        return "✅ LLMエンジン準備完了"
    elif engine_ready == False and engine is None:
        return "❌ LLMエンジン初期化失敗"
    else:
        return "⏳ LLMエンジン初期化中..."

# バックグラウンドでLLMエンジン初期化開始
init_thread = threading.Thread(target=init_engine_async, daemon=True)
init_thread.start()

# Gradio UI作成
with gr.Blocks(
    title="🦙 LlamaCPP Gradio Chat",
    css=".gradio-container {max-width: 1000px; margin: auto;}"
) as demo:
    
    gr.HTML("""
    <div style="text-align: center; padding: 20px;">
        <h1>🦙 LlamaCPP Gradio Chat</h1>
        <p>ツンデレキャラとの会話</p>
    </div>
    """)
    
    with gr.Row():
        with gr.Column(scale=3):
            # チャットボット
            chatbot = gr.Chatbot(
                label="チャット",
                height=450,
                show_copy_button=True
            )
            
            # メッセージ入力
            with gr.Row():
                msg_input = gr.Textbox(
                    label="メッセージ",
                    placeholder="何か話しかけてみて...",
                    scale=4
                )
                send_btn = gr.Button("送信", variant="primary", scale=1)
            
            # ボタン群
            with gr.Row():
                clear_btn = gr.Button("クリア", variant="secondary")
                status_btn = gr.Button("状態確認", variant="secondary")
        
        with gr.Column(scale=1):
            # システム情報
            gr.HTML("<h3>📊 システム状態</h3>")
            status_display = gr.Textbox(
                label="LLM状態",
                value="⏳ 初期化中...",
                interactive=False,
                lines=3
            )
            
            # 設定
            gr.HTML("<h3>⚙️ 設定</h3>")
            gr.HTML("<p>Temperature: 0.7<br>Max Tokens: 256</p>")
    
    # イベントハンドラー
    msg_input.submit(
        fn=chat_response,
        inputs=[msg_input, chatbot],
        outputs=[chatbot, msg_input]
    )
    
    send_btn.click(
        fn=chat_response,
        inputs=[msg_input, chatbot],
        outputs=[chatbot, msg_input]
    )
    
    clear_btn.click(
        fn=lambda: ([], ""),
        outputs=[chatbot, msg_input]
    )
    
    status_btn.click(
        fn=get_status,
        outputs=[status_display]
    )

if __name__ == "__main__":
    print("🌟 Starting Gradio server...")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        debug=False,
        quiet=False
    )