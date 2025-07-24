#!/usr/bin/env python3
"""
Fast Gradio - CLIと同じ速度を目指す軽量版
"""

import gradio as gr
import logging
import asyncio
import threading
from datetime import datetime
from typing import List, Dict, Optional

# 既存のLLMエンジンを直接インポート（FastAPIと同じ）
from llm_engine import LLMEngine, GenerationConfig, create_engine

# ===== Global Variables =====
llm_engine: Optional[LLMEngine] = None

# ===== Initialization =====
def initialize_engine():
    """LLMエンジン初期化（CLIと同じ直接処理）"""
    global llm_engine
    try:
        logging.info("Initializing LLM Engine...")
        llm_engine = create_engine()
        logging.info("LLM Engine initialized successfully!")
        return True
    except Exception as e:
        logging.error(f"Failed to initialize LLM Engine: {e}")
        return False

# ===== Fast Chat Function =====
def fast_chat(message: str, history: List, temperature: float = 0.7, max_tokens: int = 256):
    """高速チャット処理（CLIと同じ直接処理）"""
    if not message.strip():
        return history, ""
    
    if llm_engine is None:
        error_msg = "❌ LLM Engine not ready"
        history.append([message, error_msg])
        return history, ""
    
    try:
        # CLIと同じ軽量設定
        config = GenerationConfig(
            max_tokens=int(max_tokens),
            temperature=float(temperature),
            top_p=0.9,
            top_k=40,
            repeat_penalty=1.1
        )
        
        # 直接推論実行（オーバーヘッドなし）
        start_time = datetime.now()
        response = llm_engine.generate_response(message, config)
        inference_time = (datetime.now() - start_time).total_seconds()
        
        # 履歴更新
        history.append([message, response])
        
        logging.info(f"Generated response in {inference_time:.2f}s")
        return history, ""
        
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        logging.error(f"Chat error: {e}")
        history.append([message, error_msg])
        return history, ""

def clear_chat():
    """履歴クリア"""
    if llm_engine:
        llm_engine.clear_conversation()
    return [], ""

# ===== Minimal Gradio Interface =====
def create_fast_interface():
    """軽量Gradio UI"""
    
    with gr.Blocks(
        title="🦙 Fast LlamaCPP Chat",
        css=".gradio-container {max-width: 900px; margin: auto;}"
    ) as demo:
        
        gr.HTML("""
        <div style="text-align: center; padding: 15px;">
            <h1>🦙 Fast LlamaCPP Chat</h1>
            <p>高速ツンデレチャット（CLI同等速度）</p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=4):
                # メインチャット
                chatbot = gr.Chatbot(
                    label="チャット",
                    height=400
                )
                
                # メッセージ入力
                with gr.Row():
                    msg_input = gr.Textbox(
                        label="メッセージ",
                        placeholder="高速チャットを試してみて...",
                        scale=4
                    )
                    send_btn = gr.Button("送信", variant="primary", scale=1)
                
                # ボタン
                with gr.Row():
                    clear_btn = gr.Button("クリア", variant="secondary")
            
            with gr.Column(scale=1):
                # 軽量設定
                gr.HTML("<h3>⚙️ 設定</h3>")
                
                temperature = gr.Slider(
                    label="Temperature",
                    minimum=0.1,
                    maximum=1.0,
                    value=0.7,
                    step=0.1
                )
                
                max_tokens = gr.Slider(
                    label="Max Tokens",
                    minimum=50,
                    maximum=512,
                    value=256,
                    step=50,
                    info="CLIと同じ256推奨"
                )
                
                gr.HTML("<p>🚀 CLI同等の高速処理</p>")
        
        # イベントハンドラー（軽量化）
        msg_input.submit(
            fn=fast_chat,
            inputs=[msg_input, chatbot, temperature, max_tokens],
            outputs=[chatbot, msg_input],
            queue=False  # キューなしで高速化
        )
        
        send_btn.click(
            fn=fast_chat,
            inputs=[msg_input, chatbot, temperature, max_tokens],
            outputs=[chatbot, msg_input],
            queue=False  # キューなしで高速化
        )
        
        clear_btn.click(
            fn=clear_chat,
            outputs=[chatbot, msg_input],
            queue=False  # キューなしで高速化
        )
    
    return demo

# ===== Main Function =====
def main():
    """メイン実行"""
    
    # ログ設定
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # LLMエンジン初期化
    print("🚀 Initializing LLM Engine...")
    success = initialize_engine()
    
    if not success:
        print("❌ Failed to initialize LLM Engine!")
        return
    
    print("✅ LLM Engine ready!")
    
    # 軽量Gradio UI起動
    demo = create_fast_interface()
    
    print("🌟 Starting Fast Gradio server...")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        debug=False,
        enable_queue=False  # キューなしで高速化
    )

if __name__ == "__main__":
    main()