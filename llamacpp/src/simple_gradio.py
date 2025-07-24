#!/usr/bin/env python3
"""
Simple Gradio UI for LlamaCPP
最小限のシンプルUI（ロケールエラー回避版）
"""

import os
import gradio as gr
import logging
from datetime import datetime

# 既存のLLMEngineをインポート
from llm_engine import LLMEngine, GenerationConfig

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# グローバルLLMエンジン
global_engine = None

def initialize_engine():
    """LLMエンジン初期化"""
    global global_engine
    if global_engine is None:
        logger.info("Initializing LLM Engine...")
        global_engine = LLMEngine()
        logger.info("LLM Engine ready!")
    return global_engine

def chat_simple(message, history, temperature, max_tokens):
    """シンプルチャット関数"""
    if not message.strip():
        return history, ""
    
    try:
        engine = initialize_engine()
        config = GenerationConfig(
            max_tokens=int(max_tokens),
            temperature=float(temperature)
        )
        
        start_time = datetime.now()
        response = engine.generate_response(message, config)
        inference_time = (datetime.now() - start_time).total_seconds()
        
        # 履歴更新（シンプル形式）
        history = history or []
        history.append([message, response])
        
        logger.info(f"Response generated in {inference_time:.2f}s")
        return history, ""
        
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        logger.error(f"Chat error: {e}")
        history = history or []
        history.append([message, error_msg])
        return history, ""

def clear_chat():
    """チャットクリア"""
    global global_engine
    if global_engine:
        global_engine.clear_conversation()
    return [], ""

# シンプルインターフェース作成
def create_simple_interface():
    """最小限のGradio UI"""
    
    with gr.Blocks(title="LlamaCPP Simple Chat") as demo:
        
        gr.HTML("""
        <div style="text-align: center; padding: 20px;">
            <h1>🦙 LlamaCPP Simple Chat</h1>
            <p>シンプルなツンデレチャット</p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=3):
                # メインチャット（従来形式）
                chatbot = gr.Chatbot(
                    label="チャット", 
                    height=400
                )
                
                # メッセージ入力
                msg = gr.Textbox(
                    label="メッセージ", 
                    placeholder="何か話しかけて...",
                    lines=2
                )
                
                with gr.Row():
                    send_btn = gr.Button("送信", variant="primary")
                    clear_btn = gr.Button("クリア", variant="secondary")
            
            with gr.Column(scale=1):
                # 設定
                gr.HTML("<h3>設定</h3>")
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
                    step=50
                )
        
        # イベント設定
        msg.submit(
            fn=chat_simple,
            inputs=[msg, chatbot, temperature, max_tokens],
            outputs=[chatbot, msg]
        )
        
        send_btn.click(
            fn=chat_simple,
            inputs=[msg, chatbot, temperature, max_tokens],
            outputs=[chatbot, msg]
        )
        
        clear_btn.click(
            fn=clear_chat,
            outputs=[chatbot, msg]
        )
    
    return demo

def main():
    """メイン実行"""
    try:
        demo = create_simple_interface()
        
        # サーバー起動
        logger.info("Starting Simple Gradio server...")
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            debug=False,
            show_error=True
        )
        
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise

if __name__ == "__main__":
    main()