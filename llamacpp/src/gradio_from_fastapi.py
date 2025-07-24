#!/usr/bin/env python3
"""
Gradio Server using existing FastAPI logic
既存のapi_server.pyの成功ロジックをGradioに移植
"""

import gradio as gr
import logging
import asyncio
from datetime import datetime
from typing import List, Dict, Optional

# 既存のFastAPIサーバーと同じインポート
from llm_engine import LLMEngine, GenerationConfig, create_engine

# ===== Global Variables (FastAPIと同じ) =====
llm_engine: Optional[LLMEngine] = None
start_time = datetime.now()

# ===== Initialization (FastAPIのstartup_eventと同じ) =====
def initialize_engine():
    """LLMエンジン初期化（FastAPIのstartup_eventと同じロジック）"""
    global llm_engine
    try:
        logging.info("Initializing LLM Engine...")
        llm_engine = create_engine()  # FastAPIと同じ関数使用
        logging.info("LLM Engine initialized successfully!")
        return True
    except Exception as e:
        logging.error(f"Failed to initialize LLM Engine: {e}")
        return False

# ===== Helper Functions (FastAPIと同じ) =====
def check_engine():
    """エンジンの可用性チェック（FastAPIと同じ）"""
    if llm_engine is None:
        return False
    return True

def create_generation_config(temperature=0.7, max_tokens=512, top_p=0.9) -> GenerationConfig:
    """生成設定を作成（FastAPIと同じロジック）"""
    return GenerationConfig(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=40,
        repeat_penalty=1.1
    )

# ===== Chat Function (FastAPIのchatエンドポイントと同じロジック) =====
def chat_with_llm(message: str, history: List, temperature: float, max_tokens: int, top_p: float):
    """チャット処理（FastAPIと同じロジック）"""
    if not message.strip():
        return history, ""
    
    if not check_engine():
        error_msg = "❌ LLM Engine not initialized"
        history.append([message, error_msg])
        return history, ""
    
    try:
        # FastAPIと同じ生成設定作成
        generation_config = create_generation_config(temperature, max_tokens, top_p)
        
        # FastAPIと同じ推論実行
        start_time = datetime.now()
        response = llm_engine.generate_response(message, generation_config)
        inference_time = (datetime.now() - start_time).total_seconds()
        
        # 履歴更新
        history.append([message, response])
        
        logging.info(f"Generated response in {inference_time:.2f}s")
        return history, ""
        
    except Exception as e:
        error_msg = f"❌ Generation error: {str(e)}"
        logging.error(f"Chat error: {e}")
        history.append([message, error_msg])
        return history, ""

def clear_conversation():
    """会話履歴クリア"""
    if llm_engine:
        llm_engine.clear_conversation()
    return [], ""

def get_health_status():
    """ヘルス状態取得（FastAPIのhealthエンドポイントと同じ）"""
    uptime = datetime.now() - start_time
    return {
        "status": "healthy" if check_engine() else "unhealthy",
        "model_loaded": check_engine(),
        "gpu_available": True,
        "uptime": str(uptime)
    }

# ===== Gradio Interface =====
def create_gradio_interface():
    """Gradio UI作成"""
    
    with gr.Blocks(
        title="🦙 LlamaCPP Gradio Chat",
        theme=gr.themes.Default()
    ) as demo:
        
        gr.HTML("""
        <div style="text-align: center; padding: 20px;">
            <h1>🦙 LlamaCPP Gradio Chat</h1>
            <p>既存FastAPIロジックを使用したGradio版</p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=3):
                # メインチャット
                chatbot = gr.Chatbot(
                    label="チャット",
                    height=450,
                    show_copy_button=True
                )
                
                # メッセージ入力
                with gr.Row():
                    msg_input = gr.Textbox(
                        label="メッセージ",
                        placeholder="何か話しかけて...",
                        scale=4
                    )
                    send_btn = gr.Button("送信", variant="primary", scale=1)
                
                # ボタン群
                with gr.Row():
                    clear_btn = gr.Button("履歴クリア", variant="secondary")
                    status_btn = gr.Button("状態確認", variant="secondary")
            
            with gr.Column(scale=1):
                # 設定パネル（FastAPIと同じパラメータ）
                gr.HTML("<h3>⚙️ 生成設定</h3>")
                
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
                    maximum=1000,
                    value=512,
                    step=50
                )
                
                top_p = gr.Slider(
                    label="Top-p",
                    minimum=0.1,
                    maximum=1.0,
                    value=0.9,
                    step=0.1
                )
                
                # システム情報
                gr.HTML("<h3>📊 システム情報</h3>")
                status_display = gr.JSON(
                    label="システム状態",
                    value=get_health_status()
                )
        
        # イベントハンドラー
        msg_input.submit(
            fn=chat_with_llm,
            inputs=[msg_input, chatbot, temperature, max_tokens, top_p],
            outputs=[chatbot, msg_input]
        )
        
        send_btn.click(
            fn=chat_with_llm,
            inputs=[msg_input, chatbot, temperature, max_tokens, top_p],
            outputs=[chatbot, msg_input]
        )
        
        clear_btn.click(
            fn=clear_conversation,
            outputs=[chatbot, msg_input]
        )
        
        status_btn.click(
            fn=get_health_status,
            outputs=[status_display]
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
    
    # LLMエンジン初期化（FastAPIと同じ）
    print("🚀 Initializing LLM Engine using FastAPI logic...")
    success = initialize_engine()
    
    if not success:
        print("❌ Failed to initialize LLM Engine!")
        return
    
    print("✅ LLM Engine initialized successfully!")
    
    # Gradio UI作成・起動
    demo = create_gradio_interface()
    
    print("🌟 Starting Gradio server...")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        debug=False
    )

if __name__ == "__main__":
    main()