#!/usr/bin/env python3
"""
Minimal Gradio UI - 最小限で確実に動作する版
"""

import gradio as gr
import logging
from datetime import datetime
from llm_engine import LLMEngine, GenerationConfig

# グローバル変数
engine = None

def init_engine():
    global engine
    if engine is None:
        print("Loading LLM...")
        engine = LLMEngine()
        print("LLM Ready!")
    return engine

def chat(message, history):
    """最小限のチャット関数"""
    if not message:
        return history, ""
    
    try:
        llm = init_engine()
        config = GenerationConfig(max_tokens=256, temperature=0.7)
        response = llm.generate_response(message, config)
        history.append((message, response))
        return history, ""
    except Exception as e:
        history.append((message, f"Error: {str(e)}"))
        return history, ""

# 最小限UI
demo = gr.Interface(
    fn=lambda message, history: chat(message, history or []),
    inputs=[
        gr.Textbox(placeholder="メッセージを入力..."),
        gr.State([])
    ],
    outputs=[
        gr.Chatbot(),
        gr.Textbox()
    ],
    title="🦙 LlamaCPP Chat",
    description="シンプルなツンデレチャット"
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)