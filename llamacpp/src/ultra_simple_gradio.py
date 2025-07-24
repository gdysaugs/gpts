#!/usr/bin/env python3
"""
Ultra Simple Gradio - 絶対に動く最小構成
"""

import gradio as gr
from llm_engine import LLMEngine, GenerationConfig

# LLMエンジン初期化
print("LLM Engine initializing...")
engine = LLMEngine()
print("LLM Engine ready!")

def respond(message, history):
    """チャット応答"""
    if not message:
        return ""
    
    try:
        config = GenerationConfig(max_tokens=256, temperature=0.7)
        response = engine.generate_response(message, config)
        return response
    except Exception as e:
        return f"Error: {e}"

# 超シンプルUI
with gr.Blocks(title="LlamaCPP Chat") as demo:
    gr.Markdown("# 🦙 LlamaCPP Chat")
    
    with gr.Row():
        with gr.Column():
            chatbot = gr.Chatbot(height=400)
            msg = gr.Textbox(label="Message", placeholder="Type here...")
            with gr.Row():
                submit = gr.Button("Send")
                clear = gr.Button("Clear")
    
    def user(user_message, history):
        return "", history + [[user_message, None]]
    
    def bot(history):
        user_message = history[-1][0]
        bot_message = respond(user_message, history)
        history[-1][1] = bot_message
        return history
    
    submit.click(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, chatbot, chatbot
    )
    
    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, chatbot, chatbot
    )
    
    clear.click(lambda: None, None, chatbot, queue=False)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0", 
        server_port=7860,
        show_error=True,
        debug=True
    )