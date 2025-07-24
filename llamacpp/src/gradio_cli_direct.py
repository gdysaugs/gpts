#!/usr/bin/env python3
"""
Gradio with CLI Direct Processing
CLIの直接処理ロジックをGradioに移植（1.88秒目標）
"""

import gradio as gr
import yaml
import os
import logging
from datetime import datetime
from typing import List, Dict, Optional
from llama_cpp import Llama

# ===== Global Variables =====
llm_model = None
config = None
conversation_history = []

# ===== CLI Direct Logic =====
def load_config(config_path: str = "/app/config/model_config.yaml") -> Dict:
    """CLIと同じ設定ロード"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Config error: {e}")
        return {
            "model": {
                "path": "/app/models/Berghof-NSFW-7B.i1-Q4_K_S.gguf",
                "n_gpu_layers": -1,
                "n_ctx": 4096,
                "n_batch": 256,
                "verbose": False,
                "f16_kv": True,
                "use_mmap": True,
                "use_mlock": False,
                "low_vram": True,
                "n_threads": 8
            },
            "generation": {
                "max_tokens": 256,
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 40,
                "repeat_penalty": 1.1
            },
            "chat": {
                "system_prompt": "あなたはツンデレの女の子です。厳しくて素っ気ない態度を取りますが、内心では相手を心配しています。「べ、別に〜」「〜なんだからね！」のような口調で話します。",
                "user_name": "User",
                "assistant_name": "ツンデレちゃん"
            }
        }

def initialize_model():
    """CLIと同じ直接初期化"""
    global llm_model, config
    
    print("🚀 Initializing model using CLI logic...")
    config = load_config()
    model_config = config["model"]
    
    try:
        llm_model = Llama(
            model_path=model_config["path"],
            n_gpu_layers=model_config.get("n_gpu_layers", -1),
            n_ctx=model_config.get("n_ctx", 4096),
            n_batch=model_config.get("n_batch", 256),
            verbose=model_config.get("verbose", False),
            # CLIと同じFP16最適化
            f16_kv=model_config.get("f16_kv", True),
            use_mmap=model_config.get("use_mmap", True),
            use_mlock=model_config.get("use_mlock", False),
            low_vram=model_config.get("low_vram", True),
            n_threads=model_config.get("n_threads", 8)
        )
        print("✅ Model initialized successfully!")
        return True
    except Exception as e:
        print(f"❌ Model initialization failed: {e}")
        return False

def build_prompt(user_input: str) -> str:
    """CLIと同じプロンプト構築"""
    global conversation_history, config
    
    chat_config = config["chat"]
    system_prompt = chat_config["system_prompt"]
    
    # CLIと同じロジック
    prompt_parts = [system_prompt, "\n\n"]
    
    # 最近の会話履歴（最大10メッセージ）
    recent_history = conversation_history[-10:]
    for turn in recent_history:
        prompt_parts.append(f"User: {turn['user']}\n")
        prompt_parts.append(f"Assistant: {turn['assistant']}\n\n")
    
    prompt_parts.append(f"User: {user_input}\n")
    prompt_parts.append("Assistant: ")
    
    return "".join(prompt_parts)

def generate_response_direct(user_input: str, temperature: float = 0.7, max_tokens: int = 512) -> str:
    """CLIと同じ直接生成処理"""
    global llm_model, config, conversation_history
    
    if llm_model is None:
        return "❌ Model not initialized"
    
    try:
        # CLIと同じプロンプト構築
        prompt = build_prompt(user_input)
        
        # CLIと同じ直接呼び出し
        start_time = datetime.now()
        response = llm_model(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=0.9,
            top_k=40,
            repeat_penalty=1.1,
            stop=["User:", "Human:", "\n\n"]
        )
        inference_time = (datetime.now() - start_time).total_seconds()
        
        # 応答テキスト抽出
        response_text = response["choices"][0]["text"].strip()
        
        # 会話履歴に追加（CLIと同じ）
        conversation_history.append({
            "user": user_input,
            "assistant": response_text,
            "timestamp": datetime.now().isoformat()
        })
        
        print(f"✅ Generated response in {inference_time:.2f}s")
        return response_text
        
    except Exception as e:
        error_msg = f"❌ Generation error: {str(e)}"
        print(error_msg)
        return error_msg

# ===== Gradio Interface =====
def chat_interface(message: str, history: List, temperature: float, max_tokens: int):
    """Gradio用チャットインターフェース"""
    if not message.strip():
        return history, ""
    
    # CLIの直接処理を使用
    response = generate_response_direct(message, temperature, max_tokens)
    
    # Gradio履歴更新
    history.append([message, response])
    return history, ""

def clear_conversation():
    """会話履歴クリア"""
    global conversation_history
    conversation_history = []
    return [], ""

# ===== Gradio UI =====
def create_interface():
    """CLI同等速度のGradio UI"""
    
    with gr.Blocks(
        title="🦙 CLI-Direct Gradio Chat",
        theme=gr.themes.Default()
    ) as demo:
        
        gr.HTML("""
        <div style="text-align: center; padding: 15px;">
            <h1>🦙 CLI-Direct Gradio Chat</h1>
            <p>CLIの直接処理で1.88秒目標</p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="チャット",
                    height=400
                )
                
                with gr.Row():
                    msg_input = gr.Textbox(
                        label="メッセージ",
                        placeholder="CLIと同じ速度でチャット...",
                        scale=4
                    )
                    send_btn = gr.Button("送信", variant="primary", scale=1)
                
                with gr.Row():
                    clear_btn = gr.Button("クリア", variant="secondary")
            
            with gr.Column(scale=1):
                gr.HTML("<h3>⚙️ CLI設定</h3>")
                
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
                    info="CLIと同じ256"
                )
                
                gr.HTML("<p>🚀 CLIダイレクト処理</p>")
        
        # イベント（キューなし高速化）
        msg_input.submit(
            fn=chat_interface,
            inputs=[msg_input, chatbot, temperature, max_tokens],
            outputs=[chatbot, msg_input],
            queue=False
        )
        
        send_btn.click(
            fn=chat_interface,
            inputs=[msg_input, chatbot, temperature, max_tokens],
            outputs=[chatbot, msg_input],
            queue=False
        )
        
        clear_btn.click(
            fn=clear_conversation,
            outputs=[chatbot, msg_input],
            queue=False
        )
    
    return demo

# ===== Main =====
def main():
    """メイン実行"""
    
    # CLIと同じ初期化
    success = initialize_model()
    if not success:
        print("❌ Failed to initialize model!")
        return
    
    # Gradio起動
    demo = create_interface()
    print("🌟 Starting CLI-Direct Gradio server...")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        enable_queue=False
    )

if __name__ == "__main__":
    main()