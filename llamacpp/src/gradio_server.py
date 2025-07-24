#!/usr/bin/env python3
"""
Gradio UI Server for LlamaCPP
既存のLLMEngineを使用したWebUI
"""

import os
import gradio as gr
import logging
from datetime import datetime
from typing import List, Tuple, Optional
import json

# 既存のLLMEngineをインポート
from llm_engine import LLMEngine, ChatMessage, GenerationConfig

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class GradioLLMServer:
    """Gradio UI付きLLMサーバー"""
    
    def __init__(self, host: str = "0.0.0.0", port: int = 7860):
        self.host = host
        self.port = port
        self.llm_engine = None
        self.conversation_history = []
        
        # LLMエンジン初期化
        logger.info("Initializing LLM Engine...")
        self.llm_engine = LLMEngine()
        logger.info("LLM Engine initialized successfully!")
    
    def chat_response(self, 
                     message: str, 
                     history: List[dict], 
                     temperature: float = 0.7,
                     max_tokens: int = 512,
                     top_p: float = 0.9) -> Tuple[List[dict], str]:
        """チャット応答を生成"""
        if not message.strip():
            return history, ""
        
        try:
            # 生成設定
            config = GenerationConfig(
                max_tokens=int(max_tokens),
                temperature=float(temperature),
                top_p=float(top_p)
            )
            
            # LLMエンジンで応答生成
            start_time = datetime.now()
            response = self.llm_engine.generate_response(message, config)
            inference_time = (datetime.now() - start_time).total_seconds()
            
            # 履歴に追加（OpenAI形式）
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": response})
            
            logger.info(f"Generated response in {inference_time:.2f}s")
            return history, ""
            
        except Exception as e:
            error_msg = f"エラーが発生しました: {str(e)}"
            logger.error(f"Chat error: {e}")
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": error_msg})
            return history, ""
    
    def clear_history(self) -> Tuple[List, str]:
        """会話履歴をクリア"""
        if self.llm_engine:
            self.llm_engine.clear_conversation()
        logger.info("Conversation history cleared")
        return [], ""
    
    def change_preset(self, preset_name: str) -> str:
        """キャラクタープリセット変更"""
        try:
            if self.llm_engine:
                self.llm_engine.set_character_preset(preset_name)
                return f"プリセットを '{preset_name}' に変更しました"
        except Exception as e:
            return f"プリセット変更エラー: {str(e)}"
    
    def get_system_info(self) -> str:
        """システム情報取得"""
        try:
            if self.llm_engine:
                info = {
                    "model_loaded": self.llm_engine.model is not None,
                    "gpu_available": self.llm_engine.gpu_available,
                    "model_path": self.llm_engine.config.get("model", {}).get("path", "Unknown"),
                    "conversation_count": len(self.llm_engine.conversation_history)
                }
                return json.dumps(info, indent=2, ensure_ascii=False)
            return "LLM Engine not initialized"
        except Exception as e:
            return f"System info error: {str(e)}"
    
    def create_interface(self) -> gr.Blocks:
        """Gradio UIインターフェース作成"""
        
        with gr.Blocks(
            title="🦙 LlamaCPP Gradio Chat",
            theme=gr.themes.Soft(),
            css="""
            .gradio-container {
                max-width: 1200px;
                margin: auto;
            }
            .chat-container {
                height: 500px;
            }
            """
        ) as interface:
            
            # ヘッダー
            gr.HTML("""
            <div style="text-align: center; padding: 20px;">
                <h1>🦙 LlamaCPP Gradio Chat</h1>
                <p>GPU加速ローカルLLM with ツンデレキャラクター</p>
            </div>
            """)
            
            with gr.Row():
                with gr.Column(scale=3):
                    # メインチャット
                    chatbot = gr.Chatbot(
                        label="チャット",
                        height=500,
                        show_copy_button=True,
                        type="messages"
                    )
                    
                    with gr.Row():
                        msg_input = gr.Textbox(
                            label="メッセージ",
                            placeholder="ツンデレちゃんと話してみて...",
                            scale=4,
                            submit_btn=True
                        )
                        send_btn = gr.Button("送信", variant="primary", scale=1)
                    
                    # ボタン群
                    with gr.Row():
                        clear_btn = gr.Button("履歴クリア", variant="secondary")
                        
                with gr.Column(scale=1):
                    # 設定パネル
                    gr.HTML("<h3>⚙️ 生成設定</h3>")
                    
                    temperature = gr.Slider(
                        label="Temperature",
                        minimum=0.1,
                        maximum=1.0,
                        value=0.7,
                        step=0.1,
                        info="創造性 (低=保守的, 高=創造的)"
                    )
                    
                    max_tokens = gr.Slider(
                        label="Max Tokens",
                        minimum=50,
                        maximum=1000,
                        value=512,
                        step=50,
                        info="最大生成トークン数"
                    )
                    
                    top_p = gr.Slider(
                        label="Top-p",
                        minimum=0.1,
                        maximum=1.0,
                        value=0.9,
                        step=0.1,
                        info="Nucleus sampling"
                    )
                    
                    # キャラクタープリセット
                    gr.HTML("<h3>👤 キャラクター</h3>")
                    preset_dropdown = gr.Dropdown(
                        choices=["tsundere", "friendly", "technical"],
                        value="tsundere",
                        label="プリセット",
                        info="キャラクター設定"
                    )
                    
                    preset_btn = gr.Button("プリセット適用", variant="secondary")
                    preset_status = gr.Textbox(
                        label="ステータス",
                        value="Ready",
                        interactive=False
                    )
                    
                    # システム情報
                    gr.HTML("<h3>📊 システム情報</h3>")
                    info_btn = gr.Button("情報更新")
                    system_info = gr.Textbox(
                        label="システム状態",
                        lines=8,
                        interactive=False,
                        value="Click '情報更新' to load system info"
                    )
            
            # イベントハンドラー
            def submit_message(msg, history, temp, tokens, tp):
                return self.chat_response(msg, history, temp, tokens, tp)
            
            # 送信イベント
            msg_input.submit(
                fn=submit_message,
                inputs=[msg_input, chatbot, temperature, max_tokens, top_p],
                outputs=[chatbot, msg_input]
            )
            
            send_btn.click(
                fn=submit_message,
                inputs=[msg_input, chatbot, temperature, max_tokens, top_p],
                outputs=[chatbot, msg_input]
            )
            
            # クリアイベント
            clear_btn.click(
                fn=self.clear_history,
                outputs=[chatbot, msg_input]
            )
            
            # プリセット変更
            preset_btn.click(
                fn=self.change_preset,
                inputs=[preset_dropdown],
                outputs=[preset_status]
            )
            
            # システム情報更新
            info_btn.click(
                fn=self.get_system_info,
                outputs=[system_info]
            )
        
        return interface
    
    def launch(self, share: bool = False, debug: bool = False):
        """Gradioサーバー起動"""
        interface = self.create_interface()
        
        logger.info(f"Starting Gradio server on {self.host}:{self.port}")
        
        interface.launch(
            server_name=self.host,
            server_port=self.port,
            share=share,
            debug=debug,
            show_error=True,
            quiet=False
        )


def main():
    """メイン関数"""
    try:
        # 環境変数から設定取得
        host = os.getenv("GRADIO_HOST", "0.0.0.0")
        port = int(os.getenv("GRADIO_PORT", "7860"))
        share = os.getenv("GRADIO_SHARE", "false").lower() == "true"
        debug = os.getenv("GRADIO_DEBUG", "false").lower() == "true"
        
        # Gradioサーバー起動
        server = GradioLLMServer(host=host, port=port)
        server.launch(share=share, debug=debug)
        
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise


if __name__ == "__main__":
    main()