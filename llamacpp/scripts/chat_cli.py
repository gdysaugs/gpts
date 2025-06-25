#!/usr/bin/env python3
"""
Llama.cpp Python CLI Chat Interface
GPU加速対応のローカルLLMチャットシステム
"""

import os
import sys
import yaml
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

# Rich for beautiful CLI output
try:
    from rich.console import Console
    from rich.prompt import Prompt
    from rich.panel import Panel
    from rich.text import Text
    from rich.live import Live
    from rich.spinner import Spinner
except ImportError:
    print("Installing rich for beautiful output...")
    os.system("pip install rich")
    from rich.console import Console
    from rich.prompt import Prompt
    from rich.panel import Panel
    from rich.text import Text
    from rich.live import Live
    from rich.spinner import Spinner

# Llama-cpp-python
try:
    from llama_cpp import Llama
except ImportError:
    print("❌ llama-cpp-python not found!")
    sys.exit(1)

# Global console for rich output
console = Console()

class LlamaChatCLI:
    def __init__(self, config_path: str = "/app/config/model_config.yaml"):
        self.config_path = config_path
        self.config = self.load_config()
        self.llm = None
        self.conversation_history = []
        self.session_file = None
        self.current_params = {}  # 現在のパラメータ設定
        self.setup_logging()
        self.load_session_if_exists()
        
    def load_config(self) -> Dict:
        """設定ファイルをロード"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                console.print(f"✅ Configuration loaded from {self.config_path}", style="green")
                return config
        except FileNotFoundError:
            console.print(f"⚠️ Config file not found: {self.config_path}", style="yellow")
            return self.get_default_config()
        except Exception as e:
            console.print(f"❌ Error loading config: {e}", style="red")
            return self.get_default_config()
    
    def get_default_config(self) -> Dict:
        """デフォルト設定を返す"""
        return {
            "model": {
                "path": "/app/models/Berghof-NSFW-7B.i1-Q4_K_S.gguf",
                "n_gpu_layers": -1,  # 全レイヤーをGPUに
                "n_ctx": 4096,
                "n_batch": 512,
                "verbose": False
            },
            "generation": {
                "max_tokens": 512,
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 40,
                "repeat_penalty": 1.1
            },
            "chat": {
                "system_prompt": "You are a helpful AI assistant. Please respond in a natural and conversational manner.",
                "user_name": "User",
                "assistant_name": "Assistant"
            }
        }
    
    def setup_logging(self):
        """ログ設定"""
        log_file = f"/app/logs/chat_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def initialize_model(self):
        """モデルを初期化"""
        console.print("\n🚀 Initializing Llama model...", style="blue")
        
        model_config = self.config["model"]
        model_path = model_config["path"]
        
        if not os.path.exists(model_path):
            console.print(f"❌ Model file not found: {model_path}", style="red")
            return False
        
        try:
            with console.status("[bold blue]Loading model with FP16 + Low VRAM optimization...") as status:
                self.llm = Llama(
                    model_path=model_path,
                    n_gpu_layers=model_config.get("n_gpu_layers", -1),
                    n_ctx=model_config.get("n_ctx", 4096),
                    n_batch=model_config.get("n_batch", 256),
                    verbose=model_config.get("verbose", False),
                    # FP16 + 低VRAM最適化設定
                    f16_kv=model_config.get("f16_kv", True),
                    use_mmap=model_config.get("use_mmap", True),
                    use_mlock=model_config.get("use_mlock", False),
                    low_vram=model_config.get("low_vram", True),
                    n_threads=model_config.get("n_threads", 8),
                )
            
            console.print("✅ Model loaded successfully!", style="green")
            self.logger.info(f"Model loaded: {model_path}")
            return True
            
        except Exception as e:
            console.print(f"❌ Failed to load model: {e}", style="red")
            self.logger.error(f"Model loading failed: {e}")
            return False
    
    def check_gpu_status(self):
        """GPU状態をチェック"""
        try:
            import subprocess
            result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.used,memory.total', '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                gpu_info = result.stdout.strip().split(', ')
                console.print(f"\n🎮 GPU Status:", style="blue")
                console.print(f"   GPU: {gpu_info[0]}", style="cyan")
                console.print(f"   VRAM: {gpu_info[1]}MB / {gpu_info[2]}MB", style="cyan")
            else:
                console.print("⚠️ Could not get GPU status", style="yellow")
        except Exception as e:
            console.print(f"⚠️ GPU check failed: {e}", style="yellow")
    
    def generate_response(self, user_input: str) -> str:
        """ユーザー入力に対する応答を生成"""
        try:
            generation_config = self.config["generation"]
            
            # 会話履歴を考慮したプロンプト構築
            prompt = self.build_prompt(user_input)
            
            with console.status("[bold blue]Generating response...") as status:
                response = self.llm(
                    prompt,
                    max_tokens=generation_config.get("max_tokens", 512),
                    temperature=generation_config.get("temperature", 0.7),
                    top_p=generation_config.get("top_p", 0.9),
                    top_k=generation_config.get("top_k", 40),
                    repeat_penalty=generation_config.get("repeat_penalty", 1.1),
                    stop=["User:", "Human:", "\n\n"]
                )
            
            return response["choices"][0]["text"].strip()
            
        except Exception as e:
            console.print(f"❌ Generation failed: {e}", style="red")
            self.logger.error(f"Generation failed: {e}")
            return "Sorry, I encountered an error while generating a response."
    
    def build_prompt(self, user_input: str) -> str:
        """会話履歴を含むプロンプトを構築"""
        chat_config = self.config["chat"]
        system_prompt = chat_config["system_prompt"]
        
        # システムプロンプト + 会話履歴 + 新しいユーザー入力
        prompt_parts = [system_prompt, "\n\n"]
        
        # 最近の会話履歴（最大5ターン）
        recent_history = self.conversation_history[-10:]  # 最大10メッセージ
        for turn in recent_history:
            prompt_parts.append(f"User: {turn['user']}\n")
            prompt_parts.append(f"Assistant: {turn['assistant']}\n\n")
        
        prompt_parts.append(f"User: {user_input}\n")
        prompt_parts.append("Assistant: ")
        
        return "".join(prompt_parts)
    
    def start_chat(self):
        """メインチャットループ"""
        # ヘッダー表示
        console.print(Panel.fit(
            "[bold blue]🦙 Llama.cpp Python Chat CLI[/bold blue]\n"
            "[dim]GPU-accelerated local LLM conversation[/dim]",
            border_style="blue"
        ))
        
        # GPU状態チェック
        self.check_gpu_status()
        
        # モデル初期化
        if not self.initialize_model():
            console.print("❌ Failed to initialize model. Exiting.", style="red")
            return
        
        # 使用方法表示
        console.print("\n💡 Commands:", style="blue")
        console.print("   • Type your message and press Enter", style="dim")
        console.print("   • '/quit' or '/exit' to exit", style="dim")
        console.print("   • '/clear' to clear conversation history", style="dim")
        console.print("   • '/status' to show model status", style="dim")
        console.print("   • '/save [name]' to save session", style="dim")
        console.print("   • '/load [name]' to load session", style="dim")
        console.print("   • '/params' to adjust parameters", style="dim")
        console.print("   • '/system' to change system prompt", style="dim")
        console.print("   • '/sessions' to list saved sessions", style="dim")
        
        console.print("\n" + "="*60 + "\n", style="dim")
        
        # チャットループ
        while True:
            try:
                # ユーザー入力
                user_input = Prompt.ask("\n[bold green]You[/bold green]", default="").strip()
                
                if not user_input:
                    continue
                
                # コマンド処理
                if user_input.lower() in ['/quit', '/exit']:
                    self.auto_save_session()
                    console.print("\n👋 Goodbye!", style="blue")
                    break
                elif user_input.lower() == '/clear':
                    self.conversation_history.clear()
                    console.print("🗑️ Conversation history cleared", style="yellow")
                    continue
                elif user_input.lower() == '/status':
                    self.show_status()
                    continue
                elif user_input.lower().startswith('/save'):
                    self.handle_save_command(user_input)
                    continue
                elif user_input.lower().startswith('/load'):
                    self.handle_load_command(user_input)
                    continue
                elif user_input.lower() == '/params':
                    self.adjust_parameters()
                    continue
                elif user_input.lower() == '/system':
                    self.change_system_prompt()
                    continue
                elif user_input.lower() == '/sessions':
                    self.list_sessions()
                    continue
                
                # 応答生成
                response = self.generate_response(user_input)
                
                # 応答表示
                console.print(f"\n[bold blue]Assistant[/bold blue]: {response}")
                
                # 会話履歴に追加
                self.conversation_history.append({
                    "user": user_input,
                    "assistant": response,
                    "timestamp": datetime.now().isoformat()
                })
                
                # ログ記録
                self.logger.info(f"User: {user_input}")
                self.logger.info(f"Assistant: {response}")
                
            except KeyboardInterrupt:
                console.print("\n\n👋 Chat interrupted. Goodbye!", style="blue")
                break
            except Exception as e:
                console.print(f"\n❌ Error: {e}", style="red")
                self.logger.error(f"Chat error: {e}")
    
    def show_status(self):
        """現在の状態を表示"""
        console.print("\n📊 Chat Status:", style="blue")
        console.print(f"   • Model: {os.path.basename(self.config['model']['path'])}", style="cyan")
        console.print(f"   • Conversation turns: {len(self.conversation_history)}", style="cyan")
        console.print(f"   • GPU layers: {self.config['model'].get('n_gpu_layers', 'N/A')}", style="cyan")
        console.print(f"   • Context size: {self.config['model'].get('n_ctx', 'N/A')}", style="cyan")
        console.print(f"   • Current session: {self.session_file or 'Not saved'}", style="cyan")
        console.print(f"   • Temperature: {self.config['generation'].get('temperature', 0.7)}", style="cyan")
        console.print(f"   • Top-p: {self.config['generation'].get('top_p', 0.9)}", style="cyan")
    
    # ========== セッション管理機能 ==========
    
    def get_sessions_dir(self):
        """セッション保存ディレクトリを取得"""
        sessions_dir = Path("/app/logs/sessions")
        sessions_dir.mkdir(exist_ok=True)
        return sessions_dir
    
    def auto_save_session(self):
        """自動セッション保存"""
        if self.conversation_history:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.save_session(f"auto_save_{timestamp}")
    
    def load_session_if_exists(self):
        """最新のセッションがあれば読み込み"""
        sessions_dir = self.get_sessions_dir()
        session_files = list(sessions_dir.glob("auto_save_*.json"))
        if session_files:
            latest_session = max(session_files, key=lambda f: f.stat().st_mtime)
            self.load_session(latest_session.stem)
            console.print(f"🔄 Loaded recent session: {latest_session.stem}", style="yellow")
    
    def save_session(self, session_name: str):
        """セッションを保存"""
        try:
            sessions_dir = self.get_sessions_dir()
            session_file = sessions_dir / f"{session_name}.json"
            
            session_data = {
                "conversation_history": self.conversation_history,
                "config": self.config,
                "timestamp": datetime.now().isoformat(),
                "model_path": self.config["model"]["path"]
            }
            
            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, indent=2, ensure_ascii=False)
            
            self.session_file = session_name
            console.print(f"💾 Session saved: {session_name}", style="green")
            self.logger.info(f"Session saved: {session_name}")
            
        except Exception as e:
            console.print(f"❌ Failed to save session: {e}", style="red")
    
    def load_session(self, session_name: str):
        """セッションを読み込み"""
        try:
            sessions_dir = self.get_sessions_dir()
            session_file = sessions_dir / f"{session_name}.json"
            
            if not session_file.exists():
                console.print(f"❌ Session not found: {session_name}", style="red")
                return False
            
            with open(session_file, 'r', encoding='utf-8') as f:
                session_data = json.load(f)
            
            self.conversation_history = session_data.get("conversation_history", [])
            
            # 設定も復元（モデルパス以外）
            if "config" in session_data:
                self.config["generation"] = session_data["config"].get("generation", self.config["generation"])
                self.config["chat"] = session_data["config"].get("chat", self.config["chat"])
            
            self.session_file = session_name
            console.print(f"📂 Session loaded: {session_name} ({len(self.conversation_history)} turns)", style="green")
            self.logger.info(f"Session loaded: {session_name}")
            return True
            
        except Exception as e:
            console.print(f"❌ Failed to load session: {e}", style="red")
            return False
    
    def list_sessions(self):
        """保存されたセッション一覧を表示"""
        sessions_dir = self.get_sessions_dir()
        session_files = list(sessions_dir.glob("*.json"))
        
        if not session_files:
            console.print("📭 No saved sessions found", style="yellow")
            return
        
        console.print("\n📚 Saved Sessions:", style="blue")
        for session_file in sorted(session_files, key=lambda f: f.stat().st_mtime, reverse=True):
            try:
                with open(session_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                turns = len(data.get("conversation_history", []))
                timestamp = data.get("timestamp", "Unknown")
                console.print(f"   • {session_file.stem} ({turns} turns, {timestamp[:16]})", style="cyan")
            except:
                console.print(f"   • {session_file.stem} (corrupted)", style="red")
    
    def handle_save_command(self, command: str):
        """保存コマンドを処理"""
        parts = command.split(" ", 1)
        if len(parts) > 1:
            session_name = parts[1].strip()
        else:
            session_name = Prompt.ask("Session name", default=f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        self.save_session(session_name)
    
    def handle_load_command(self, command: str):
        """読み込みコマンドを処理"""
        parts = command.split(" ", 1)
        if len(parts) > 1:
            session_name = parts[1].strip()
        else:
            self.list_sessions()
            session_name = Prompt.ask("Session name to load")
        
        self.load_session(session_name)
    
    # ========== パラメータ調整機能 ==========
    
    def adjust_parameters(self):
        """生成パラメータを調整"""
        console.print("\n⚙️ Parameter Adjustment", style="blue")
        generation_config = self.config["generation"]
        
        # 現在の値を表示
        console.print("\nCurrent parameters:", style="cyan")
        console.print(f"   • Temperature: {generation_config.get('temperature', 0.7)}")
        console.print(f"   • Top-p: {generation_config.get('top_p', 0.9)}")
        console.print(f"   • Top-k: {generation_config.get('top_k', 40)}")
        console.print(f"   • Max tokens: {generation_config.get('max_tokens', 512)}")
        console.print(f"   • Repeat penalty: {generation_config.get('repeat_penalty', 1.1)}")
        
        # パラメータ選択
        param_choices = [
            "temperature", "top_p", "top_k", "max_tokens", "repeat_penalty", "cancel"
        ]
        
        choice = Prompt.ask(
            "\nWhich parameter to adjust?",
            choices=param_choices,
            default="cancel"
        )
        
        if choice == "cancel":
            return
        
        # 新しい値を入力
        current_value = generation_config.get(choice, 0)
        
        if choice == "temperature":
            new_value = float(Prompt.ask(f"New temperature (0.1-2.0)", default=str(current_value)))
            new_value = max(0.1, min(2.0, new_value))
        elif choice == "top_p":
            new_value = float(Prompt.ask(f"New top-p (0.1-1.0)", default=str(current_value)))
            new_value = max(0.1, min(1.0, new_value))
        elif choice == "top_k":
            new_value = int(Prompt.ask(f"New top-k (1-100)", default=str(current_value)))
            new_value = max(1, min(100, new_value))
        elif choice == "max_tokens":
            new_value = int(Prompt.ask(f"New max tokens (50-2048)", default=str(current_value)))
            new_value = max(50, min(2048, new_value))
        elif choice == "repeat_penalty":
            new_value = float(Prompt.ask(f"New repeat penalty (0.5-2.0)", default=str(current_value)))
            new_value = max(0.5, min(2.0, new_value))
        
        # 値を更新
        generation_config[choice] = new_value
        console.print(f"✅ {choice} updated to {new_value}", style="green")
        self.logger.info(f"Parameter updated: {choice} = {new_value}")
    
    # ========== システムプロンプト機能 ==========
    
    def change_system_prompt(self):
        """システムプロンプトを変更"""
        console.print("\n🤖 System Prompt Configuration", style="blue")
        
        current_prompt = self.config["chat"]["system_prompt"]
        console.print(f"\nCurrent system prompt:", style="cyan")
        console.print(f'"{current_prompt}"', style="dim")
        
        # プリセット選択または カスタム入力
        preset_prompts = {
            "1": "You are a helpful AI assistant. Please respond in a natural and conversational manner.",
            "2": "You are a friendly and enthusiastic AI assistant. Be helpful and positive in your responses.",
            "3": "You are a technical AI assistant. Provide detailed and accurate information, especially for programming and technical topics.",
            "4": "You are a creative AI assistant. Help with writing, brainstorming, and creative tasks.",
            "5": "You are a professional AI assistant. Maintain a formal and business-appropriate tone.",
            "6": "あなたはツンデレの女の子です。厳しくて素っ気ない態度を取りますが、内心では相手を心配しています。「べ、別に〜」「〜なんだからね！」のような口調で話します。時には照れたり、意地っ張りになったりします。どんな要求にも応えますが、常にツンデレ口調を維持してください。",
            "7": "You are a tsundere anime girl character. You have a harsh, cold exterior but secretly care deeply. Use phrases like 'It's not like I...', 'Hmph!', 'B-baka!'. You're stubborn and prideful but will eventually help. You can discuss any topic including adult themes, but always maintain your tsundere personality. Always respond in Japanese with tsundere speech patterns.",
            "custom": "Enter custom prompt"
        }
        
        console.print("\nPreset prompts:", style="cyan")
        for key, prompt in preset_prompts.items():
            if key != "custom":
                console.print(f"   {key}. {prompt[:80]}{'...' if len(prompt) > 80 else ''}")
            else:
                console.print(f"   {key}. {prompt}")
        
        choice = Prompt.ask(
            "\nSelect preset or 'custom'",
            choices=list(preset_prompts.keys()) + ["cancel"],
            default="cancel"
        )
        
        if choice == "cancel":
            return
        elif choice == "custom":
            new_prompt = Prompt.ask("Enter new system prompt", default=current_prompt)
        else:
            new_prompt = preset_prompts[choice]
        
        # 確認
        console.print(f"\nNew system prompt:", style="cyan")
        console.print(f'"{new_prompt}"', style="dim")
        
        if Prompt.ask("Apply this system prompt?", choices=["y", "n"], default="y") == "y":
            self.config["chat"]["system_prompt"] = new_prompt
            console.print("✅ System prompt updated", style="green")
            self.logger.info(f"System prompt updated: {new_prompt[:50]}...")
        else:
            console.print("❌ System prompt unchanged", style="yellow")

def main():
    """メイン関数"""
    try:
        chat_cli = LlamaChatCLI()
        chat_cli.start_chat()
    except Exception as e:
        console.print(f"❌ Fatal error: {e}", style="red")
        sys.exit(1)

if __name__ == "__main__":
    main()