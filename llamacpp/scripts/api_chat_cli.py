#!/usr/bin/env python3
"""
API Chat CLI Client
FastAPI LLMサーバーとのCLI会話クライアント
"""

import asyncio
import httpx
import json
import sys
from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel

console = Console()

class APIChatCLI:
    def __init__(self, base_url: str = "http://localhost:8001"):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=60.0)
    
    async def check_server(self):
        """サーバー接続確認"""
        try:
            response = await self.client.get(f"{self.base_url}/health")
            if response.status_code == 200:
                health = response.json()
                console.print("✅ Server connected successfully!", style="green")
                console.print(f"   Status: {health.get('status', 'unknown')}")
                console.print(f"   Model loaded: {health.get('model_loaded', False)}")
                return True
            else:
                console.print(f"❌ Server error: {response.status_code}", style="red")
                return False
        except Exception as e:
            console.print(f"❌ Cannot connect to server: {e}", style="red")
            console.print(f"   Make sure server is running at {self.base_url}", style="yellow")
            return False
    
    async def set_tsundere_preset(self):
        """ツンデレプリセット設定"""
        try:
            response = await self.client.post(f"{self.base_url}/presets/tsundere")
            if response.status_code == 200:
                console.print("🎭 Tsundere mode activated!", style="magenta")
            else:
                console.print("⚠️ Failed to set tsundere preset", style="yellow")
        except Exception as e:
            console.print(f"❌ Preset error: {e}", style="red")
    
    async def send_message(self, message: str, use_history: bool = True):
        """メッセージ送信"""
        try:
            chat_request = {
                "message": message,
                "use_history": use_history,
                "stream": False
            }
            
            console.print("🤔 Thinking...", style="dim")
            response = await self.client.post(
                f"{self.base_url}/chat",
                json=chat_request
            )
            
            if response.status_code == 200:
                result = response.json()
                response_text = result.get("response", "No response")
                inference_time = result.get("inference_time", 0)
                
                console.print(f"\n[bold blue]ツンデレちゃん[/bold blue]: {response_text}")
                console.print(f"[dim]⏱️ Response time: {inference_time:.2f}s[/dim]")
                return True
            else:
                console.print(f"❌ Chat error: {response.status_code} - {response.text}", style="red")
                return False
                
        except Exception as e:
            console.print(f"❌ Message error: {e}", style="red")
            return False
    
    async def get_history(self):
        """会話履歴表示"""
        try:
            response = await self.client.get(f"{self.base_url}/history")
            if response.status_code == 200:
                history = response.json()
                console.print(f"\n📚 Conversation history ({history['count']} messages):")
                for msg in history['messages'][-6:]:  # 最新6件
                    role = msg['role']
                    content = msg['content'][:100] + "..." if len(msg['content']) > 100 else msg['content']
                    if role == "user":
                        console.print(f"   [green]You[/green]: {content}")
                    elif role == "assistant":
                        console.print(f"   [blue]ツンデレちゃん[/blue]: {content}")
            else:
                console.print("❌ Cannot get history", style="red")
        except Exception as e:
            console.print(f"❌ History error: {e}", style="red")
    
    async def clear_history(self):
        """会話履歴クリア"""
        try:
            response = await self.client.delete(f"{self.base_url}/history")
            if response.status_code == 200:
                console.print("🗑️ History cleared!", style="yellow")
            else:
                console.print("❌ Cannot clear history", style="red")
        except Exception as e:
            console.print(f"❌ Clear error: {e}", style="red")
    
    async def start_chat(self):
        """チャット開始"""
        # ヘッダー表示
        console.print(Panel.fit(
            "[bold blue]🤖 FastAPI LLM Chat CLI[/bold blue]\n"
            "[dim]Interactive chat with API server[/dim]",
            border_style="blue"
        ))
        
        # サーバー接続確認
        if not await self.check_server():
            return
        
        # ツンデレプリセット設定
        await self.set_tsundere_preset()
        
        # 使用方法表示
        console.print("\n💡 Commands:", style="blue")
        console.print("   • Type your message and press Enter", style="dim")
        console.print("   • '/quit' or '/exit' to exit", style="dim")
        console.print("   • '/history' to show conversation history", style="dim")
        console.print("   • '/clear' to clear conversation history", style="dim")
        console.print("   • '/help' to show this help", style="dim")
        
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
                    console.print("\n👋 Goodbye!", style="blue")
                    break
                elif user_input.lower() == '/history':
                    await self.get_history()
                    continue
                elif user_input.lower() == '/clear':
                    await self.clear_history()
                    continue
                elif user_input.lower() == '/help':
                    console.print("\n💡 Available commands:", style="blue")
                    console.print("   /quit, /exit - Exit chat", style="dim")
                    console.print("   /history - Show conversation history", style="dim")
                    console.print("   /clear - Clear conversation history", style="dim")
                    console.print("   /help - Show this help", style="dim")
                    continue
                
                # メッセージ送信
                await self.send_message(user_input)
                
            except KeyboardInterrupt:
                console.print("\n\n👋 Chat interrupted. Goodbye!", style="blue")
                break
            except Exception as e:
                console.print(f"\n❌ Error: {e}", style="red")
        
        await self.client.aclose()

async def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="FastAPI LLM Chat CLI Client")
    parser.add_argument("--url", default="http://localhost:8001", help="API server URL")
    args = parser.parse_args()
    
    cli = APIChatCLI(args.url)
    await cli.start_chat()

if __name__ == "__main__":
    asyncio.run(main())