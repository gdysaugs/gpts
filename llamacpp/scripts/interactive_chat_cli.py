#!/usr/bin/env python3
"""
Interactive Chat CLI for Enhanced FastAPI Server
事前ロード最適化FastAPIサーバー用インタラクティブチャットCLI
"""

import argparse
import requests
import json
import time
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional

# カラー出力用
try:
    from colorama import Fore, Back, Style, init
    init()
    COLORS_AVAILABLE = True
except ImportError:
    COLORS_AVAILABLE = False
    # フォールバック用の空クラス
    class Fore:
        RED = GREEN = YELLOW = BLUE = MAGENTA = CYAN = WHITE = ""
    class Back:
        RED = GREEN = YELLOW = BLUE = MAGENTA = CYAN = WHITE = ""
    class Style:
        DIM = NORMAL = BRIGHT = RESET_ALL = ""

class InteractiveChatCLI:
    """インタラクティブチャットCLI"""
    
    def __init__(self, api_url: str = "http://localhost:8001"):
        self.api_url = api_url
        self.current_character = "tsundere"
        self.temperature = 0.7
        self.max_tokens = 512
        self.available_presets = []
        self.session_start = datetime.now()
        self.message_count = 0
        
        # サーバー接続テスト
        self.test_connection()
        
        # プリセット取得
        self.load_presets()
        
    def test_connection(self):
        """サーバー接続テスト"""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=5)
            if response.status_code == 200:
                health_data = response.json()
                self.print_success(f"✅ サーバー接続成功: {self.api_url}")
                self.print_info(f"   モデル読み込み: {health_data.get('model_loaded', 'N/A')}")
                self.print_info(f"   稼働時間: {health_data.get('uptime', 'N/A')}")
                
                # 事前ロード状態表示
                preload_status = health_data.get('preload_status', {})
                self.print_info(f"   事前ロード状態: {preload_status.get('cache_size', 0)}項目")
                
            else:
                self.print_error(f"❌ サーバー応答エラー: {response.status_code}")
                sys.exit(1)
                
        except requests.exceptions.RequestException as e:
            self.print_error(f"❌ サーバー接続失敗: {e}")
            self.print_info("💡 サーバーを起動してください:")
            self.print_info("   docker run --gpus all --rm -it -p 8001:8001 llama-cpp-python:cuda python /app/src/fastapi_chat_server.py")
            sys.exit(1)
    
    def load_presets(self):
        """プリセット読み込み"""
        try:
            response = requests.get(f"{self.api_url}/presets")
            if response.status_code == 200:
                data = response.json()
                self.available_presets = data.get('presets', [])
                self.print_success(f"✅ {len(self.available_presets)}個のプリセットを読み込み")
            else:
                self.print_warning("⚠️ プリセット読み込み失敗")
                self.available_presets = ["tsundere", "friendly", "technical"]
        except Exception as e:
            self.print_warning(f"⚠️ プリセット読み込みエラー: {e}")
            self.available_presets = ["tsundere", "friendly", "technical"]
    
    def print_success(self, message: str):
        """成功メッセージ"""
        if COLORS_AVAILABLE:
            print(f"{Fore.GREEN}{message}{Style.RESET_ALL}")
        else:
            print(message)
    
    def print_error(self, message: str):
        """エラーメッセージ"""
        if COLORS_AVAILABLE:
            print(f"{Fore.RED}{message}{Style.RESET_ALL}")
        else:
            print(message)
    
    def print_warning(self, message: str):
        """警告メッセージ"""
        if COLORS_AVAILABLE:
            print(f"{Fore.YELLOW}{message}{Style.RESET_ALL}")
        else:
            print(message)
    
    def print_info(self, message: str):
        """情報メッセージ"""
        if COLORS_AVAILABLE:
            print(f"{Fore.CYAN}{message}{Style.RESET_ALL}")
        else:
            print(message)
    
    def print_user(self, message: str):
        """ユーザーメッセージ"""
        if COLORS_AVAILABLE:
            print(f"{Fore.BLUE}👤 You: {message}{Style.RESET_ALL}")
        else:
            print(f"👤 You: {message}")
    
    def print_assistant(self, message: str, character: str, inference_time: float = 0, tokens_per_second: float = 0):
        """アシスタントメッセージ"""
        # キャラクターに応じたアイコン
        icons = {
            'tsundere': '🎭',
            'friendly': '😊',
            'technical': '🔧',
            'casual': '😎',
            'polite': '🙏',
            'creative': '🎨',
            'academic': '📚'
        }
        icon = icons.get(character, '🤖')
        
        if COLORS_AVAILABLE:
            print(f"{Fore.MAGENTA}{icon} {character.capitalize()}: {message}{Style.RESET_ALL}")
            if inference_time > 0:
                print(f"{Fore.WHITE}{Style.DIM}   (⏱️ {inference_time:.2f}s, 🚀 {tokens_per_second:.1f} tokens/s){Style.RESET_ALL}")
        else:
            print(f"{icon} {character.capitalize()}: {message}")
            if inference_time > 0:
                print(f"   (⏱️ {inference_time:.2f}s, 🚀 {tokens_per_second:.1f} tokens/s)")
    
    def print_header(self):
        """ヘッダー表示"""
        if COLORS_AVAILABLE:
            print(f"{Fore.CYAN}{'='*80}{Style.RESET_ALL}")
            print(f"{Fore.CYAN}🚀 LlamaCPP Interactive Chat CLI (Enhanced FastAPI){Style.RESET_ALL}")
            print(f"{Fore.CYAN}{'='*80}{Style.RESET_ALL}")
        else:
            print("="*80)
            print("🚀 LlamaCPP Interactive Chat CLI (Enhanced FastAPI)")
            print("="*80)
        
        print(f"📡 サーバー: {self.api_url}")
        print(f"🎭 キャラクター: {self.current_character}")
        print(f"🌡️  温度: {self.temperature}")
        print(f"📝 最大トークン: {self.max_tokens}")
        print()
    
    def print_help(self):
        """ヘルプ表示"""
        help_text = """
🎯 利用可能なコマンド:
  /help              - このヘルプを表示
  /character <name>  - キャラクターを変更 (例: /character friendly)
  /temp <value>      - 温度を変更 (0.1-2.0, 例: /temp 0.8)
  /tokens <value>    - 最大トークン数を変更 (例: /tokens 256)
  /presets           - 利用可能なプリセット一覧を表示
  /status            - サーバーステータスを表示
  /clear             - 会話履歴をクリア
  /stats             - セッション統計を表示
  /quit, /exit       - 終了

🎭 利用可能なキャラクター:
"""
        self.print_info(help_text)
        
        for preset in self.available_presets:
            descriptions = {
                'tsundere': '🎭 ツンデレ - 「べ、別に〜」「〜なんだからね！」',
                'friendly': '😊 フレンドリー - 明るく親しみやすい',
                'technical': '🔧 技術的 - プログラミング・技術特化',
                'casual': '😎 カジュアル - 友達感覚のタメ口',
                'polite': '🙏 丁寧 - 非常に礼儀正しい敬語',
                'creative': '🎨 クリエイティブ - 詩的で創造的',
                'academic': '📚 学術的 - 論理的で専門的'
            }
            desc = descriptions.get(preset, f'🤖 {preset.capitalize()}')
            self.print_info(f"  {preset:<12} - {desc}")
        
        print()
    
    def handle_command(self, user_input: str) -> bool:
        """コマンド処理"""
        parts = user_input.strip().split()
        command = parts[0].lower()
        
        if command in ['/help', '/h']:
            self.print_help()
            return True
        
        elif command == '/character':
            if len(parts) < 2:
                self.print_error("❌ 使用方法: /character <キャラクター名>")
                return True
            
            new_character = parts[1].lower()
            if new_character in self.available_presets:
                self.current_character = new_character
                self.print_success(f"✅ キャラクターを '{new_character}' に変更しました")
            else:
                self.print_error(f"❌ キャラクター '{new_character}' は利用できません")
                self.print_info(f"利用可能: {', '.join(self.available_presets)}")
            return True
        
        elif command == '/temp':
            if len(parts) < 2:
                self.print_error("❌ 使用方法: /temp <温度値>")
                return True
            
            try:
                new_temp = float(parts[1])
                if 0.1 <= new_temp <= 2.0:
                    self.temperature = new_temp
                    self.print_success(f"✅ 温度を {new_temp} に変更しました")
                else:
                    self.print_error("❌ 温度は 0.1-2.0 の範囲で指定してください")
            except ValueError:
                self.print_error("❌ 無効な温度値です")
            return True
        
        elif command == '/tokens':
            if len(parts) < 2:
                self.print_error("❌ 使用方法: /tokens <トークン数>")
                return True
            
            try:
                new_tokens = int(parts[1])
                if 1 <= new_tokens <= 2048:
                    self.max_tokens = new_tokens
                    self.print_success(f"✅ 最大トークン数を {new_tokens} に変更しました")
                else:
                    self.print_error("❌ トークン数は 1-2048 の範囲で指定してください")
            except ValueError:
                self.print_error("❌ 無効なトークン数です")
            return True
        
        elif command == '/presets':
            self.print_info("🎭 利用可能なプリセット:")
            for preset in self.available_presets:
                marker = "👉" if preset == self.current_character else "  "
                self.print_info(f"{marker} {preset}")
            return True
        
        elif command == '/status':
            self.show_status()
            return True
        
        elif command == '/clear':
            self.clear_history()
            return True
        
        elif command == '/stats':
            self.show_stats()
            return True
        
        elif command in ['/quit', '/exit']:
            self.print_success("👋 チャットを終了します")
            return False
        
        else:
            self.print_error(f"❌ 不明なコマンド: {command}")
            self.print_info("💡 /help でヘルプを表示")
            return True
    
    def show_status(self):
        """サーバーステータス表示"""
        try:
            response = requests.get(f"{self.api_url}/status")
            if response.status_code == 200:
                data = response.json()
                
                self.print_info("📊 サーバーステータス:")
                self.print_info(f"   稼働時間: {data.get('server', {}).get('uptime', 'N/A')}")
                self.print_info(f"   モデル読み込み: {data.get('server', {}).get('models_loaded', 'N/A')}")
                self.print_info(f"   キャッシュサイズ: {data.get('server', {}).get('cache_size', 'N/A')}")
                
                model_info = data.get('model', {})
                self.print_info(f"   モデルパス: {model_info.get('model_path', 'N/A')}")
                self.print_info(f"   コンテキストサイズ: {model_info.get('context_size', 'N/A')}")
                
            else:
                self.print_error(f"❌ ステータス取得エラー: {response.status_code}")
        except Exception as e:
            self.print_error(f"❌ ステータス取得失敗: {e}")
    
    def clear_history(self):
        """履歴クリア"""
        try:
            response = requests.delete(f"{self.api_url}/history")
            if response.status_code == 200:
                self.print_success("✅ 会話履歴をクリアしました")
            else:
                self.print_error(f"❌ 履歴クリアエラー: {response.status_code}")
        except Exception as e:
            self.print_error(f"❌ 履歴クリア失敗: {e}")
    
    def show_stats(self):
        """セッション統計表示"""
        uptime = datetime.now() - self.session_start
        
        self.print_info("📈 セッション統計:")
        self.print_info(f"   セッション開始: {self.session_start.strftime('%Y-%m-%d %H:%M:%S')}")
        self.print_info(f"   セッション時間: {uptime}")
        self.print_info(f"   メッセージ数: {self.message_count}")
        self.print_info(f"   現在のキャラクター: {self.current_character}")
        self.print_info(f"   現在の温度: {self.temperature}")
        self.print_info(f"   最大トークン数: {self.max_tokens}")
    
    def send_message(self, message: str):
        """メッセージ送信"""
        try:
            request_data = {
                "message": message,
                "character": self.current_character,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens
            }
            
            start_time = time.time()
            response = requests.post(f"{self.api_url}/interactive", json=request_data)
            request_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                self.print_assistant(
                    data['response'],
                    data['character'],
                    data['inference_time'],
                    data['tokens_per_second']
                )
                self.message_count += 1
            else:
                self.print_error(f"❌ 送信エラー: {response.status_code}")
                try:
                    error_data = response.json()
                    self.print_error(f"   詳細: {error_data.get('detail', 'N/A')}")
                except:
                    pass
                
        except Exception as e:
            self.print_error(f"❌ 送信失敗: {e}")
    
    def run(self):
        """メインループ"""
        self.print_header()
        self.print_info("💡 /help でヘルプを表示、/quit で終了")
        print()
        
        while True:
            try:
                # プロンプト表示
                if COLORS_AVAILABLE:
                    prompt = f"{Fore.BLUE}💬 > {Style.RESET_ALL}"
                else:
                    prompt = "💬 > "
                
                user_input = input(prompt).strip()
                
                # 空入力のスキップ
                if not user_input:
                    continue
                
                # コマンド処理
                if user_input.startswith('/'):
                    if not self.handle_command(user_input):
                        break
                    continue
                
                # ユーザーメッセージ表示
                self.print_user(user_input)
                
                # メッセージ送信
                self.send_message(user_input)
                
                print()  # 改行
                
            except KeyboardInterrupt:
                print()
                self.print_warning("⚠️ Ctrl+C が押されました")
                confirm = input("本当に終了しますか？ (y/N): ").lower()
                if confirm in ['y', 'yes']:
                    break
                print()
            except EOFError:
                print()
                self.print_success("👋 チャットを終了します")
                break

def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description="Interactive Chat CLI for Enhanced FastAPI Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # デフォルトサーバーに接続
  python interactive_chat_cli.py
  
  # 別のサーバーに接続
  python interactive_chat_cli.py --api-url http://localhost:8000
  
  # 初期キャラクターを指定
  python interactive_chat_cli.py --character friendly
  
  # 初期温度を指定
  python interactive_chat_cli.py --temperature 0.8
        """
    )
    
    parser.add_argument(
        "--api-url",
        default="http://localhost:8001",
        help="FastAPIサーバーのURL (デフォルト: http://localhost:8001)"
    )
    
    parser.add_argument(
        "--character",
        default="tsundere",
        help="初期キャラクター (デフォルト: tsundere)"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="初期温度 (デフォルト: 0.7)"
    )
    
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="最大トークン数 (デフォルト: 512)"
    )
    
    args = parser.parse_args()
    
    # CLIインスタンス作成
    cli = InteractiveChatCLI(api_url=args.api_url)
    cli.current_character = args.character
    cli.temperature = args.temperature
    cli.max_tokens = args.max_tokens
    
    # 実行
    cli.run()

if __name__ == "__main__":
    main()