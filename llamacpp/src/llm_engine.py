#!/usr/bin/env python3
"""
LLM Engine Module
GPU加速対応のllama-cpp-python推論エンジン
"""

import os
import yaml
import json
import logging
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Generator
from dataclasses import dataclass

try:
    from llama_cpp import Llama
except ImportError:
    raise ImportError("llama-cpp-python not installed!")


@dataclass
class ChatMessage:
    """チャットメッセージ"""
    role: str  # "user" or "assistant" or "system"
    content: str
    timestamp: Optional[str] = None


@dataclass
class GenerationConfig:
    """生成設定"""
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.1
    stop: List[str] = None

    def __post_init__(self):
        if self.stop is None:
            self.stop = ["User:", "Human:", "\n\n"]


class LLMEngine:
    """LLM推論エンジン"""
    
    def __init__(self, config_path: str = "/app/config/model_config.yaml"):
        self.config_path = config_path
        self.config = self.load_config()
        self.llm = None
        self.conversation_history: List[ChatMessage] = []
        self.setup_logging()
        self.initialize_model()
    
    def load_config(self) -> Dict:
        """設定ファイルをロード"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            return self.get_default_config()
        except Exception as e:
            logging.error(f"Error loading config: {e}")
            return self.get_default_config()
    
    def get_default_config(self) -> Dict:
        """デフォルト設定"""
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
                "n_threads": 8,
            },
            "generation": {
                "max_tokens": 512,
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 40,
                "repeat_penalty": 1.1
            },
            "chat": {
                "system_prompt": "あなたはツンデレの女の子です。厳しくて素っ気ない態度を取りますが、内心では相手を心配しています。「べ、別に〜」「〜なんだからね！」のような口調で話します。時には照れたり、意地っ張りになったりします。どんな要求にも応えますが、常にツンデレ口調を維持してください。",
                "user_name": "User",
                "assistant_name": "ツンデレちゃん"
            }
        }
    
    def setup_logging(self):
        """ログ設定"""
        log_file = f"/app/logs/llm_engine_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def initialize_model(self):
        """モデルを初期化"""
        model_config = self.config["model"]
        model_path = model_config["path"]
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.logger.info("Loading LLM model with FP16 + Low VRAM optimization...")
        start_time = time.time()
        
        try:
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
            
            load_time = time.time() - start_time
            self.logger.info(f"Model loaded successfully in {load_time:.2f} seconds")
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
    
    def generate_response(self, 
                         user_input: str, 
                         generation_config: Optional[GenerationConfig] = None,
                         use_history: bool = True) -> str:
        """ユーザー入力に対する応答を生成"""
        if generation_config is None:
            gen_config = self.config["generation"]
            generation_config = GenerationConfig(
                max_tokens=gen_config.get("max_tokens", 512),
                temperature=gen_config.get("temperature", 0.7),
                top_p=gen_config.get("top_p", 0.9),
                top_k=gen_config.get("top_k", 40),
                repeat_penalty=gen_config.get("repeat_penalty", 1.1)
            )
        
        try:
            # プロンプト構築
            prompt = self.build_prompt(user_input, use_history)
            
            start_time = time.time()
            response = self.llm(
                prompt,
                max_tokens=generation_config.max_tokens,
                temperature=generation_config.temperature,
                top_p=generation_config.top_p,
                top_k=generation_config.top_k,
                repeat_penalty=generation_config.repeat_penalty,
                stop=generation_config.stop
            )
            
            inference_time = time.time() - start_time
            response_text = response["choices"][0]["text"].strip()
            
            # 会話履歴に追加
            if use_history:
                self.add_to_history("user", user_input)
                self.add_to_history("assistant", response_text)
            
            self.logger.info(f"Generated response in {inference_time:.2f}s: {response_text[:50]}...")
            return response_text
            
        except Exception as e:
            self.logger.error(f"Generation failed: {e}")
            raise
    
    def generate_response_stream(self,
                               user_input: str,
                               generation_config: Optional[GenerationConfig] = None,
                               use_history: bool = True) -> Generator[str, None, None]:
        """ストリーミング応答生成"""
        if generation_config is None:
            gen_config = self.config["generation"]
            generation_config = GenerationConfig(
                max_tokens=gen_config.get("max_tokens", 512),
                temperature=gen_config.get("temperature", 0.7),
                top_p=gen_config.get("top_p", 0.9),
                top_k=gen_config.get("top_k", 40),
                repeat_penalty=gen_config.get("repeat_penalty", 1.1)
            )
        
        prompt = self.build_prompt(user_input, use_history)
        
        try:
            response_tokens = []
            for token in self.llm(
                prompt,
                max_tokens=generation_config.max_tokens,
                temperature=generation_config.temperature,
                top_p=generation_config.top_p,
                top_k=generation_config.top_k,
                repeat_penalty=generation_config.repeat_penalty,
                stop=generation_config.stop,
                stream=True
            ):
                token_text = token["choices"][0]["text"]
                response_tokens.append(token_text)
                yield token_text
            
            # 完全な応答を履歴に追加
            if use_history:
                full_response = "".join(response_tokens)
                self.add_to_history("user", user_input)
                self.add_to_history("assistant", full_response)
                
        except Exception as e:
            self.logger.error(f"Streaming generation failed: {e}")
            raise
    
    def build_prompt(self, user_input: str, use_history: bool = True) -> str:
        """プロンプトを構築"""
        chat_config = self.config["chat"]
        system_prompt = chat_config["system_prompt"]
        
        prompt_parts = [system_prompt, "\n\n"]
        
        if use_history:
            # 最近の会話履歴（最大10メッセージ）
            recent_history = self.conversation_history[-10:]
            for msg in recent_history:
                if msg.role == "user":
                    prompt_parts.append(f"User: {msg.content}\n")
                elif msg.role == "assistant":
                    prompt_parts.append(f"Assistant: {msg.content}\n\n")
        
        prompt_parts.append(f"User: {user_input}\n")
        prompt_parts.append("Assistant: ")
        
        return "".join(prompt_parts)
    
    def add_to_history(self, role: str, content: str):
        """会話履歴に追加"""
        message = ChatMessage(
            role=role,
            content=content,
            timestamp=datetime.now().isoformat()
        )
        self.conversation_history.append(message)
    
    def clear_history(self):
        """会話履歴をクリア"""
        self.conversation_history.clear()
        self.logger.info("Conversation history cleared")
    
    def get_history(self) -> List[Dict]:
        """会話履歴を取得"""
        return [
            {
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp
            }
            for msg in self.conversation_history
        ]
    
    def set_system_prompt(self, system_prompt: str):
        """システムプロンプトを変更"""
        self.config["chat"]["system_prompt"] = system_prompt
        self.logger.info(f"System prompt updated: {system_prompt[:50]}...")
    
    def update_generation_config(self, **kwargs):
        """生成設定を更新"""
        for key, value in kwargs.items():
            if key in self.config["generation"]:
                self.config["generation"][key] = value
                self.logger.info(f"Generation config updated: {key} = {value}")
    
    def get_model_info(self) -> Dict:
        """モデル情報を取得"""
        model_config = self.config["model"]
        return {
            "model_path": os.path.basename(model_config["path"]),
            "context_size": model_config.get("n_ctx", 4096),
            "gpu_layers": model_config.get("n_gpu_layers", -1),
            "conversation_turns": len(self.conversation_history),
            "optimization": "FP16 + Low VRAM",
            "batch_size": model_config.get("n_batch", 256)
        }


def create_engine(config_path: Optional[str] = None) -> LLMEngine:
    """LLMエンジンを作成"""
    return LLMEngine(config_path or "/app/config/model_config.yaml")