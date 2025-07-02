#!/usr/bin/env python3
"""
API Test Script
FastAPI LLMサーバーのテスト
"""

import asyncio
import httpx
import json
import time
from typing import Dict, Any

class APITester:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=60.0)
    
    async def test_health(self) -> Dict[str, Any]:
        """ヘルスチェックテスト"""
        print("🏥 Testing health endpoint...")
        response = await self.client.get(f"{self.base_url}/health")
        result = response.json()
        print(f"✅ Health: {result}")
        return result
    
    async def test_model_info(self) -> Dict[str, Any]:
        """モデル情報テスト"""
        print("🤖 Testing model info endpoint...")
        response = await self.client.get(f"{self.base_url}/model-info")
        result = response.json()
        print(f"✅ Model Info: {result}")
        return result
    
    async def test_chat(self) -> Dict[str, Any]:
        """チャットテスト"""
        print("💬 Testing chat endpoint...")
        
        chat_request = {
            "message": "こんにちは！調子はどう？",
            "use_history": True,
            "stream": False
        }
        
        start_time = time.time()
        response = await self.client.post(
            f"{self.base_url}/chat",
            json=chat_request
        )
        api_time = time.time() - start_time
        
        result = response.json()
        print(f"✅ Chat Response: {result['response']}")
        print(f"⏱️ API Time: {api_time:.2f}s, Inference Time: {result.get('inference_time', 'N/A')}s")
        return result
    
    async def test_chat_stream(self) -> None:
        """ストリーミングチャットテスト"""
        print("🌊 Testing streaming chat endpoint...")
        
        chat_request = {
            "message": "短い詩を作って！",
            "use_history": True,
            "stream": True
        }
        
        print("📝 Streaming response:")
        async with self.client.stream(
            "POST",
            f"{self.base_url}/chat/stream",
            json=chat_request
        ) as response:
            response_text = ""
            async for chunk in response.aiter_text():
                if chunk.startswith("data: "):
                    data_str = chunk[6:].strip()
                    if data_str:
                        try:
                            data = json.loads(data_str)
                            if "token" in data:
                                print(data["token"], end="", flush=True)
                                response_text += data["token"]
                            elif "done" in data:
                                print("\n✅ Streaming complete!")
                                break
                            elif "error" in data:
                                print(f"\n❌ Streaming error: {data['error']}")
                                break
                        except json.JSONDecodeError:
                            continue
        
        return {"response": response_text}
    
    async def test_system_prompt(self) -> Dict[str, Any]:
        """システムプロンプトテスト"""
        print("🎭 Testing system prompt endpoints...")
        
        # ツンデレプリセット設定
        response = await self.client.post(f"{self.base_url}/presets/tsundere")
        result1 = response.json()
        print(f"✅ Tsundere preset: {result1}")
        
        # ツンデレで会話テスト
        chat_response = await self.client.post(
            f"{self.base_url}/chat",
            json={"message": "手伝ってくれる？", "use_history": False}
        )
        tsundere_result = chat_response.json()
        print(f"✅ Tsundere response: {tsundere_result['response']}")
        
        return {"preset_result": result1, "chat_result": tsundere_result}
    
    async def test_generation_config(self) -> Dict[str, Any]:
        """生成設定テスト"""
        print("⚙️ Testing generation config endpoint...")
        
        config_request = {
            "temperature": 0.9,
            "top_p": 0.8,
            "max_tokens": 100
        }
        
        response = await self.client.post(
            f"{self.base_url}/generation-config",
            json=config_request
        )
        result = response.json()
        print(f"✅ Config update: {result}")
        
        # 設定変更後の会話テスト
        chat_response = await self.client.post(
            f"{self.base_url}/chat",
            json={"message": "創造的な話をして！", "use_history": False}
        )
        chat_result = chat_response.json()
        print(f"✅ High creativity response: {chat_result['response'][:100]}...")
        
        return {"config_result": result, "chat_result": chat_result}
    
    async def test_history(self) -> Dict[str, Any]:
        """履歴管理テスト"""
        print("📚 Testing history endpoints...")
        
        # 履歴確認
        response = await self.client.get(f"{self.base_url}/history")
        history_result = response.json()
        print(f"✅ Current history: {history_result['count']} messages")
        
        # 履歴クリア
        clear_response = await self.client.delete(f"{self.base_url}/history")
        clear_result = clear_response.json()
        print(f"✅ History cleared: {clear_result}")
        
        # クリア後の履歴確認
        response = await self.client.get(f"{self.base_url}/history")
        final_history = response.json()
        print(f"✅ History after clear: {final_history['count']} messages")
        
        return {"history": history_result, "clear": clear_result, "final": final_history}
    
    async def run_all_tests(self):
        """全テスト実行"""
        print("🚀 Starting API Tests")
        print("=" * 50)
        
        try:
            # 基本テスト
            await self.test_health()
            await self.test_model_info()
            
            # チャット機能テスト
            await self.test_chat()
            await self.test_chat_stream()
            
            # 高度な機能テスト
            await self.test_system_prompt()
            await self.test_generation_config()
            await self.test_history()
            
            print("\n" + "=" * 50)
            print("🎉 All tests completed successfully!")
            
        except Exception as e:
            print(f"\n❌ Test failed: {e}")
            raise
        finally:
            await self.client.aclose()

async def main():
    """メイン関数"""
    print("🧪 FastAPI LLM Server Test Suite")
    print("Make sure the server is running on http://localhost:8000")
    print()
    
    tester = APITester()
    await tester.run_all_tests()

if __name__ == "__main__":
    asyncio.run(main())