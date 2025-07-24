#!/usr/bin/env python3
"""
GPU Redis Distributed Lock Implementation
SAASシステムのスケーラブルなGPU排他制御
"""

import redis
import time
import uuid
import asyncio
from typing import Optional, Union
from contextlib import asynccontextmanager

class GPURedisLock:
    """
    Redis分散ロックを使用したGPU排他制御
    複数サーバー/プロセス間でGPUアクセスを調整
    """
    
    def __init__(
        self, 
        redis_client: redis.Redis,
        lock_name: str = "gpu_lock",
        timeout: int = 300,  # 5分のタイムアウト
        retry_delay: float = 0.1,
        max_retries: int = 600  # 最大60秒待機
    ):
        self.redis = redis_client
        self.lock_name = lock_name
        self.timeout = timeout
        self.retry_delay = retry_delay
        self.max_retries = max_retries
        self.identifier = str(uuid.uuid4())
    
    def acquire(self) -> bool:
        """同期的にロックを取得"""
        end_time = time.time() + self.timeout
        
        for attempt in range(self.max_retries):
            # SET NX EX でアトミックにロック取得を試行
            if self.redis.set(
                self.lock_name, 
                self.identifier, 
                nx=True,  # 存在しない場合のみセット
                ex=self.timeout  # 自動期限切れ
            ):
                return True
            
            # リトライ待機
            time.sleep(self.retry_delay)
            
            # タイムアウトチェック
            if time.time() > end_time:
                raise TimeoutError(f"Failed to acquire lock '{self.lock_name}' after {self.timeout}s")
        
        return False
    
    def release(self) -> bool:
        """ロックを解放（所有者チェック付き）"""
        # Luaスクリプトでアトミックに所有者確認と削除
        lua_script = """
        if redis.call("get", KEYS[1]) == ARGV[1] then
            return redis.call("del", KEYS[1])
        else
            return 0
        end
        """
        
        result = self.redis.eval(lua_script, 1, self.lock_name, self.identifier)
        return bool(result)
    
    @asynccontextmanager
    async def async_acquire(self):
        """非同期コンテキストマネージャー"""
        # 非同期でロック取得を待機
        loop = asyncio.get_event_loop()
        acquired = False
        
        try:
            # ブロッキング処理を別スレッドで実行
            acquired = await loop.run_in_executor(None, self.acquire)
            if not acquired:
                raise RuntimeError(f"Could not acquire lock '{self.lock_name}'")
            yield
        finally:
            if acquired:
                await loop.run_in_executor(None, self.release)


class GPUResourceManager:
    """
    GPU リソース管理クラス
    複数GPUや優先度管理に対応
    """
    
    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        self.redis = redis.from_url(redis_url)
        self.gpu_locks = {}
        
    def get_gpu_lock(
        self, 
        gpu_id: Union[int, str] = "gpu_0",
        timeout: int = 300,
        priority: int = 0  # 将来の優先度実装用
    ) -> GPURedisLock:
        """指定GPUのロックオブジェクトを取得"""
        lock_name = f"gpu_lock:{gpu_id}"
        
        if lock_name not in self.gpu_locks:
            self.gpu_locks[lock_name] = GPURedisLock(
                self.redis,
                lock_name=lock_name,
                timeout=timeout
            )
        
        return self.gpu_locks[lock_name]
    
    async def get_usage_stats(self) -> dict:
        """GPU使用統計を取得"""
        stats = {
            "active_locks": [],
            "total_gpus": 1,  # 現在は1GPU想定
            "available_gpus": 0
        }
        
        # アクティブなロックを確認
        for gpu_id in range(stats["total_gpus"]):
            lock_name = f"gpu_lock:gpu_{gpu_id}"
            if self.redis.exists(lock_name):
                ttl = self.redis.ttl(lock_name)
                stats["active_locks"].append({
                    "gpu_id": gpu_id,
                    "lock_name": lock_name,
                    "ttl_seconds": ttl
                })
            else:
                stats["available_gpus"] += 1
        
        return stats


# FastAPI統合用のシングルトンインスタンス
gpu_manager = None

def init_gpu_manager(redis_url: str = "redis://localhost:6379/0"):
    """GPU管理システムの初期化"""
    global gpu_manager
    gpu_manager = GPUResourceManager(redis_url)
    return gpu_manager


# 使用例
async def example_usage():
    """使用例"""
    # 初期化
    manager = init_gpu_manager("redis://localhost:6379/0")
    
    # GPUロックを取得して処理
    gpu_lock = manager.get_gpu_lock(gpu_id="gpu_0")
    
    async with gpu_lock.async_acquire():
        print("GPU locked! Processing...")
        # ここでGPU処理を実行
        await asyncio.sleep(2)  # 仮の処理
        print("GPU processing completed!")
    
    # 統計情報取得
    stats = await manager.get_usage_stats()
    print(f"GPU Stats: {stats}")


if __name__ == "__main__":
    asyncio.run(example_usage())