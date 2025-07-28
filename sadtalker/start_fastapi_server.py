#!/usr/bin/env python3
"""
SadTalker FastAPI Server 起動スクリプト
Docker環境での実行専用
"""

import subprocess
import sys
import os
from pathlib import Path

def check_environment():
    """環境チェック"""
    print("🔍 環境チェック中...")
    
    # 必要ディレクトリの確認
    required_dirs = ["checkpoints", "gfpgan", "src"]
    for dir_name in required_dirs:
        if not Path(dir_name).exists():
            print(f"❌ 必要ディレクトリが見つかりません: {dir_name}")
            return False
    
    # 必要ファイルの確認
    required_files = ["sadtalker_engine.py", "fastapi_server.py"]
    for file_name in required_files:
        if not Path(file_name).exists():
            print(f"❌ 必要ファイルが見つかりません: {file_name}")
            return False
    
    print("✅ 環境チェック完了")
    return True

def start_docker_server():
    """Docker環境でFastAPIサーバー起動"""
    print("🐳 DockerでFastAPIサーバー起動中...")
    
    # 出力ディレクトリ作成
    os.makedirs("output", exist_ok=True)
    
    docker_cmd = [
        "docker", "run", "--rm",
        "--privileged",
        "-v", "/usr/lib/wsl:/usr/lib/wsl",
        "-e", "LD_LIBRARY_PATH=/usr/lib/wsl/lib",
        "-e", "NVIDIA_VISIBLE_DEVICES=all",
        "-e", "CUDA_VISIBLE_DEVICES=0",
        "-p", "8000:8000",  # FastAPIポート
        "-v", f"{os.getcwd()}:/app",
        "-w", "/app",
        "sadtalker:latest",
        "bash", "-c", 
        "pip install fastapi uvicorn python-multipart aiofiles && python fastapi_server.py"
    ]
    
    print("🚀 FastAPIサーバー起動コマンド:")
    print(" ".join(docker_cmd))
    print("\n📡 起動後のアクセスURL:")
    print("   WebUI: http://localhost:8000")
    print("   API Docs: http://localhost:8000/docs")
    print("   Status: http://localhost:8000/status")
    print("\n⚠️ 初回起動時はモデルロードで1-2分かかります")
    print("   モデルロード完了後は超高速処理が可能です\n")
    
    try:
        subprocess.run(docker_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Dockerサーバー起動エラー: {e}")
        return False
    except KeyboardInterrupt:
        print("\n🔄 サーバー停止中...")
        return True
    
    return True

def start_native_server():
    """ネイティブ環境でFastAPIサーバー起動（開発用）"""
    print("🐍 ネイティブ環境でFastAPIサーバー起動中...")
    print("⚠️ この方法はSadTalkerモジュールが利用できないため、開発・テスト専用です")
    
    try:
        cmd = ["python", "-m", "uvicorn", "fastapi_server:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ ネイティブサーバー起動エラー: {e}")
        return False
    except KeyboardInterrupt:
        print("\n🔄 サーバー停止中...")
        return True
    
    return True

def main():
    """メイン関数"""
    print("🎭 SadTalker FastAPIサーバー起動スクリプト")
    print("=" * 50)
    
    if not check_environment():
        print("❌ 環境チェックに失敗しました")
        sys.exit(1)
    
    print("\n起動方法を選択してください:")
    print("1. Docker環境で起動（推奨・本格運用）")
    print("2. ネイティブ環境で起動（開発・テスト用）")
    
    try:
        choice = input("\n選択 (1/2): ").strip()
        
        if choice == "1":
            start_docker_server()
        elif choice == "2":
            start_native_server()
        else:
            print("❌ 無効な選択です")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n🔄 起動をキャンセルしました")
        sys.exit(0)

if __name__ == "__main__":
    main()