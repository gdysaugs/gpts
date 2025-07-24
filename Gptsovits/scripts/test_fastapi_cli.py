#!/usr/bin/env python3
"""
GPT-SoVITS FastAPI テストCLI
reference_5sec.wavの声質で指定したテキストを読み上げる
"""

import argparse
import requests
import time
import os
import sys
from datetime import datetime
from pathlib import Path

def test_voice_generation(text: str, output_dir: str = "./output", api_url: str = "http://localhost:8000", save_filename: str = None):
    """
    FastAPIサーバーで音声生成をテスト
    
    Args:
        text: 読み上げたいテキスト
        output_dir: 出力ディレクトリ
        api_url: FastAPIサーバーのURL
        save_filename: 保存ファイル名（指定しない場合は自動生成）
    """
    print(f"\n🎯 GPT-SoVITS FastAPIテストCLI")
    print(f"📝 テキスト: {text}")
    print(f"🌐 API URL: {api_url}")
    
    # 出力ディレクトリ作成
    os.makedirs(output_dir, exist_ok=True)
    
    # ヘルスチェック
    try:
        health_response = requests.get(f"{api_url}/")
        health_data = health_response.json()
        print(f"\n✅ サーバー状態:")
        print(f"   - モデル読み込み済み: {health_data.get('models_loaded', False)}")
        print(f"   - GPU利用可能: {health_data.get('gpu_available', False)}")
        print(f"   - GPU名: {health_data.get('gpu_name', 'N/A')}")
    except Exception as e:
        print(f"\n❌ サーバーに接続できません: {e}")
        print(f"💡 ヒント: FastAPIサーバーを起動してください:")
        print(f"   cd /home/adama/gpts/Gptsovits")
        print(f"   docker run --gpus all -d -p 8000:8000 --name gpt-sovits-api \\")
        print(f"     --privileged -v /usr/lib/wsl:/usr/lib/wsl -e LD_LIBRARY_PATH=/usr/lib/wsl/lib \\")
        print(f"     -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output \\")
        print(f"     -v $(pwd)/scripts:/app/scripts \\")
        print(f"     -v $(pwd)/models/v4/GPT-SoVITS/gpt-sovits-ja-h:/app/GPT_SoVITS/pretrained_models/gpt-sovits-ja-h \\")
        print(f"     gpt-sovits:v4 bash -c \"pip install fastapi uvicorn python-multipart && python /app/scripts/fastapi_voice_server.py\"")
        return
    
    # 音声生成リクエスト
    print(f"\n🎤 音声生成開始...")
    start_time = time.time()
    
    try:
        # URLエンコードされたパラメータでGETリクエスト
        params = {
            "ref_text": "おはようございます",  # reference_5sec.wavの内容に合わせて変更
            "target_text": text,
            "temperature": 1.0
        }
        
        response = requests.get(
            f"{api_url}/clone-voice-simple",
            params=params,
            stream=True
        )
        
        if response.status_code == 200:
            # 音声データを保存
            if save_filename:
                output_path = os.path.join(output_dir, save_filename)
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                safe_text = text.replace(' ', '_').replace('！', '').replace('？', '')[:30]
                output_path = os.path.join(output_dir, f"cli_test_{timestamp}_{safe_text}.wav")
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            generation_time = time.time() - start_time
            file_size = os.path.getsize(output_path)
            
            print(f"\n✅ 音声生成成功!")
            print(f"📊 統計:")
            print(f"   - 生成時間: {generation_time:.2f}秒")
            print(f"   - ファイルサイズ: {file_size / 1024:.1f} KB")
            print(f"   - 保存先: {output_path}")
            
            # 音声ファイルの情報を表示（waveモジュールがある場合）
            try:
                import wave
                with wave.open(output_path, 'rb') as wav_file:
                    frames = wav_file.getnframes()
                    rate = wav_file.getframerate()
                    duration = frames / float(rate)
                    print(f"   - 音声長: {duration:.2f}秒")
                    print(f"   - サンプリングレート: {rate} Hz")
                    print(f"   - リアルタイム係数: {duration / generation_time:.2f}x")
            except ImportError:
                pass
            
        else:
            print(f"\n❌ エラー: HTTPステータス {response.status_code}")
            print(f"レスポンス: {response.text}")
            
    except Exception as e:
        print(f"\n❌ リクエストエラー: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="GPT-SoVITS FastAPI音声生成テストCLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # 基本的な使用
  python test_fastapi_cli.py "こんにちは、音声生成のテストです"
  
  # 長文テスト
  python test_fastapi_cli.py "今日のAI技術はMachine LearningやDeep Learningの進歩により、革命的な変化をもたらしています。"
  
  # 感情表現テスト
  python test_fastapi_cli.py "わあああ！すごい！本当に素晴らしい結果です！"
  
  # 出力ディレクトリとファイル名指定
  python test_fastapi_cli.py "テスト音声" --output-dir ./my_output --filename test.wav
  
  # 別のサーバーに接続
  python test_fastapi_cli.py "テスト" --api-url http://192.168.1.100:8000
        """
    )
    
    parser.add_argument("text", help="読み上げたいテキスト")
    parser.add_argument("--output-dir", default="./output", help="出力ディレクトリ（デフォルト: ./output）")
    parser.add_argument("--filename", help="保存ファイル名（指定しない場合は自動生成）")
    parser.add_argument("--api-url", default="http://localhost:8000", help="FastAPIサーバーのURL（デフォルト: http://localhost:8000）")
    
    args = parser.parse_args()
    
    # テスト実行
    test_voice_generation(
        text=args.text,
        output_dir=args.output_dir,
        api_url=args.api_url,
        save_filename=args.filename
    )

if __name__ == "__main__":
    main()