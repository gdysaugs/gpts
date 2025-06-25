#!/usr/bin/env python3
"""
異なる長さのテキストでの音声生成テスト
短い文と長い文の両方で問題を検証
"""

import os
import sys
import subprocess
import logging

# ロギング設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_audio_generation():
    """異なる長さのテキストで音声生成をテスト"""
    
    # テストケース
    test_cases = [
        {
            "name": "短い文",
            "text": "こんにちは。",
            "output": "/app/output/test_short.wav"
        },
        {
            "name": "中程度の文", 
            "text": "こんにちは、これはテスト音声です。今日はいい天気ですね。",
            "output": "/app/output/test_medium.wav"
        },
        {
            "name": "長い文",
            "text": "こんにちは、これは音声クローニングのテストです。今日はとても良い天気で、空が青く澄んでいます。このような日は散歩をするのにとても適していて、多くの人が公園や街を歩いているのを見かけます。技術の進歩により、このような高品質な音声合成が可能になりました。",
            "output": "/app/output/test_long.wav"
        }
    ]
    
    ref_audio = "/app/input/reference_5sec.wav"
    ref_text = "おはようございます"
    
    logger.info("=== 音声生成テスト開始 ===")
    
    for test_case in test_cases:
        logger.info(f"\n--- {test_case['name']}のテスト ---")
        logger.info(f"テキスト: {test_case['text']}")
        logger.info(f"文字数: {len(test_case['text'])}文字")
        
        # test_voice_clone.pyを実行
        cmd = [
            "python", "/app/scripts/test_voice_clone.py",
            "--ref-audio", ref_audio,
            "--ref-text", ref_text,
            "--target-text", test_case['text'],
            "--output", test_case['output']
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0:
                logger.info(f"✓ {test_case['name']}: 生成成功")
                
                # ファイルサイズと音声長をチェック
                if os.path.exists(test_case['output']):
                    file_size = os.path.getsize(test_case['output'])
                    logger.info(f"ファイルサイズ: {file_size} bytes")
                    
                    # soundfileで音声長を取得
                    try:
                        import soundfile as sf
                        data, sr = sf.read(test_case['output'])
                        duration = len(data) / sr
                        logger.info(f"音声時間: {duration:.2f}秒")
                        
                        # 短すぎる音声をチェック
                        if duration < 0.5:
                            logger.warning(f"⚠️ 音声が短すぎます: {duration:.2f}秒")
                        elif duration > 0.5:
                            logger.info(f"✓ 適切な音声長: {duration:.2f}秒")
                            
                    except Exception as e:
                        logger.error(f"音声ファイル分析エラー: {e}")
                else:
                    logger.error(f"✗ 出力ファイルが生成されませんでした")
            else:
                logger.error(f"✗ {test_case['name']}: 生成失敗")
                logger.error(f"エラー出力: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            logger.error(f"✗ {test_case['name']}: タイムアウト")
        except Exception as e:
            logger.error(f"✗ {test_case['name']}: 実行エラー - {e}")
    
    logger.info("\n=== テスト完了 ===")

if __name__ == "__main__":
    test_audio_generation()