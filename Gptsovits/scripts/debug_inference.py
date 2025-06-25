#!/usr/bin/env python3
"""
GPT-SoVITS推論デバッグスクリプト
テキスト処理とセグメント生成の詳細を調査
"""

import os
import sys
import logging

# GPT-SoVITSのパスを追加
os.chdir('/app')
sys.path.insert(0, '/app')
sys.path.insert(0, '/app/GPT_SoVITS')

# 詳細ログ設定
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def debug_text_processing():
    """テキスト処理の詳細をデバッグ"""
    print("=== テキスト処理デバッグ開始 ===")
    
    # clean_text関数を直接テスト
    from GPT_SoVITS.text.cleaner import clean_text
    
    test_text = "こんにちは皆さん、今日はとても良い天気ですね"
    print(f"入力テキスト: {test_text}")
    
    try:
        phones, word2ph, norm_text = clean_text(test_text, "Japanese", "v2")
        print(f"正規化テキスト: {norm_text}")
        print(f"音素数: {len(phones)}")
        print(f"音素: {phones[:20]}...")  # 最初の20個のみ表示
        print(f"word2ph: {word2ph}")
    except Exception as e:
        print(f"clean_text エラー: {e}")
        import traceback
        traceback.print_exc()

def debug_sentence_segmentation():
    """文章分割のデバッグ"""
    print("\n=== 文章分割デバッグ ===")
    
    from GPT_SoVITS.text import chinese
    
    test_text = "こんにちは皆さん、今日はとても良い天気ですね"
    print(f"入力テキスト: {test_text}")
    
    # 分割方法をテスト
    try:
        # how_to_cut="不切"の場合の処理を確認
        sentences = [test_text]  # 分割なし
        print(f"分割結果 (不切): {sentences}")
        
        # how_to_cut="按标点符号切"の場合
        import re
        sentences_punct = re.split(r'[。！？、，]', test_text)
        sentences_punct = [s.strip() for s in sentences_punct if s.strip()]
        print(f"分割結果 (按标点符号切): {sentences_punct}")
        
    except Exception as e:
        print(f"文章分割エラー: {e}")

def debug_full_inference():
    """完全な推論プロセスをデバッグ"""
    print("\n=== 完全推論デバッグ ===")
    
    from GPT_SoVITS.inference_webui import get_tts_wav
    
    # モデルロード状況確認
    print("モデル状況確認中...")
    
    try:
        # 短いテストから開始
        test_cases = [
            "こんにちは",
            "こんにちは皆さん",
            "こんにちは皆さん、今日はとても良い天気ですね"
        ]
        
        for i, text in enumerate(test_cases):
            print(f"\n--- テストケース {i+1}: {text} ---")
            
            result = get_tts_wav(
                ref_wav_path='/app/input/reference_5sec.wav',
                prompt_text='おはようございます',
                prompt_language='Japanese',
                text=text,
                text_language='Japanese',
                how_to_cut='不切',
                top_k=15,
                top_p=0.8,
                temperature=0.7,
                ref_free=False
            )
            
            print(f"結果の型: {type(result)}")
            
            # ジェネレーターの内容を確認
            segments = []
            for j, item in enumerate(result):
                print(f"アイテム {j}: {type(item)}")
                if isinstance(item, tuple) and len(item) >= 2:
                    sr, audio = item[0], item[1]
                    print(f"  サンプリングレート: {sr}Hz")
                    print(f"  音声長: {len(audio)} samples ({len(audio)/sr:.2f}秒)")
                    segments.append((sr, audio))
                else:
                    print(f"  予期しない形式: {item}")
            
            print(f"総セグメント数: {len(segments)}")
            
    except Exception as e:
        print(f"推論エラー: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_text_processing()
    debug_sentence_segmentation()
    debug_full_inference()