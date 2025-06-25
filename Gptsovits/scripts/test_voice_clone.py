#!/usr/bin/env python3
"""
GPT-SoVITS CLI テストスクリプト
音声クローニングのCLI実行用
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import torch
import soundfile as sf

# GPT-SoVITSのパスを追加
import os
os.chdir('/app')  # 作業ディレクトリを変更
sys.path.insert(0, '/app')
sys.path.insert(0, '/app/GPT_SoVITS')

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/app/logs/test_voice_clone.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def check_gpu():
    """GPU利用可能性をチェック"""
    if torch.cuda.is_available():
        logger.info(f"GPU利用可能: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA バージョン: {torch.version.cuda}")
        return True
    else:
        logger.warning("GPUが利用できません。CPUモードで実行します。")
        return False

def load_models():
    """モデルをロード"""
    # モデルパス設定 - 日本語特化モデルを使用
    sovits_path = "/app/GPT_SoVITS/pretrained_models/gpt-sovits-ja-h/hscene-e17.ckpt"
    # GPTモデルは既存のv2モデルを使用（互換性のため）
    gpt_path = "/app/GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth"
    
    # モデルの存在確認
    if not os.path.exists(sovits_path):
        raise FileNotFoundError(f"SoVITSモデルが見つかりません: {sovits_path}")
    
    logger.info("モデルをロード中...")
    
    try:
        from GPT_SoVITS.inference_cli import GPTSoVITSInference
        
        inference = GPTSoVITSInference(
            sovits_path=sovits_path,
            gpt_path=gpt_path,
            device="cuda" if torch.cuda.is_available() else "cpu",
            is_half=os.getenv("IS_HALF", "True").lower() == "true"
        )
        logger.info("モデルのロード完了")
        return inference
    
    except (ImportError, Exception) as e:
        logger.error(f"GPT-SoVITSモジュールのインポートに失敗しました: {e}")
        logger.info("代替方法で実行を試みます...")
        
        # 代替実装 - inference_webuiを直接使用
        from GPT_SoVITS.inference_webui import change_gpt_weights, change_sovits_weights
        
        change_sovits_weights(sovits_path)
        # change_gpt_weights(gpt_path)  # v4では不要の場合がある
        
        logger.info("代替方法でのモデルロード完了")
        return None

def process_voice_clone(inference, ref_audio_path, ref_text, target_text, output_path):
    """音声クローニング処理"""
    try:
        logger.info(f"参照音声: {ref_audio_path}")
        logger.info(f"参照テキスト: {ref_text}")
        logger.info(f"ターゲットテキスト: {target_text}")
        
        if inference:
            # 推論実行
            result = inference.get_tts_wav(
                ref_wav_path=ref_audio_path,
                prompt_text=ref_text,
                prompt_language="ja",
                text=target_text,
                text_language="ja"
            )
        else:
            # 代替方法での実行
            from GPT_SoVITS.inference_webui import get_tts_wav
            
            # GPT-SoVITSのget_tts_wav関数の正しい引数で呼び出し
            result = get_tts_wav(
                ref_wav_path=ref_audio_path,
                prompt_text=ref_text,
                prompt_language="Japanese",  # キー名を使用
                text=target_text,
                text_language="Japanese",    # キー名を使用
                how_to_cut="不切",              # 分割なし - 全文を一度に生成
                top_k=20,                    # より多様性を増加
                top_p=1.0,                   # より完全な生成を強制
                temperature=1.0,             # より完全な音声生成
                ref_free=True               # 参照音声の制約を緩和
            )
        
        # 結果を保存（ジェネレーターの処理）
        if result:
            try:
                logger.info(f"結果の型: {type(result)}")
                
                # ジェネレーターの場合は1つずつ取得して直接処理
                if hasattr(result, '__iter__') and not isinstance(result, (list, tuple, str)):
                    logger.info("ジェネレーターから結果を取得中...")
                    
                    # 音声データを探して連結
                    import numpy as np
                    audio_segments = []
                    sample_rate = None
                    
                    # ジェネレーターから直接音声データを取得
                    for i, item in enumerate(result):
                        logger.info(f"アイテム {i}: {type(item)}")
                        if isinstance(item, tuple) and len(item) >= 2:
                            sr, audio = item[0], item[1]
                            if isinstance(audio, np.ndarray) and len(audio) > 0:
                                logger.info(f"セグメント {i}: {sr}Hz, {len(audio)} samples, {len(audio)/sr:.2f}秒")
                                audio_segments.append(audio)
                                if sample_rate is None:
                                    sample_rate = sr
                    
                    if audio_segments and sample_rate:
                        # 全セグメントを連結
                        final_audio = np.concatenate(audio_segments)
                        
                        logger.info(f"最終音声: サンプリングレート {sample_rate}Hz")
                        logger.info(f"最終音声データ長: {len(final_audio)} samples")
                        logger.info(f"最終音声時間: {len(final_audio)/sample_rate:.2f}秒")
                        logger.info(f"セグメント数: {len(audio_segments)}")
                        
                        # 音声データの詳細分析
                        logger.info(f"音声データ統計: max={np.max(final_audio):.6f}, min={np.min(final_audio):.6f}, mean={np.mean(final_audio):.6f}")
                        logger.info(f"音声RMS: {np.sqrt(np.mean(final_audio**2)):.6f}")
                        
                        # 無音チェック
                        silence_threshold = 0.001
                        non_silent_samples = np.sum(np.abs(final_audio) > silence_threshold)
                        logger.info(f"非無音サンプル: {non_silent_samples}/{len(final_audio)} ({non_silent_samples/len(final_audio)*100:.1f}%)")
                        
                        sf.write(output_path, final_audio, sample_rate)
                        logger.info(f"音声生成完了: {output_path}")
                        return True
                    
                    logger.error("有効な音声データが見つかりません")
                    logger.error(f"取得したアイテム: {[type(item) for item in result_items]}")
                    return False
                    
                elif isinstance(result, (list, tuple)) and len(result) >= 2:
                    sr, audio = result[0], result[1]
                    import numpy as np
                    if not isinstance(audio, np.ndarray):
                        audio = np.array(audio)
                    
                    sf.write(output_path, audio, sr)
                    logger.info(f"音声生成完了: {output_path}")
                    return True
                else:
                    logger.error(f"音声生成結果の形式が不正です: {type(result)}")
                    return False
                    
            except Exception as save_error:
                logger.error(f"音声保存でエラー: {save_error}")
                import traceback
                logger.error(traceback.format_exc())
                return False
        else:
            logger.error("音声生成に失敗しました")
            return False
            
    except Exception as e:
        logger.error(f"処理中にエラーが発生: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    parser = argparse.ArgumentParser(description="GPT-SoVITS CLI音声クローニング")
    parser.add_argument("--ref-audio", type=str, required=True, help="参照音声ファイルパス")
    parser.add_argument("--ref-text", type=str, required=True, help="参照音声のテキスト")
    parser.add_argument("--target-text", type=str, required=True, help="生成したい音声のテキスト")
    parser.add_argument("--output", type=str, default="/app/output/result.wav", help="出力ファイルパス")
    parser.add_argument("--language", type=str, default="ja", choices=["ja", "en", "zh", "ko"], help="言語設定")
    
    args = parser.parse_args()
    
    # GPU確認
    check_gpu()
    
    # 入力ファイルの存在確認
    if not os.path.exists(args.ref_audio):
        logger.error(f"参照音声ファイルが見つかりません: {args.ref_audio}")
        sys.exit(1)
    
    # 出力ディレクトリ作成
    output_dir = os.path.dirname(args.output)
    os.makedirs(output_dir, exist_ok=True)
    
    # モデルロード
    inference = load_models()
    
    # 音声クローニング実行
    success = process_voice_clone(
        inference,
        args.ref_audio,
        args.ref_text,
        args.target_text,
        args.output
    )
    
    if success:
        logger.info("処理が正常に完了しました")
        sys.exit(0)
    else:
        logger.error("処理に失敗しました")
        sys.exit(1)

if __name__ == "__main__":
    main()