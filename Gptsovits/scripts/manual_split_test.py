#!/usr/bin/env python3
"""
手動テキスト分割テスト
確実に全文を読ませるために手動で分割して連結
"""

import os
import sys
import argparse
import logging
import numpy as np
import soundfile as sf

# GPT-SoVITSのパスを追加
os.chdir('/app')
sys.path.insert(0, '/app')
sys.path.insert(0, '/app/GPT_SoVITS')

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/app/logs/manual_split_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def generate_single_segment(ref_audio_path, ref_text, target_text):
    """単一セグメントの音声生成"""
    try:
        from GPT_SoVITS.inference_webui import get_tts_wav
        
        logger.info(f"セグメント生成: '{target_text}'")
        
        result = get_tts_wav(
            ref_wav_path=ref_audio_path,
            prompt_text=ref_text,
            prompt_language="Japanese",
            text=target_text,
            text_language="Japanese",
            how_to_cut="不切",  # 分割しない（短い文なので）
            top_k=15,
            top_p=0.8,
            temperature=0.7,
            ref_free=False
        )
        
        # ジェネレーターから音声データを取得
        for item in result:
            if isinstance(item, tuple) and len(item) >= 2:
                sr, audio = item[0], item[1]
                if isinstance(audio, np.ndarray) and len(audio) > 0:
                    logger.info(f"セグメント音声: {sr}Hz, {len(audio)} samples, {len(audio)/sr:.2f}秒")
                    return sr, audio
        
        logger.error("有効な音声データが取得できませんでした")
        return None, None
        
    except Exception as e:
        logger.error(f"音声生成エラー: {e}")
        return None, None

def main():
    parser = argparse.ArgumentParser(description="手動テキスト分割音声生成")
    parser.add_argument("--ref-audio", type=str, required=True, help="参照音声ファイルパス")
    parser.add_argument("--ref-text", type=str, required=True, help="参照音声のテキスト")
    parser.add_argument("--target-text", type=str, required=True, help="生成したい音声のテキスト")
    parser.add_argument("--output", type=str, default="/app/output/manual_split.wav", help="出力ファイルパス")
    
    args = parser.parse_args()
    
    # 入力ファイルの存在確認
    if not os.path.exists(args.ref_audio):
        logger.error(f"参照音声ファイルが見つかりません: {args.ref_audio}")
        sys.exit(1)
    
    # 出力ディレクトリ作成
    output_dir = os.path.dirname(args.output)
    os.makedirs(output_dir, exist_ok=True)
    
    # テキストを手動で分割
    # 「こんにちは、これはテスト音声です」→「こんにちは」「これはテスト音声です」
    if "こんにちは" in args.target_text and "テスト音声" in args.target_text:
        segments = ["こんにちは", "これはテスト音声です"]
    else:
        # 一般的な分割: 句読点で分割
        segments = [s.strip() for s in args.target_text.replace('、', ',').split(',') if s.strip()]
    
    logger.info(f"テキストを{len(segments)}個のセグメントに分割: {segments}")
    
    # 各セグメントの音声生成
    audio_segments = []
    sample_rate = None
    
    for i, segment in enumerate(segments):
        logger.info(f"セグメント {i+1}/{len(segments)} を処理中: '{segment}'")
        
        sr, audio = generate_single_segment(args.ref_audio, args.ref_text, segment)
        
        if sr is not None and audio is not None:
            audio_segments.append(audio)
            if sample_rate is None:
                sample_rate = sr
            logger.info(f"セグメント {i+1} 生成成功: {len(audio)} samples")
        else:
            logger.error(f"セグメント {i+1} 生成失敗")
            sys.exit(1)
    
    if audio_segments and sample_rate:
        # 全セグメントを連結（無音を挟まず直接連結）
        final_audio = np.concatenate(audio_segments)
        
        # 音量正規化（重要！）
        # 音声レベルが低すぎる場合は正規化
        max_val = np.max(np.abs(final_audio))
        if max_val > 0 and max_val < 1000:  # 音声が小さすぎる場合
            # 適切なレベル（16bit音声の範囲）に正規化
            target_max = 10000  # 安全な最大値
            final_audio = final_audio * (target_max / max_val)
            logger.info(f"音量正規化: {max_val:.1f} → {np.max(np.abs(final_audio)):.1f}")
        
        logger.info(f"=== 最終結果 ===")
        logger.info(f"セグメント数: {len(audio_segments)}")
        logger.info(f"サンプリングレート: {sample_rate}Hz")
        logger.info(f"最終音声長: {len(final_audio)} samples")
        logger.info(f"最終音声時間: {len(final_audio)/sample_rate:.2f}秒")
        
        # 音声統計
        logger.info(f"音声データ統計: max={np.max(final_audio):.6f}, min={np.min(final_audio):.6f}")
        logger.info(f"音声RMS: {np.sqrt(np.mean(final_audio**2)):.6f}")
        
        # 無音チェック
        silence_threshold = 0.001
        non_silent_samples = np.sum(np.abs(final_audio) > silence_threshold)
        logger.info(f"非無音サンプル: {non_silent_samples}/{len(final_audio)} ({non_silent_samples/len(final_audio)*100:.1f}%)")
        
        # 保存
        sf.write(args.output, final_audio, sample_rate)
        logger.info(f"音声生成完了: {args.output}")
        logger.info("処理が正常に完了しました")
    else:
        logger.error("音声セグメントの生成に失敗しました")
        sys.exit(1)

if __name__ == "__main__":
    main()