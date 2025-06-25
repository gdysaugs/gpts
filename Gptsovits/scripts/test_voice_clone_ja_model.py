#!/usr/bin/env python3
"""
GPT-SoVITS CLI テストスクリプト - 日本語特化モデル対応版
change_sovits_weights関数をモンキーパッチで上書き
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
os.chdir('/app')
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

# グローバル変数（モデルパスを保存）
CUSTOM_SOVITS_PATH = None

def check_gpu():
    """GPU利用可能性をチェック"""
    if torch.cuda.is_available():
        logger.info(f"GPU利用可能: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA バージョン: {torch.version.cuda}")
        return True
    else:
        logger.warning("GPUが利用できません。CPUモードで実行します。")
        return False

def monkey_patch_model_loading():
    """モデルロード関数をモンキーパッチで上書き"""
    from GPT_SoVITS import inference_webui
    
    # オリジナルの関数を保存
    original_change_sovits = inference_webui.change_sovits_weights
    original_load_sovits = inference_webui.load_sovits_new
    
    # カスタムload_sovits_new関数
    def custom_load_sovits_new(sovits_path):
        """カスタムパスを使用するload_sovits_new"""
        global CUSTOM_SOVITS_PATH
        if CUSTOM_SOVITS_PATH and os.path.exists(CUSTOM_SOVITS_PATH):
            logger.info(f"カスタムモデルを使用: {CUSTOM_SOVITS_PATH}")
            actual_path = CUSTOM_SOVITS_PATH
        else:
            actual_path = sovits_path
            
        # ckptファイルの場合の処理
        if actual_path.endswith('.ckpt'):
            logger.info("ckptファイルを読み込み中...")
            import torch
            checkpoint = torch.load(actual_path, map_location='cpu')
            
            # ckpt形式からpth形式への変換
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
                
            # 必要な構造を作成
            dict_s2 = {
                'weight': state_dict,
                'config': {
                    'model': {
                        'version': 'v2',  # デフォルトでv2
                        'semantic_frame_rate': '25hz'
                    },
                    'data': {
                        'sampling_rate': 32000,
                        'filter_length': 2048,
                        'hop_length': 640,
                        'win_length': 2048,
                        'n_speakers': 300
                    }
                }
            }
            return dict_s2
        else:
            # 通常のpthファイルの処理
            return original_load_sovits(actual_path)
    
    # モンキーパッチを適用
    inference_webui.load_sovits_new = custom_load_sovits_new
    logger.info("モンキーパッチ適用完了")

def load_models(sovits_path=None):
    """モデルをロード"""
    global CUSTOM_SOVITS_PATH
    
    # カスタムモデルパスを設定
    if sovits_path:
        CUSTOM_SOVITS_PATH = sovits_path
        logger.info(f"カスタムSoVITSモデルパス設定: {CUSTOM_SOVITS_PATH}")
    
    # モンキーパッチを適用
    monkey_patch_model_loading()
    
    # 既存のv2モデルパスを使用（ダミー）
    default_sovits_path = "/app/GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth"
    
    # モデルの存在確認
    if not os.path.exists(default_sovits_path):
        logger.warning(f"デフォルトモデルが見つかりません: {default_sovits_path}")
    
    logger.info("モデルをロード中...")
    
    try:
        from GPT_SoVITS.inference_webui import change_gpt_weights, change_sovits_weights
        
        # SoVITSモデルをロード（内部でカスタムパスが使用される）
        change_sovits_weights(default_sovits_path)
        
        logger.info("モデルのロード完了")
        return None
    
    except Exception as e:
        logger.error(f"モデルロードでエラー: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def process_voice_clone(inference, ref_audio_path, ref_text, target_text, output_path):
    """音声クローニング処理"""
    try:
        logger.info(f"参照音声: {ref_audio_path}")
        logger.info(f"参照テキスト: {ref_text}")
        logger.info(f"ターゲットテキスト: {target_text}")
        
        # 代替方法での実行
        from GPT_SoVITS.inference_webui import get_tts_wav
        
        # GPT-SoVITSのget_tts_wav関数の正しい引数で呼び出し
        result = get_tts_wav(
            ref_wav_path=ref_audio_path,
            prompt_text=ref_text,
            prompt_language="Japanese",
            text=target_text,
            text_language="Japanese",
            how_to_cut="不切",              # 分割なし
            top_k=20,
            top_p=1.0,
            temperature=1.0,
            ref_free=True                   # 参照音声の制約を緩和
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
    parser = argparse.ArgumentParser(description="GPT-SoVITS CLI音声クローニング（日本語特化モデル対応）")
    parser.add_argument("--ref-audio", type=str, required=True, help="参照音声ファイルパス")
    parser.add_argument("--ref-text", type=str, required=True, help="参照音声のテキスト")
    parser.add_argument("--target-text", type=str, required=True, help="生成したい音声のテキスト")
    parser.add_argument("--output", type=str, default="/app/output/result.wav", help="出力ファイルパス")
    parser.add_argument("--sovits-model", type=str, default=None, help="カスタムSoVITSモデルパス")
    
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
    
    # モデルロード（カスタムパス対応）
    inference = load_models(args.sovits_model)
    
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