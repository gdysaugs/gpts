#!/usr/bin/env python3
"""
GPT-SoVITS CLI テストスクリプト - 日本語特化モデル完全対応版
すべての関数をモンキーパッチで上書き
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import torch
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
        logging.FileHandler('/app/logs/test_voice_clone.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# グローバル変数
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

def comprehensive_monkey_patch():
    """包括的なモンキーパッチ + Torch.compile最適化を適用"""
    from GPT_SoVITS import inference_webui
    import torch
    
    # Torch.compile最適化設定
    def setup_torch_optimizations():
        """Torch.compile + TensorCore最適化セットアップ"""
        if torch.cuda.is_available():
            # RTX 3050 TensorCore最適化
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            
            # メモリ最適化
            torch.cuda.empty_cache()
            logger.info("RTX 3050 TensorCore最適化有効化")
        
        # PyTorch 2.x最適化
        torch.set_float32_matmul_precision('medium')
        torch.set_num_threads(8)
        logger.info("PyTorch最適化設定完了")
    
    setup_torch_optimizations()
    
    # オリジナル関数を保存
    original_load_sovits = inference_webui.load_sovits_new
    original_change_sovits = inference_webui.change_sovits_weights
    
    # カスタムload_sovits_new関数
    def custom_load_sovits_new(sovits_path):
        """カスタムパスを優先するload_sovits_new"""
        global CUSTOM_SOVITS_PATH
        
        # カスタムパスがある場合は使用
        if CUSTOM_SOVITS_PATH and os.path.exists(CUSTOM_SOVITS_PATH):
            logger.info(f"カスタムモデルをロード: {CUSTOM_SOVITS_PATH}")
            actual_path = CUSTOM_SOVITS_PATH
        else:
            actual_path = sovits_path
        
        # ckptファイルの処理
        if actual_path.endswith('.ckpt'):
            logger.info(f"ckptファイルをロード中: {actual_path}")
            checkpoint = torch.load(actual_path, map_location='cpu')
            
            # state_dictを抽出
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # v2形式の構造を作成
            dict_s2 = {
                'weight': state_dict,
                'config': {
                    'model': {
                        'version': 'v2',
                        'semantic_frame_rate': '25hz',
                        'inter_channels': 192,
                        'hidden_channels': 192,
                        'filter_channels': 768,
                        'n_heads': 2,
                        'n_layers': 6,
                        'kernel_size': 3,
                        'p_dropout': 0.1,
                        'ssl_dim': 768,
                        'n_speakers': 300
                    },
                    'data': {
                        'sampling_rate': 32000,
                        'filter_length': 2048,
                        'hop_length': 640,
                        'win_length': 2048,
                        'n_speakers': 300,
                        'text_cleaners': ['japanese_cleaners']
                    }
                }
            }
            
            # 必要なキーが存在しない場合、ダミーのweightを追加
            if 'enc_p.text_embedding.weight' not in dict_s2['weight']:
                # v2形式のダミーweight
                dict_s2['weight']['enc_p.text_embedding.weight'] = torch.zeros(512, 192)
            
            return dict_s2
        else:
            # 通常のpthファイル
            return original_load_sovits(actual_path)
    
    # カスタムchange_sovits_weights関数
    def custom_change_sovits_weights(sovits_path, prompt_language=None, text_language=None):
        """カスタムパスを認識するchange_sovits_weights"""
        global CUSTOM_SOVITS_PATH
        
        # カスタムパスが設定されている場合は、それを優先
        if CUSTOM_SOVITS_PATH:
            logger.info(f"カスタムモデルパスを使用: {CUSTOM_SOVITS_PATH}")
            # ダミーのv2パスを渡して、実際にはカスタムパスが使われるようにする
            sovits_path = "/app/GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth"
        
        # オリジナル関数を呼び出し
        return original_change_sovits(sovits_path, prompt_language, text_language)
    
    # モンキーパッチを適用
    inference_webui.load_sovits_new = custom_load_sovits_new
    inference_webui.change_sovits_weights = custom_change_sovits_weights
    
    logger.info("包括的モンキーパッチ適用完了")

def apply_torch_compile_optimization():
    """ロード済みモデルにTorch.compile最適化を適用"""
    try:
        # PyTorch 2.0+ Torch.compile対応確認
        if not hasattr(torch, 'compile'):
            logger.warning("PyTorch 2.0+が必要です（Torch.compile非対応）")
            return
        
        logger.info("Torch.compile最適化開始...")
        
        # GPT-SoVITSのグローバルモデル取得
        from GPT_SoVITS import inference_webui
        
        # SoVITSモデル最適化
        if hasattr(inference_webui, 'vq_model') and inference_webui.vq_model is not None:
            logger.info("SoVITSモデルをcompile最適化中...")
            inference_webui.vq_model = torch.compile(
                inference_webui.vq_model,
                mode="max-autotune",  # RTX 3050最適化
                dynamic=True,
                backend="inductor"
            )
            logger.info("SoVITSモデル最適化完了")
        
        # GPTモデル最適化
        if hasattr(inference_webui, 't2s_model') and inference_webui.t2s_model is not None:
            logger.info("GPTモデルをcompile最適化中...")
            inference_webui.t2s_model = torch.compile(
                inference_webui.t2s_model,
                mode="max-autotune",
                dynamic=True,
                backend="inductor"
            )
            logger.info("GPTモデル最適化完了")
        
        # HuBERTモデル最適化
        if hasattr(inference_webui, 'hubert_model') and inference_webui.hubert_model is not None:
            logger.info("HuBERTモデルをcompile最適化中...")
            inference_webui.hubert_model = torch.compile(
                inference_webui.hubert_model,
                mode="reduce-overhead",  # HuBERTは軽量化優先
                dynamic=True,
                backend="inductor"
            )
            logger.info("HuBERTモデル最適化完了")
        
        logger.info("Torch.compile最適化完了 - 2-4倍高速化期待")
        
    except Exception as e:
        logger.error(f"Torch.compile最適化エラー: {e}")
        logger.warning("最適化なしで続行します")

def load_models(sovits_path=None):
    """モデルをロード"""
    global CUSTOM_SOVITS_PATH
    
    # カスタムモデルパスを設定
    if sovits_path:
        CUSTOM_SOVITS_PATH = sovits_path
        logger.info(f"カスタムSoVITSモデルパス設定: {CUSTOM_SOVITS_PATH}")
    
    # モンキーパッチを適用
    comprehensive_monkey_patch()
    
    logger.info("モデルをロード中...")
    
    try:
        from GPT_SoVITS.inference_webui import change_sovits_weights
        
        # ダミーパスでモデルロード（内部でカスタムパスが使用される）
        default_path = "/app/GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth"
        change_sovits_weights(default_path)
        
        logger.info("モデルのロード完了")
        
        # Torch.compile最適化を適用
        apply_torch_compile_optimization()
        
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
        
        from GPT_SoVITS.inference_webui import get_tts_wav
        
        # 生成実行
        result = get_tts_wav(
            ref_wav_path=ref_audio_path,
            prompt_text=ref_text,
            prompt_language="Japanese",
            text=target_text,
            text_language="Japanese",
            how_to_cut="不切",
            top_k=20,
            top_p=1.0,
            temperature=1.0,
            ref_free=True
        )
        
        # 結果を保存
        if result:
            try:
                logger.info(f"結果の型: {type(result)}")
                
                if hasattr(result, '__iter__') and not isinstance(result, (list, tuple, str)):
                    logger.info("ジェネレーターから結果を取得中...")
                    
                    import numpy as np
                    audio_segments = []
                    sample_rate = None
                    
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
                        final_audio = np.concatenate(audio_segments)
                        
                        logger.info(f"最終音声: サンプリングレート {sample_rate}Hz")
                        logger.info(f"最終音声データ長: {len(final_audio)} samples")
                        logger.info(f"最終音声時間: {len(final_audio)/sample_rate:.2f}秒")
                        logger.info(f"セグメント数: {len(audio_segments)}")
                        
                        # 音声統計
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
    parser = argparse.ArgumentParser(description="GPT-SoVITS CLI音声クローニング（日本語特化モデル完全対応）")
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
    
    # モデルロード
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