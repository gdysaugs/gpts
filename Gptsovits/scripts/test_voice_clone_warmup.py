#!/usr/bin/env python3
"""
モデル事前初期化（Warm-up）最適化版GPT-SoVITS音声クローニング
初期化オーバーヘッドを大幅削減し、特に短文で劇的な高速化を実現
"""

import os
import sys
import argparse
import logging
import time
import torch
import numpy as np
from pathlib import Path

# GPT-SoVITSパス設定
os.chdir('/app')
sys.path.insert(0, '/app')
sys.path.insert(0, '/app/GPT_SoVITS')

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/app/logs/test_voice_clone_warmup.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# グローバル変数
CUSTOM_SOVITS_PATH = None
MODELS_WARMED_UP = False
WARMUP_CACHE = {}

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

def comprehensive_monkey_patch():
    """包括的なモンキーパッチ + Torch.compile最適化を適用"""
    from GPT_SoVITS import inference_webui
    import torch
    
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
                        'cleaned_text': True,
                        'add_blank': True,
                        'n_symbols': 178
                    }
                }
            }
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

def warmup_models():
    """モデル事前初期化（Warm-up）実行"""
    global MODELS_WARMED_UP, WARMUP_CACHE
    
    if MODELS_WARMED_UP:
        logger.info("モデルは既にWarm-up済みです")
        return
    
    logger.info("=== モデルWarm-up開始 ===")
    warmup_start = time.time()
    
    try:
        from GPT_SoVITS import inference_webui
        
        # ダミー入力でモデル初期化
        logger.info("ダミー推論でモデルをWarm-up中...")
        
        # 短いダミーテキスト
        dummy_text = "テスト"
        dummy_ref_text = "こんにちは"
        
        # ダミー推論実行（結果は破棄）
        try:
            result_gen = inference_webui.get_tts_wav(
                ref_wav_path="/app/input/reference_5sec.wav",
                prompt_text=dummy_ref_text,
                prompt_language="Japanese",
                text=dummy_text,
                text_language="Japanese",
                how_to_cut="不切",
                top_k=5,
                top_p=1.0,
                temperature=1.0,
                ref_free=True
            )
            
            # ジェネレーターから一つだけ取得
            for i, item in enumerate(result_gen):
                if i == 0:  # 最初の結果のみ
                    logger.info("Warm-up推論完了")
                    break
            
        except Exception as e:
            logger.warning(f"Warm-up推論中にエラー: {e}")
        
        # Torch.compile最適化適用
        apply_torch_compile_optimization()
        
        # CUDAキャッシュ最適化
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            logger.info("CUDAキャッシュ最適化完了")
        
        warmup_time = time.time() - warmup_start
        MODELS_WARMED_UP = True
        
        logger.info(f"=== モデルWarm-up完了: {warmup_time:.2f}秒 ===")
        
    except Exception as e:
        logger.error(f"モデルWarm-upエラー: {e}")
        import traceback
        traceback.print_exc()

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
        
        logger.info("Torch.compile最適化完了")
        
    except Exception as e:
        logger.error(f"Torch.compile最適化エラー: {e}")
        logger.warning("最適化なしで続行します")

def load_models_with_warmup(sovits_path=None):
    """Warm-up付きモデルロード"""
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
        
        # モデルWarm-up実行
        warmup_models()
        
        return None
    
    except Exception as e:
        logger.error(f"モデルロードでエラー: {e}")
        import traceback
        traceback.print_exc()
        raise

def generate_voice_optimized(ref_audio_path, ref_text, target_text, output_path):
    """Warm-up最適化版音声生成"""
    
    logger.info("=== Warm-up最適化音声生成開始 ===")
    generation_start = time.time()
    
    try:
        from GPT_SoVITS.inference_webui import get_tts_wav
        
        logger.info(f"参照音声: {ref_audio_path}")
        logger.info(f"参照テキスト: {ref_text}")
        logger.info(f"ターゲットテキスト: {target_text}")
        
        # 音声生成実行
        result_generator = get_tts_wav(
            ref_wav_path=ref_audio_path,
            prompt_text=ref_text,
            prompt_language="Japanese",
            text=target_text,
            text_language="Japanese",
            how_to_cut="不切",
            top_k=5,
            top_p=1.0,
            temperature=1.0,
            ref_free=True
        )
        
        logger.info(f"結果の型: {type(result_generator)}")
        logger.info("ジェネレーターから結果を取得中...")
        
        # 結果の処理
        audio_segments = []
        for i, item in enumerate(result_generator):
            logger.info(f"アイテム {i}: {type(item)}")
            
            if isinstance(item, tuple) and len(item) == 2:
                sample_rate, audio_data = item
                logger.info(f"セグメント {i}: {sample_rate}Hz, {len(audio_data)} samples, {len(audio_data)/sample_rate:.2f}秒")
                audio_segments.append(audio_data)
            else:
                logger.warning(f"予期しない形式のアイテム: {item}")
        
        # 音声データの結合
        if len(audio_segments) > 1:
            final_audio = np.concatenate(audio_segments)
        elif len(audio_segments) == 1:
            final_audio = audio_segments[0]
        else:
            raise ValueError("音声生成に失敗しました")
        
        logger.info(f"最終音声: サンプリングレート 32000Hz")
        logger.info(f"最終音声データ長: {len(final_audio)} samples")
        logger.info(f"最終音声時間: {len(final_audio)/32000:.2f}秒")
        logger.info(f"セグメント数: {len(audio_segments)}")
        
        # 音声品質統計
        audio_max = float(np.max(np.abs(final_audio)))
        audio_min = float(np.min(final_audio))
        audio_mean = float(np.mean(final_audio))
        audio_rms = float(np.sqrt(np.mean(final_audio ** 2)))
        non_silence_ratio = float(np.sum(np.abs(final_audio) > np.max(np.abs(final_audio)) * 0.01) / len(final_audio))
        
        logger.info(f"音声データ統計: max={audio_max:.6f}, min={audio_min:.6f}, mean={audio_mean:.6f}")
        logger.info(f"音声RMS: {audio_rms:.6f}")
        logger.info(f"非無音サンプル: {int(non_silence_ratio * len(final_audio))}/{len(final_audio)} ({non_silence_ratio:.1%})")
        
        # ファイル保存
        import soundfile as sf
        sf.write(output_path, final_audio, 32000)
        
        generation_time = time.time() - generation_start
        
        logger.info(f"音声生成完了: {output_path}")
        logger.info(f"=== 総生成時間: {generation_time:.2f}秒 ===")
        logger.info("処理が正常に完了しました")
        
        return {
            'generation_time': generation_time,
            'audio_duration': len(final_audio) / 32000,
            'realtime_factor': (len(final_audio) / 32000) / generation_time,
            'audio_rms': audio_rms,
            'non_silence_ratio': non_silence_ratio,
            'segments': len(audio_segments)
        }
        
    except Exception as e:
        logger.error(f"音声生成でエラー: {e}")
        import traceback
        traceback.print_exc()
        raise

def main():
    """メイン実行関数"""
    parser = argparse.ArgumentParser(description='Warm-up最適化GPT-SoVITS音声クローニング')
    
    parser.add_argument('--sovits-model', 
                       default='/app/GPT_SoVITS/pretrained_models/gpt-sovits-ja-h/hscene-e17.ckpt',
                       help='SoVITSモデルパス')
    parser.add_argument('--ref-audio', required=True, help='参照音声ファイル')
    parser.add_argument('--ref-text', required=True, help='参照テキスト')
    parser.add_argument('--target-text', required=True, help='生成テキスト')
    parser.add_argument('--output', required=True, help='出力音声ファイル')
    
    args = parser.parse_args()
    
    # GPU確認
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name()
        cuda_version = torch.version.cuda
        logger.info(f"GPU利用可能: {gpu_name}")
        logger.info(f"CUDA バージョン: {cuda_version}")
    else:
        logger.warning("GPU利用不可、CPU実行")
    
    # 総処理時間測定開始
    total_start = time.time()
    
    # モデルロード & Warm-up
    load_models_with_warmup(args.sovits_model)
    
    # 音声生成
    stats = generate_voice_optimized(
        ref_audio_path=args.ref_audio,
        ref_text=args.ref_text,
        target_text=args.target_text,
        output_path=args.output
    )
    
    total_time = time.time() - total_start
    
    # パフォーマンス報告
    logger.info("=== Warm-up最適化パフォーマンス統計 ===")
    logger.info(f"総処理時間: {total_time:.2f}秒")
    logger.info(f"音声生成時間: {stats['generation_time']:.2f}秒")
    logger.info(f"音声時間: {stats['audio_duration']:.2f}秒")
    logger.info(f"リアルタイム係数: {stats['realtime_factor']:.2f}x")
    logger.info(f"音声RMS: {stats['audio_rms']:.3f}")
    logger.info(f"非無音率: {stats['non_silence_ratio']:.1%}")

if __name__ == "__main__":
    main()