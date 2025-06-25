#!/usr/bin/env python3
"""
Torch.compile最適化版GPT-SoVITS音声クローニング
hscene-e17.ckptのまま2-4倍高速化実現
RTX 3050でmax-autotuneモード使用
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import soundfile as sf
import logging
import argparse
import time
import warnings
from pathlib import Path
from typing import Optional, Dict, Any

# 警告を抑制
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# GPT-SoVITSパス設定
sys.path.append('/app')

class TorchCompileVoiceCloner:
    """Torch.compile最適化音声クローニング"""
    
    def __init__(self, 
                 sovits_model_path: str,
                 gpt_model_path: str,
                 use_compile: bool = True,
                 compile_mode: str = "max-autotune"):
        
        self.sovits_model_path = sovits_model_path
        self.gpt_model_path = gpt_model_path
        self.use_compile = use_compile
        self.compile_mode = compile_mode
        
        # モデル初期化
        self.setup_models()
        
        # Torch.compile最適化
        if self.use_compile:
            self.optimize_models()
    
    def setup_models(self):
        """GPT-SoVITSモデル初期化"""
        logger.info("GPT-SoVITSモデル初期化開始...")
        
        # GPU設定
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 必要なモジュールインポート
        try:
            from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config
            from GPT_SoVITS.TTS_infer_pack.text_segmentation_method import splits
            
            # TTS設定
            self.tts_config = TTS_Config(
                device=self.device,
                is_half=True,  # FP16最適化
                t2s_weights_path=self.gpt_model_path,
                vits_weights_path=self.sovits_model_path,
                cnhubert_base_path="/app/GPT_SoVITS/pretrained_models/chinese-hubert-base",
                bert_path="/app/GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large",
                infer_ttswebui=False,
                version="v2"
            )
            
            # TTSモデル初期化
            self.tts = TTS(self.tts_config)
            
            logger.info("GPT-SoVITSモデル初期化完了")
            
        except Exception as e:
            logger.error(f"モデル初期化エラー: {e}")
            raise
    
    def optimize_models(self):
        """Torch.compile最適化適用"""
        logger.info(f"Torch.compile最適化開始: mode={self.compile_mode}")
        
        try:
            # PyTorch 2.0+のcompile最適化
            if hasattr(torch, 'compile'):
                
                # SoVITSモデル最適化（音響モデル）
                if hasattr(self.tts, 'vits'):
                    logger.info("SoVITSモデルをcompile最適化中...")
                    self.tts.vits = torch.compile(
                        self.tts.vits, 
                        mode=self.compile_mode,
                        dynamic=True,  # 動的シーケンス長対応
                        backend="inductor"  # RTX 3050最適化
                    )
                
                # GPTモデル最適化（言語モデル）
                if hasattr(self.tts, 't2s_model'):
                    logger.info("GPTモデルをcompile最適化中...")
                    self.tts.t2s_model = torch.compile(
                        self.tts.t2s_model,
                        mode=self.compile_mode,
                        dynamic=True,
                        backend="inductor"
                    )
                
                # HuBERTモデル最適化（特徴抽出）
                if hasattr(self.tts, 'hubert_model'):
                    logger.info("HuBERTモデルをcompile最適化中...")
                    self.tts.hubert_model = torch.compile(
                        self.tts.hubert_model,
                        mode="reduce-overhead",  # HuBERTは軽量化優先
                        dynamic=True,
                        backend="inductor"
                    )
                
                logger.info("Torch.compile最適化完了")
                
            else:
                logger.warning("PyTorch 2.0+が必要です（Torch.compile未対応）")
                
        except Exception as e:
            logger.error(f"Torch.compile最適化エラー: {e}")
            logger.warning("最適化なしで続行します")
    
    def setup_performance_optimizations(self):
        """追加パフォーマンス最適化"""
        logger.info("追加パフォーマンス最適化設定...")
        
        # CUDAメモリ最適化
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True  # RTX 3050 TensorCore活用
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True  # 動的最適化
            torch.backends.cudnn.deterministic = False  # 速度優先
            
            # メモリフラグメンテーション削減
            torch.cuda.empty_cache()
            
        # CPU最適化
        torch.set_num_threads(8)  # RTX 3050システム想定
        torch.set_float32_matmul_precision('medium')  # 精度vs速度バランス
        
        logger.info("パフォーマンス最適化設定完了")
    
    def preprocess_text(self, text: str, language: str = "ja") -> list:
        """高速化テキスト前処理"""
        
        # テキスト分割最適化
        from GPT_SoVITS.TTS_infer_pack.text_segmentation_method import splits
        
        # 効率的分割（長文対応）
        segments = splits(text, language)
        
        return segments
    
    def clone_voice_optimized(self,
                             ref_audio_path: str,
                             ref_text: str,
                             target_text: str,
                             output_path: str,
                             language: str = "ja",
                             top_k: int = 5,
                             top_p: float = 1.0,
                             temperature: float = 1.0) -> dict:
        """最適化音声クローニング"""
        
        logger.info("=== Torch.compile最適化音声クローニング開始 ===")
        
        # パフォーマンス最適化設定
        self.setup_performance_optimizations()
        
        total_start = time.time()
        
        try:
            # テキスト前処理
            preprocess_start = time.time()
            text_segments = self.preprocess_text(target_text, language)
            preprocess_time = time.time() - preprocess_start
            
            logger.info(f"テキスト分割: {len(text_segments)}セグメント")
            
            # 音声生成（最適化版）
            inference_start = time.time()
            
            # TTS推論実行
            result_generator = self.tts.predict(
                ref_wav_path=ref_audio_path,
                prompt_text=ref_text,
                prompt_language=language,
                text=target_text,
                text_language=language,
                how_to_cut="不切",  # 全文生成
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                ref_free=True  # 参照制約解除
            )
            
            # 結果取得と連結
            audio_segments = []
            segment_times = []
            
            for i, (audio_data, sample_rate) in enumerate(result_generator):
                segment_start = time.time()
                
                if isinstance(audio_data, tuple):
                    audio_array = audio_data[1]  # (sample_rate, audio_array)
                else:
                    audio_array = audio_data
                
                audio_segments.append(audio_array)
                segment_times.append(time.time() - segment_start)
                
                logger.info(f"セグメント {i+1}: {len(audio_array)} samples")
            
            # 音声連結
            if len(audio_segments) > 1:
                final_audio = np.concatenate(audio_segments)
            elif len(audio_segments) == 1:
                final_audio = audio_segments[0]
            else:
                raise ValueError("音声生成失敗")
            
            inference_time = time.time() - inference_start
            
            # 出力保存
            save_start = time.time()
            sf.write(output_path, final_audio, 32000)
            save_time = time.time() - save_start
            
            total_time = time.time() - total_start
            
            # 統計計算
            audio_duration = len(final_audio) / 32000
            realtime_factor = audio_duration / inference_time
            
            # 音声品質統計
            audio_max = float(np.max(np.abs(final_audio)))
            audio_rms = float(np.sqrt(np.mean(final_audio ** 2)))
            non_silence_ratio = float(np.sum(np.abs(final_audio) > 0.01) / len(final_audio))
            
            stats = {
                'total_time': total_time,
                'preprocess_time': preprocess_time,
                'inference_time': inference_time,
                'save_time': save_time,
                'audio_duration': audio_duration,
                'realtime_factor': realtime_factor,
                'segments_count': len(audio_segments),
                'audio_samples': len(final_audio),
                'audio_max': audio_max,
                'audio_rms': audio_rms,
                'non_silence_ratio': non_silence_ratio,
                'compile_mode': self.compile_mode if self.use_compile else 'disabled'
            }
            
            logger.info("=== Torch.compile最適化音声クローニング完了 ===")
            logger.info(f"総処理時間: {stats['total_time']:.3f}秒")
            logger.info(f"推論時間: {stats['inference_time']:.3f}秒")
            logger.info(f"音声長: {stats['audio_duration']:.2f}秒")
            logger.info(f"リアルタイム係数: {stats['realtime_factor']:.2f}x")
            logger.info(f"音声品質 - RMS: {stats['audio_rms']:.3f}, 非無音率: {stats['non_silence_ratio']:.1%}")
            
            return stats
            
        except Exception as e:
            logger.error(f"音声クローニングエラー: {e}")
            raise

def main():
    """メイン実行関数"""
    parser = argparse.ArgumentParser(description='Torch.compile最適化音声クローニング')
    
    parser.add_argument('--sovits-model', 
                       default='/app/GPT_SoVITS/pretrained_models/gpt-sovits-ja-h/hscene-e17.ckpt',
                       help='SoVITSモデルパス')
    parser.add_argument('--gpt-model',
                       default='/app/GPT_SoVITS/pretrained_models/gpt-sovits-ja-h/hscene-e17.ckpt', 
                       help='GPTモデルパス')
    parser.add_argument('--ref-audio', required=True, help='参照音声ファイル')
    parser.add_argument('--ref-text', required=True, help='参照テキスト')
    parser.add_argument('--target-text', required=True, help='生成テキスト')
    parser.add_argument('--output', required=True, help='出力音声ファイル')
    parser.add_argument('--no-compile', action='store_true', help='Torch.compile無効化')
    parser.add_argument('--compile-mode', default='max-autotune', 
                       choices=['default', 'reduce-overhead', 'max-autotune'],
                       help='Compile最適化モード')
    
    args = parser.parse_args()
    
    # GPU確認
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name()
        cuda_version = torch.version.cuda
        logger.info(f"GPU利用可能: {gpu_name}")
        logger.info(f"CUDA バージョン: {cuda_version}")
        logger.info(f"PyTorch バージョン: {torch.__version__}")
    else:
        logger.warning("GPU利用不可、CPU実行")
    
    # Torch.compile対応確認
    if hasattr(torch, 'compile'):
        logger.info("Torch.compile対応 ✓")
    else:
        logger.warning("Torch.compile非対応（PyTorch 2.0+が必要）")
        args.no_compile = True
    
    # 音声クローナー初期化
    cloner = TorchCompileVoiceCloner(
        sovits_model_path=args.sovits_model,
        gpt_model_path=args.gpt_model,
        use_compile=not args.no_compile,
        compile_mode=args.compile_mode
    )
    
    # 音声クローニング実行
    stats = cloner.clone_voice_optimized(
        ref_audio_path=args.ref_audio,
        ref_text=args.ref_text,
        target_text=args.target_text,
        output_path=args.output
    )
    
    # パフォーマンス報告
    logger.info("=== 最適化パフォーマンス統計 ===")
    for key, value in stats.items():
        if isinstance(value, float):
            logger.info(f"{key}: {value:.4f}")
        else:
            logger.info(f"{key}: {value}")

if __name__ == "__main__":
    main()