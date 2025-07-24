#!/usr/bin/env python3
"""
TensorRT Engine Builder for Super Wav2Lip
動的解像度対応 ONNX → TensorRT 変換スクリプト

使用例:
python scripts/tensorrt_engine_builder.py --model wav2lip --optimize
python scripts/tensorrt_engine_builder.py --model gfpgan --dynamic --precision fp16
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import tensorrt as trt
import onnx
import numpy as np

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TensorRTEngineBuilder:
    def __init__(self, verbose=False):
        """TensorRT エンジンビルダー初期化"""
        self.logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.INFO)
        self.builder = trt.Builder(self.logger)
        self.config = self.builder.create_builder_config()
        
        # GPU メモリ設定（RTX 3050用最適化）
        self.config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)  # 2GB
        
    def build_wav2lip_engine(self, onnx_path: str, engine_path: str, 
                           dynamic_batch=True, fp16=True, int8=False):
        """
        Wav2Lip ONNX → TensorRT エンジン変換
        
        Args:
            onnx_path: 入力ONNXモデルパス
            engine_path: 出力TensorRTエンジンパス  
            dynamic_batch: 動的バッチサイズ対応
            fp16: FP16精度使用
            int8: INT8精度使用（要キャリブレーション）
        """
        logger.info(f"🚀 Wav2Lip TensorRT エンジン構築開始: {onnx_path}")
        
        # 精度設定
        if fp16 and self.builder.platform_has_fast_fp16:
            self.config.set_flag(trt.BuilderFlag.FP16)
            logger.info("✅ FP16精度有効")
        if int8 and self.builder.platform_has_fast_int8:
            self.config.set_flag(trt.BuilderFlag.INT8)
            logger.info("✅ INT8精度有効")
            
        # ONNX読み込み
        network = self.builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, self.logger)
        
        with open(onnx_path, 'rb') as model:
            if not parser.parse(model.read()):
                logger.error("❌ ONNX解析失敗")
                for error in range(parser.num_errors):
                    logger.error(parser.get_error(error))
                return False
                
        # 動的形状設定（Wav2Lip用）
        if dynamic_batch:
            # 1つのプロファイルに全ての入力を設定
            profile = self.builder.create_optimization_profile()
            
            # メルスペクトログラム入力: [batch, 1, 80, 16] 固定サイズ
            mel_input = network.get_input(0)
            profile.set_shape(mel_input.name, 
                            (1, 1, 80, 16),     # min
                            (1, 1, 80, 16),     # opt  
                            (1, 1, 80, 16))     # max
            
            # 動画フレーム入力: [batch, 6, 96, 96] 固定
            video_input = network.get_input(1)
            profile.set_shape(video_input.name,
                            (1, 6, 96, 96),   # min
                            (1, 6, 96, 96),   # opt
                            (1, 6, 96, 96))   # max
            
            self.config.add_optimization_profile(profile)
            logger.info("✅ 動的形状プロファイル設定完了")
            
        # エンジン構築
        logger.info("🔧 TensorRT エンジン構築中...")
        serialized_engine = self.builder.build_serialized_network(network, self.config)
        
        if serialized_engine is None:
            logger.error("❌ エンジン構築失敗")
            return False
            
        # エンジン保存
        with open(engine_path, 'wb') as f:
            f.write(serialized_engine)
            
        engine_size = os.path.getsize(engine_path) / (1024 * 1024)
        logger.info(f"✅ Wav2Lip TensorRT エンジン完成: {engine_path} ({engine_size:.1f}MB)")
        return True
        
    def build_gfpgan_engine(self, onnx_path: str, engine_path: str,
                          dynamic_resolution=True, fp16=True):
        """
        GFPGAN ONNX → TensorRT エンジン変換（動的解像度対応）
        
        Args:
            onnx_path: 入力ONNXモデルパス
            engine_path: 出力TensorRTエンジンパス
            dynamic_resolution: 動的解像度対応
            fp16: FP16精度使用
        """
        logger.info(f"🎨 GFPGAN TensorRT エンジン構築開始: {onnx_path}")
        
        # 精度設定
        if fp16 and self.builder.platform_has_fast_fp16:
            self.config.set_flag(trt.BuilderFlag.FP16)
            logger.info("✅ FP16精度有効")
            
        # ONNX読み込み
        network = self.builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, self.logger)
        
        with open(onnx_path, 'rb') as model:
            if not parser.parse(model.read()):
                logger.error("❌ ONNX解析失敗")
                for error in range(parser.num_errors):
                    logger.error(parser.get_error(error))
                return False
                
        # 動的解像度設定（GFPGAN用）
        if dynamic_resolution:
            input_tensor = network.get_input(0)
            profile = self.builder.create_optimization_profile()
            
            # 動的解像度: 256x256 → 512x512 → 1024x1024
            profile.set_shape(input_tensor.name,
                            (1, 3, 256, 256),    # min: 軽量処理
                            (1, 3, 512, 512),    # opt: 標準品質  
                            (1, 3, 1024, 1024))  # max: 最高品質
            
            self.config.add_optimization_profile(profile)
            logger.info("✅ 動的解像度プロファイル設定完了 (256-1024px)")
            
        # 最適化設定
        self.config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
        
        # エンジン構築
        logger.info("🔧 GFPGAN TensorRT エンジン構築中...")
        serialized_engine = self.builder.build_serialized_network(network, self.config)
        
        if serialized_engine is None:
            logger.error("❌ エンジン構築失敗")
            return False
            
        # エンジン保存
        with open(engine_path, 'wb') as f:
            f.write(serialized_engine)
            
        engine_size = os.path.getsize(engine_path) / (1024 * 1024)
        logger.info(f"✅ GFPGAN TensorRT エンジン完成: {engine_path} ({engine_size:.1f}MB)")
        return True
        
    def benchmark_engine(self, engine_path: str, input_shape: tuple, iterations=100):
        """
        TensorRT エンジンのベンチマーク実行
        
        Args:
            engine_path: TensorRTエンジンパス
            input_shape: 入力形状
            iterations: 実行回数
        """
        logger.info(f"⚡ TensorRT エンジンベンチマーク: {engine_path}")
        
        # エンジン読み込み
        runtime = trt.Runtime(self.logger)
        with open(engine_path, 'rb') as f:
            engine = runtime.deserialize_cuda_engine(f.read())
            
        context = engine.create_execution_context()
        
        # GPU メモリ確保
        import pycuda.driver as cuda
        import pycuda.autoinit
        
        # 入力・出力バッファ確保
        input_buffer = cuda.mem_alloc(np.prod(input_shape) * np.dtype(np.float32).itemsize)
        output_shape = (1, 3, input_shape[2], input_shape[3])  # 同じサイズ出力想定
        output_buffer = cuda.mem_alloc(np.prod(output_shape) * np.dtype(np.float32).itemsize)
        
        # ベンチマーク実行
        import time
        cuda.Context.synchronize()
        start_time = time.time()
        
        for _ in range(iterations):
            context.execute_v2([int(input_buffer), int(output_buffer)])
            cuda.Context.synchronize()
            
        end_time = time.time()
        
        avg_time = (end_time - start_time) / iterations * 1000  # ms
        fps = 1000 / avg_time
        
        logger.info(f"✅ ベンチマーク結果:")
        logger.info(f"   平均実行時間: {avg_time:.2f}ms")
        logger.info(f"   FPS: {fps:.1f}")
        logger.info(f"   入力形状: {input_shape}")
        
        return avg_time, fps

def main():
    parser = argparse.ArgumentParser(description='TensorRT エンジンビルダー')
    parser.add_argument('--model', choices=['wav2lip', 'gfpgan', 'both'], 
                       default='both', help='変換対象モデル')
    parser.add_argument('--onnx-dir', default='/app/models/onnx', 
                       help='ONNXモデルディレクトリ')
    parser.add_argument('--engine-dir', default='/app/models/tensorrt',
                       help='TensorRTエンジン出力ディレクトリ')
    parser.add_argument('--dynamic', action='store_true',
                       help='動的形状対応')
    parser.add_argument('--precision', choices=['fp32', 'fp16', 'int8'],
                       default='fp16', help='推論精度')
    parser.add_argument('--optimize', action='store_true',
                       help='最適化有効')
    parser.add_argument('--benchmark', action='store_true',
                       help='ベンチマーク実行')
    parser.add_argument('--verbose', action='store_true',
                       help='詳細ログ')
    
    args = parser.parse_args()
    
    # ディレクトリ作成
    Path(args.engine_dir).mkdir(parents=True, exist_ok=True)
    
    # ビルダー初期化
    builder = TensorRTEngineBuilder(verbose=args.verbose)
    
    # Wav2Lip エンジン構築
    if args.model in ['wav2lip', 'both']:
        wav2lip_onnx = f"{args.onnx_dir}/wav2lip_gan.onnx"
        wav2lip_engine = f"{args.engine_dir}/wav2lip_gan.trt"
        
        if os.path.exists(wav2lip_onnx):
            success = builder.build_wav2lip_engine(
                wav2lip_onnx, wav2lip_engine,
                dynamic_batch=args.dynamic,
                fp16=(args.precision == 'fp16'),
                int8=(args.precision == 'int8')
            )
            
            if success and args.benchmark:
                builder.benchmark_engine(wav2lip_engine, (1, 1, 80, 16))
        else:
            logger.warning(f"⚠️ Wav2Lip ONNX not found: {wav2lip_onnx}")
    
    # GFPGAN エンジン構築  
    if args.model in ['gfpgan', 'both']:
        gfpgan_onnx = f"{args.onnx_dir}/../enhancers/GFPGAN/GFPGANv1.4.onnx"
        gfpgan_engine = f"{args.engine_dir}/gfpgan_v1.4.trt"
        
        if os.path.exists(gfpgan_onnx):
            success = builder.build_gfpgan_engine(
                gfpgan_onnx, gfpgan_engine,
                dynamic_resolution=args.dynamic,
                fp16=(args.precision == 'fp16')
            )
            
            if success and args.benchmark:
                builder.benchmark_engine(gfpgan_engine, (1, 3, 512, 512))
        else:
            logger.warning(f"⚠️ GFPGAN ONNX not found: {gfpgan_onnx}")
    
    logger.info("🎉 TensorRT エンジン構築完了!")

if __name__ == "__main__":
    main()