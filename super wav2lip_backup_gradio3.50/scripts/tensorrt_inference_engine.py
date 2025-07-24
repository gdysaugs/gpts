#!/usr/bin/env python3
"""
TensorRT 推論エンジン for Super Wav2Lip
動的解像度対応 TensorRT推論ラッパー

使用例:
from scripts.tensorrt_inference_engine import TensorRTInference
engine = TensorRTInference('models/tensorrt/gfpgan_v1.4.trt')
result = engine.infer(input_tensor, target_resolution=(512, 512))
"""

import os
import time
import logging
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Union
import cv2

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False

logger = logging.getLogger(__name__)

class TensorRTInference:
    """TensorRT推論エンジン（動的解像度対応）"""
    
    def __init__(self, engine_path: str, logger_level=trt.Logger.WARNING):
        """
        TensorRT推論エンジン初期化
        
        Args:
            engine_path: TensorRTエンジンファイルパス
            logger_level: TensorRTログレベル
        """
        if not TENSORRT_AVAILABLE:
            raise ImportError("TensorRT not available. Install with: pip install tensorrt")
            
        if not os.path.exists(engine_path):
            raise FileNotFoundError(f"TensorRT engine not found: {engine_path}")
            
        self.engine_path = engine_path
        self.trt_logger = trt.Logger(logger_level)
        self.runtime = trt.Runtime(self.trt_logger)
        
        # エンジン読み込み
        with open(engine_path, 'rb') as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
            
        self.context = self.engine.create_execution_context()
        
        # エンジン情報取得
        self.num_bindings = self.engine.num_bindings
        self.input_names = []
        self.output_names = []
        self.input_shapes = {}
        self.output_shapes = {}
        
        for i in range(self.num_bindings):
            name = self.engine.get_binding_name(i)
            shape = self.engine.get_binding_shape(i)
            
            if self.engine.binding_is_input(i):
                self.input_names.append(name)
                self.input_shapes[name] = shape
            else:
                self.output_names.append(name)
                self.output_shapes[name] = shape
                
        # GPU メモリバッファ（動的確保）
        self.buffers = {}
        self.buffer_sizes = {}
        
        logger.info(f"✅ TensorRT エンジン読み込み完了: {engine_path}")
        logger.info(f"   入力: {self.input_names}")
        logger.info(f"   出力: {self.output_names}")
        
    def _allocate_buffers(self, input_shape: Tuple[int, ...], output_shape: Tuple[int, ...]):
        """動的バッファ確保"""
        input_size = np.prod(input_shape) * np.dtype(np.float32).itemsize
        output_size = np.prod(output_shape) * np.dtype(np.float32).itemsize
        
        # 既存バッファより大きい場合のみ再確保
        if input_size > self.buffer_sizes.get('input', 0):
            if 'input' in self.buffers:
                cuda.mem_free(self.buffers['input'])
            self.buffers['input'] = cuda.mem_alloc(input_size)
            self.buffer_sizes['input'] = input_size
            
        if output_size > self.buffer_sizes.get('output', 0):
            if 'output' in self.buffers:
                cuda.mem_free(self.buffers['output'])
            self.buffers['output'] = cuda.mem_alloc(output_size)
            self.buffer_sizes['output'] = output_size
            
    def set_dynamic_shape(self, input_name: str, shape: Tuple[int, ...]):
        """動的形状設定"""
        if not self.context.set_binding_shape(0, shape):  # 0は入力インデックス
            raise RuntimeError(f"Failed to set dynamic shape: {shape}")
            
        # 出力形状を更新（推論実行前に必要）
        if not self.context.all_binding_shapes_specified:
            raise RuntimeError("Not all binding shapes specified")
            
    def preprocess_gfpgan(self, image: np.ndarray, target_size: Tuple[int, int] = (512, 512)) -> np.ndarray:
        """GFPGAN用前処理"""
        # リサイズ
        resized = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
        
        # BGR → RGB + 正規化
        rgb = resized[:, :, ::-1].astype(np.float32) / 255.0
        
        # HWC → CHW
        chw = rgb.transpose((2, 0, 1))
        
        # [-1, 1] 正規化
        normalized = (chw - 0.5) / 0.5
        
        # バッチ次元追加
        batched = np.expand_dims(normalized, axis=0)
        
        return batched.astype(np.float32)
        
    def postprocess_gfpgan(self, output: np.ndarray, original_size: Tuple[int, int]) -> np.ndarray:
        """GFPGAN用後処理"""
        # バッチ次元除去
        if output.ndim == 4:
            output = output[0]
            
        # CHW → HWC
        hwc = output.transpose((1, 2, 0))
        
        # [-1, 1] → [0, 255]
        denormalized = (hwc.clip(-1, 1) + 1) * 0.5 * 255
        
        # RGB → BGR
        bgr = denormalized[:, :, ::-1]
        
        # 元サイズにリサイズ
        resized = cv2.resize(bgr, original_size)
        
        return resized.clip(0, 255).astype(np.uint8)
        
    def infer_gfpgan(self, image: np.ndarray, target_resolution: Tuple[int, int] = (512, 512)) -> np.ndarray:
        """
        GFPGAN推論実行（動的解像度対応）
        
        Args:
            image: 入力画像 (H, W, 3) BGR
            target_resolution: 処理解像度
            
        Returns:
            enhanced_image: 強化済み画像 (H, W, 3) BGR
        """
        original_size = (image.shape[1], image.shape[0])  # (W, H)
        
        # 前処理
        input_tensor = self.preprocess_gfpgan(image, target_resolution)
        input_shape = input_tensor.shape  # (1, 3, H, W)
        output_shape = input_shape  # GFPGAN は同じサイズ出力
        
        # 動的形状設定
        self.set_dynamic_shape(self.input_names[0], input_shape)
        
        # バッファ確保
        self._allocate_buffers(input_shape, output_shape)
        
        # GPU メモリ転送
        cuda.memcpy_htod(self.buffers['input'], input_tensor)
        
        # 推論実行
        bindings = [int(self.buffers['input']), int(self.buffers['output'])]
        self.context.execute_v2(bindings)
        
        # 結果取得
        output_tensor = np.empty(output_shape, dtype=np.float32)
        cuda.memcpy_dtoh(output_tensor, self.buffers['output'])
        
        # 後処理
        enhanced_image = self.postprocess_gfpgan(output_tensor, original_size)
        
        return enhanced_image
        
    def infer_wav2lip(self, mel_spectrogram: np.ndarray, video_frames: np.ndarray) -> np.ndarray:
        """
        Wav2Lip推論実行
        
        Args:
            mel_spectrogram: メルスペクトログラム (1, 1, 80, T)
            video_frames: 動画フレーム (1, 6, 96, 96)
            
        Returns:
            generated_frames: 生成フレーム (1, 3, 96, 96)
        """
        # 入力形状設定
        mel_shape = mel_spectrogram.shape
        video_shape = video_frames.shape
        output_shape = (1, 3, 96, 96)  # Wav2Lip出力形状
        
        # 動的形状設定（メルスペクトログラム）
        self.set_dynamic_shape(self.input_names[0], mel_shape)
        
        # バッファ確保（2つの入力 + 1つの出力）
        total_input_size = (np.prod(mel_shape) + np.prod(video_shape)) * np.dtype(np.float32).itemsize
        output_size = np.prod(output_shape) * np.dtype(np.float32).itemsize
        
        if total_input_size > self.buffer_sizes.get('input', 0):
            if 'input' in self.buffers:
                cuda.mem_free(self.buffers['input'])
            self.buffers['mel_input'] = cuda.mem_alloc(np.prod(mel_shape) * np.dtype(np.float32).itemsize)
            self.buffers['video_input'] = cuda.mem_alloc(np.prod(video_shape) * np.dtype(np.float32).itemsize)
            
        if output_size > self.buffer_sizes.get('output', 0):
            if 'output' in self.buffers:
                cuda.mem_free(self.buffers['output'])
            self.buffers['output'] = cuda.mem_alloc(output_size)
            self.buffer_sizes['output'] = output_size
            
        # GPU メモリ転送
        cuda.memcpy_htod(self.buffers['mel_input'], mel_spectrogram)
        cuda.memcpy_htod(self.buffers['video_input'], video_frames)
        
        # 推論実行
        bindings = [
            int(self.buffers['mel_input']), 
            int(self.buffers['video_input']), 
            int(self.buffers['output'])
        ]
        self.context.execute_v2(bindings)
        
        # 結果取得
        output_tensor = np.empty(output_shape, dtype=np.float32)
        cuda.memcpy_dtoh(output_tensor, self.buffers['output'])
        
        return output_tensor
        
    def benchmark(self, input_shape: Tuple[int, ...], iterations: int = 100) -> dict:
        """ベンチマーク実行"""
        # ダミー入力生成
        dummy_input = np.random.randn(*input_shape).astype(np.float32)
        
        # ウォームアップ
        for _ in range(10):
            if len(input_shape) == 4 and input_shape[1] == 3:  # GFPGAN
                dummy_image = (dummy_input[0].transpose(1, 2, 0) * 127.5 + 127.5).astype(np.uint8)
                self.infer_gfpgan(dummy_image, (input_shape[2], input_shape[3]))
                
        # 本測定
        cuda.Context.synchronize()
        start_time = time.time()
        
        for _ in range(iterations):
            if len(input_shape) == 4 and input_shape[1] == 3:  # GFPGAN
                dummy_image = (dummy_input[0].transpose(1, 2, 0) * 127.5 + 127.5).astype(np.uint8)
                self.infer_gfpgan(dummy_image, (input_shape[2], input_shape[3]))
                
        cuda.Context.synchronize()
        end_time = time.time()
        
        avg_time = (end_time - start_time) / iterations * 1000  # ms
        fps = 1000 / avg_time
        
        return {
            'avg_time_ms': avg_time,
            'fps': fps,
            'iterations': iterations,
            'input_shape': input_shape
        }
        
    def __del__(self):
        """リソース解放"""
        for buffer in self.buffers.values():
            if isinstance(buffer, cuda.DeviceAllocation):
                cuda.mem_free(buffer)

class DynamicResolutionManager:
    """動的解像度管理クラス"""
    
    def __init__(self, base_resolution: int = 512, min_resolution: int = 256, max_resolution: int = 1024):
        self.base_resolution = base_resolution
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.current_resolution = base_resolution
        self.performance_history = []
        
    def adjust_resolution(self, processing_time: float, target_fps: float = 30.0) -> int:
        """
        処理時間に基づいて解像度を動的調整
        
        Args:
            processing_time: 前回の処理時間（秒）
            target_fps: 目標FPS
            
        Returns:
            調整後の解像度
        """
        current_fps = 1.0 / processing_time if processing_time > 0 else target_fps
        
        # 性能履歴に追加
        self.performance_history.append(current_fps)
        if len(self.performance_history) > 10:
            self.performance_history.pop(0)
            
        # 平均FPSで判定
        avg_fps = np.mean(self.performance_history)
        
        if avg_fps < target_fps * 0.8:  # 目標の80%未満
            # 解像度を下げる
            self.current_resolution = max(self.min_resolution, 
                                        int(self.current_resolution * 0.9))
        elif avg_fps > target_fps * 1.2:  # 目標の120%超過
            # 解像度を上げる
            self.current_resolution = min(self.max_resolution,
                                        int(self.current_resolution * 1.1))
            
        # 32の倍数に調整（GPU効率化）
        self.current_resolution = (self.current_resolution // 32) * 32
        
        return self.current_resolution
        
    def get_optimal_resolution(self, face_size: Tuple[int, int]) -> int:
        """顔サイズに基づく最適解像度算出"""
        face_area = face_size[0] * face_size[1]
        
        if face_area < 64 * 64:
            return max(self.min_resolution, 256)
        elif face_area < 128 * 128:
            return max(self.min_resolution, 384)
        elif face_area < 256 * 256:
            return max(self.min_resolution, 512)
        else:
            return min(self.max_resolution, 768)

def create_tensorrt_engines():
    """TensorRT エンジン一括作成"""
    script_path = Path(__file__).parent / "tensorrt_engine_builder.py"
    
    # 全モデルを FP16 + 動的形状で構築
    cmd = f"python {script_path} --model both --dynamic --precision fp16 --optimize --benchmark"
    
    logger.info(f"🚀 TensorRT エンジン一括作成実行: {cmd}")
    os.system(cmd)

if __name__ == "__main__":
    # TensorRT エンジン作成
    create_tensorrt_engines()