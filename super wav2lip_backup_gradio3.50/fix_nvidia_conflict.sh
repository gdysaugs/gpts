#!/bin/bash
# NVIDIA競合修正スクリプト

echo "🔧 NVIDIA競合ライブラリ削除開始..."

# 競合するNVIDIAライブラリファイルを削除（TensorRT/PyCUDAは保持）
rm -f /usr/lib/x86_64-linux-gnu/libnvidia-ml.so*
rm -f /usr/lib/x86_64-linux-gnu/libnvidia-cfg.so*
rm -f /usr/lib/x86_64-linux-gnu/libnvidia-compiler.so*

# 競合するCUDAランタイムライブラリを削除
rm -f /usr/lib/x86_64-linux-gnu/libcuda.so*
rm -f /usr/lib/x86_64-linux-gnu/libcudart.so*

echo "✅ 競合ライブラリ削除完了"
echo "🚀 TensorRT/PyCUDAは保持"

# 確認
echo "🔍 残存TensorRTライブラリ:"
find /usr/local/lib/python3.10/dist-packages -name "*tensorrt*" | head -3
echo "🔍 残存PyCUDAライブラリ:"
find /usr/local/lib/python3.10/dist-packages -name "*pycuda*" | head -3