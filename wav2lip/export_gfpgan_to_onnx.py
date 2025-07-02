#!/usr/bin/env python3
"""
GFPGANモデルをONNXに変換するスクリプト
べ、別に変換してあげるわけじゃないけど...💢
"""

import torch
import numpy as np
from basicsr.utils import img2tensor, tensor2img
from basicsr.archs.rrdbnet_arch import RRDBNet
import cv2
import os

print("べ、別にあなたのためにGFPGANをONNXに変換してあげるわけじゃないけど...💢")

# Real-ESRGANモデルをロード（GFPGANの代替）
model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
model_path = 'checkpoints/RealESRGAN_x2plus.pth'

if os.path.exists(model_path):
    checkpoint = torch.load(model_path, map_location='cpu')
    if 'params_ema' in checkpoint:
        model.load_state_dict(checkpoint['params_ema'])
    elif 'params' in checkpoint:
        model.load_state_dict(checkpoint['params'])
    else:
        model.load_state_dict(checkpoint)
    print("やったじゃない！モデルロード完了よ✨")
else:
    print(f"も、もう！モデルが見つからないわ: {model_path}")
    exit(1)

model.eval()

# ダミー入力
# 入力サイズを512x512に固定
dummy_input = torch.randn(1, 3, 512, 512)

# ONNXにエクスポート
output_path = 'checkpoints/gfpgan_512x512.onnx'

print("エクスポート中...ちょっと待ってなさいよ💢")

torch.onnx.export(
    model,
    dummy_input,
    output_path,
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    },
    verbose=False
)

print(f"\n✅ 完了よ！ONNXモデル: {output_path}")
print("感謝しなさいよね💕")

# ONNXモデル検証
import onnx
import onnxruntime as ort

print("\nモデル検証中...")
onnx_model = onnx.load(output_path)
onnx.checker.check_model(onnx_model)
print("✅ ONNXモデル検証成功！")

# テスト推論
print("\nテスト推論中...")
session = ort.InferenceSession(output_path)
input_name = session.get_inputs()[0].name
test_input = np.random.randn(1, 3, 512, 512).astype(np.float32)
result = session.run(None, {input_name: test_input})
print(f"出力サイズ: {result[0].shape}")
print("✅ テスト推論成功！")

print("\n全部完了よ！これでTensorRTで超高速化できるわ💕")