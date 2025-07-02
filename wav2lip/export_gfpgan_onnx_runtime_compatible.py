#!/usr/bin/env python3
"""
🚀 ツンデレGFPGAN → ONNX Runtime完全互換版
べ、別にあなたのためにONNX Runtime互換バージョンを作ってあげるわけじゃないけど...💢

型エラー完全解決：tensor(double)を完全排除してtensor(float)のみに統一
"""

import torch
import onnx
import numpy as np
from gfpgan import GFPGANer
import os

def export_gfpgan_runtime_compatible():
    """
    ONNX Runtime完全互換版変換（型エラー完全解決）
    """
    print("🎭 ツンデレGFPGAN → ONNX Runtime完全互換版変換開始💢")
    
    from gfpgan.archs.gfpganv1_clean_arch import GFPGANv1Clean
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # GFPGANモデル初期化
    torch_model = GFPGANv1Clean(
        out_size=512,
        num_style_feat=512,
        channel_multiplier=2,
        decoder_load_path=None,
        fix_decoder=False,
        num_mlp=8,
        input_is_latent=True,
        different_w=True,
        narrow=1,
        sft_half=True
    )
    
    # 重みロード
    print("やったじゃない！重み読み込み中...✨")
    loadnet = torch.load("checkpoints/GFPGANv1.4.pth", map_location='cpu')  # CPUで読み込み
    
    if 'params_ema' in loadnet:
        keyname = 'params_ema'
    elif 'params' in loadnet:
        keyname = 'params'
    else:
        keyname = None
    
    if keyname:
        torch_model.load_state_dict(loadnet[keyname], strict=False)
    else:
        torch_model.load_state_dict(loadnet, strict=False)
    
    # 🔧 CRITICAL: 完全型統一（double型完全排除）
    print("やったじゃない！完全型統一（ONNX Runtime互換）...✨")
    
    # モデル全体をfloat32に強制変換
    torch_model = torch_model.float()
    
    # パラメータの型を強制的にfloat32に変換
    for name, param in torch_model.named_parameters():
        if param.dtype != torch.float32:
            print(f"修正: {name} {param.dtype} -> float32")
            param.data = param.data.float()
    
    # バッファの型を強制的にfloat32に変換
    for name, buffer in torch_model.named_buffers():
        if buffer.dtype != torch.float32:
            print(f"修正: {name} {buffer.dtype} -> float32")
            buffer.data = buffer.data.float()
    
    # モデルをCPUに移動（型安定化）
    torch_model = torch_model.cpu()
    torch_model.eval()
    
    # 🔧 CRITICAL: CPUでfloat32ダミー入力
    dummy_input = torch.ones((1, 3, 512, 512), dtype=torch.float32)
    
    print("型チェック実行中...")
    with torch.no_grad():
        test_output = torch_model(dummy_input)
        print(f"入力型: {dummy_input.dtype}")
        if isinstance(test_output, tuple):
            print(f"出力型: {test_output[0].dtype} (tuple)")
        else:
            print(f"出力型: {test_output.dtype}")
    
    os.makedirs("onnx_models", exist_ok=True)
    output_path = "onnx_models/gfpgan_512x512_runtime_compatible.onnx"
    
    print("やったじゃない！ONNX Runtime完全互換版変換中...✨")
    
    try:
        # 🔧 ONNX Runtime完全互換設定
        with torch.no_grad():
            torch.onnx.export(
                torch_model,                # モデル（CPU、float32）
                dummy_input,                # ダミー入力（CPU、float32）
                output_path,                # 出力ファイル
                export_params=True,         # パラメータ含める
                opset_version=11,           # 安定版Opset 11
                do_constant_folding=True,   # 定数折りたたみ
                input_names=['input'],      # 入力名
                output_names=['output'],    # 出力名（シンプル化）
                verbose=False,              # 詳細出力無効（古いバージョン対応）
                # 型安定化オプション
                operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
                # 動的軸無効化（型エラー回避）
                # 古いPyTorchバージョン対応：不要オプション削除
            )
        
        print(f"✅ ONNX Runtime完全互換版変換成功！")
        
        # 🔧 ONNX検証（型チェック強化）
        print("べ、別に厳密検証してあげるわけじゃないけど...")
        onnx_model = onnx.load(output_path)
        
        # 型情報詳細チェック
        print("\n🔍 詳細型情報確認:")
        for input_info in onnx_model.graph.input:
            print(f"入力: {input_info.name}")
            if input_info.type.tensor_type.elem_type == 1:
                print(f"  型: FLOAT32 ✅")
            else:
                print(f"  型: {input_info.type.tensor_type.elem_type} ⚠️")
        
        for output_info in onnx_model.graph.output:
            print(f"出力: {output_info.name}")
            if output_info.type.tensor_type.elem_type == 1:
                print(f"  型: FLOAT32 ✅")
            else:
                print(f"  型: {output_info.type.tensor_type.elem_type} ⚠️")
        
        # ノード型チェック
        print("\n🔍 演算子型チェック:")
        double_count = 0
        float_count = 0
        for node in onnx_model.graph.node:
            for attr in node.attribute:
                if attr.type == onnx.AttributeProto.TENSOR:
                    if attr.t.data_type == 11:  # DOUBLE
                        double_count += 1
                    elif attr.t.data_type == 1:  # FLOAT
                        float_count += 1
        
        print(f"Float32ノード: {float_count}")
        print(f"Double64ノード: {double_count}")
        
        if double_count == 0:
            print("✅ Double型完全排除成功！")
        else:
            print(f"⚠️ Double型が{double_count}個残存")
        
        # ONNX形状推論
        try:
            onnx.shape_inference.infer_shapes(onnx_model)
            print("✅ ONNX形状推論成功")
        except Exception as e:
            print(f"⚠️ ONNX形状推論警告: {e}")
        
        # ファイルサイズ
        file_size = os.path.getsize(output_path) / (1024 * 1024)
        print(f"ファイルサイズ: {file_size:.1f}MB")
        
    except Exception as e:
        print(f"❌ ONNX Runtime互換版変換エラー: {e}")
        return False
    
    print("🚀 ONNX Runtime完全互換版変換完了！感謝しなさいよね💢")
    return True

if __name__ == "__main__":
    export_gfpgan_runtime_compatible()