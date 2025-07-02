#!/usr/bin/env python3
"""
🚀 ツンデレGFPGAN ONNX型エラー完全修正版
べ、別にあなたのためにONNX型エラーを根本解決してあげるわけじゃないけど...💢

最終アプローチ：モデル重みを手動でfloat32に変換してからONNX変換
"""

import torch
import onnx
import numpy as np
import os
from collections import OrderedDict

def completely_fix_onnx_types():
    """
    GFPGAN ONNX型エラー完全修正（手動重み変換版）
    """
    print("🎭 ツンデレGFPGAN ONNX型エラー完全修正開始💢")
    
    from gfpgan.archs.gfpganv1_clean_arch import GFPGANv1Clean
    
    device = torch.device('cpu')  # CPUで型統一
    
    # 🔧 Step 1: モデル初期化
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
    
    # 🔧 Step 2: 重みロード
    print("やったじゃない！重み読み込み中...✨")
    loadnet = torch.load("checkpoints/GFPGANv1.4.pth", map_location='cpu')
    
    if 'params_ema' in loadnet:
        keyname = 'params_ema'
    elif 'params' in loadnet:
        keyname = 'params'
    else:
        keyname = None
    
    if keyname:
        state_dict = loadnet[keyname]
    else:
        state_dict = loadnet
    
    # 🔧 Step 3: 手動型変換（重み辞書レベル）
    print("やったじゃない！手動型変換中（完全float32化）...✨")
    new_state_dict = OrderedDict()
    
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            # 全テンソルをfloat32に強制変換
            new_value = value.float()
            new_state_dict[key] = new_value
            if value.dtype != torch.float32:
                print(f"修正: {key} {value.dtype} -> float32")
        else:
            new_state_dict[key] = value
    
    # モデルに変換済み重みロード
    torch_model.load_state_dict(new_state_dict, strict=False)
    
    # 🔧 Step 4: モデル全体float32確認
    torch_model = torch_model.float().cpu()
    torch_model.eval()
    
    # バッファも確認
    for name, buffer in torch_model.named_buffers():
        if buffer.dtype != torch.float32:
            print(f"バッファ修正: {name} {buffer.dtype} -> float32")
            buffer.data = buffer.data.float()
    
    # 🔧 Step 5: 入力もfloat32で統一
    dummy_input = torch.ones((1, 3, 512, 512), dtype=torch.float32)
    
    print("最終型チェック実行中...")
    with torch.no_grad():
        test_output = torch_model(dummy_input)
        print(f"入力型: {dummy_input.dtype}")
        if isinstance(test_output, tuple):
            print(f"出力型: {test_output[0].dtype} (tuple)")
        else:
            print(f"出力型: {test_output.dtype}")
    
    os.makedirs("onnx_models", exist_ok=True)
    output_path = "onnx_models/gfpgan_512x512_final_fix.onnx"
    
    print("やったじゃない！完全修正版ONNX変換中...✨")
    
    try:
        # 🔧 最もシンプルな設定で変換
        with torch.no_grad():
            torch.onnx.export(
                torch_model,
                dummy_input,
                output_path,
                export_params=True,
                opset_version=11,  # 安定版
                do_constant_folding=False,  # 定数折りたたみ無効（型安定化）
                input_names=['input'],
                output_names=['output'],
                verbose=False,
                # 最小限設定
                operator_export_type=torch.onnx.OperatorExportTypes.ONNX
            )
        
        print(f"✅ 完全修正版ONNX変換成功！")
        
        # 🔧 ONNX後処理（型強制修正）
        print("べ、別にONNX後処理してあげるわけじゃないけど...")
        
        # ONNXモデル読み込み
        onnx_model = onnx.load(output_path)
        
        # 🔧 重要：initializer の型を強制的にfloat32に変換
        for initializer in onnx_model.graph.initializer:
            if initializer.data_type == 11:  # DOUBLE
                print(f"initializer修正: {initializer.name} DOUBLE -> FLOAT")
                # double を float に変換
                double_data = np.frombuffer(initializer.raw_data, dtype=np.float64)
                float_data = double_data.astype(np.float32)
                initializer.raw_data = float_data.tobytes()
                initializer.data_type = 1  # FLOAT
        
        # 修正版保存
        output_path_fixed = "onnx_models/gfpgan_512x512_completely_fixed.onnx"
        onnx.save(onnx_model, output_path_fixed)
        
        print(f"✅ 完全修正版保存完了: {output_path_fixed}")
        
        # 最終検証
        print("\n🔍 最終検証:")
        final_model = onnx.load(output_path_fixed)
        
        double_count = 0
        float_count = 0
        for initializer in final_model.graph.initializer:
            if initializer.data_type == 11:  # DOUBLE
                double_count += 1
            elif initializer.data_type == 1:  # FLOAT
                float_count += 1
        
        print(f"Float32初期化子: {float_count}")
        print(f"Double64初期化子: {double_count}")
        
        if double_count == 0:
            print("✅ Double型完全排除成功！")
        else:
            print(f"⚠️ Double型が{double_count}個残存")
        
        file_size = os.path.getsize(output_path_fixed) / (1024 * 1024)
        print(f"ファイルサイズ: {file_size:.1f}MB")
        
        return True
        
    except Exception as e:
        print(f"❌ 完全修正版変換エラー: {e}")
        return False
    
    print("🚀 完全修正版完了！感謝しなさいよね💢")

if __name__ == "__main__":
    completely_fix_onnx_types()