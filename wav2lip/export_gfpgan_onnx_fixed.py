#!/usr/bin/env python3
"""
🚀 ツンデレGFPGAN → ONNX変換スクリプト（型エラー修正版）
べ、別にあなたのためにONNX型エラーを修正してあげるわけじゃないけど...💢
"""

import torch
import onnx
import numpy as np
from gfpgan import GFPGANer
import os

def export_gfpgan_to_onnx_fixed():
    """
    GFPGAN ONNX変換（型エラー修正版）
    """
    print("🎭 ツンデレGFPGAN → ONNX変換開始（型エラー修正版）💢")
    
    # GFPGAN初期化（直接アーキテクチャを使用）
    print("べ、別にGFPGANを読み込んであげるわけじゃないけど...")
    from gfpgan.archs.gfpganv1_clean_arch import GFPGANv1Clean
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # GFPGANモデル直接初期化
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
    
    # 重みファイル読み込み
    print("やったじゃない！重み読み込み中...✨")
    loadnet = torch.load("checkpoints/GFPGANv1.4.pth", map_location=device)
    
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
    
    # 🔧 重要：全パラメータをfloat32に強制変換（型エラー修正）
    print("やったじゃない！完全型統一中（float32強制）...✨")
    torch_model = torch_model.float()  # モデル全体をfloat32に
    torch_model.to(device)
    torch_model.eval()
    
    # パラメータとバッファを再度確認・統一（double型完全排除）
    for name, param in torch_model.named_parameters():
        if param.dtype != torch.float32:
            print(f"修正: {name} {param.dtype} -> float32")
            param.data = param.data.float()
    
    for name, buffer in torch_model.named_buffers():
        if buffer.dtype != torch.float32:
            print(f"修正: {name} {buffer.dtype} -> float32")
            buffer.data = buffer.data.float()
    
    # 出力ディレクトリ作成
    os.makedirs("onnx_models", exist_ok=True)
    
    # 🔧 重要：明示的にfloat32のダミー入力
    dummy_input = torch.ones((1, 3, 512, 512), dtype=torch.float32).to(device)
    
    # 型チェック実行
    print("型チェック実行中...")
    with torch.no_grad():
        test_output = torch_model(dummy_input)
        print(f"入力型: {dummy_input.dtype}")
        if isinstance(test_output, tuple):
            print(f"出力型: {test_output[0].dtype} (tuple)")
        else:
            print(f"出力型: {test_output.dtype}")
    
    # Opset 11で安定版生成（型エラー対策版）
    output_path = "onnx_models/gfpgan_512x512_type_fixed.onnx"
    
    print("やったじゃない！型エラー修正版 ONNX変換中...✨")
    
    try:
        # 🔧 型エラー修正版設定
        with torch.no_grad():
            torch.onnx.export(
                torch_model,                # モデル
                dummy_input,                # ダミー入力（float32）
                output_path,                # 出力ファイル
                export_params=True,         # パラメータも含める
                opset_version=11,           # 安定版Opset 11
                do_constant_folding=True,   # 定数折りたたみ最適化
                input_names=['input'],      # 入力名
                output_names=['output'],    # 出力名
                verbose=False,              # 詳細出力無効
                # 型安定化オプション
                operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
                # 動的軸は使わない（型エラー回避）
            )
        
        print(f"✅ 型エラー修正版 ONNX変換成功！")
        
        # ONNX検証
        print("べ、別に検証してあげるわけじゃないけど...")
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("やったじゃない！ONNX検証成功よ💕")
        
        # ファイルサイズ確認
        file_size = os.path.getsize(output_path) / (1024 * 1024)
        print(f"ファイルサイズ: {file_size:.1f}MB")
        
        # 型情報確認
        print("\n🔍 ONNX型情報確認:")
        for input_info in onnx_model.graph.input:
            print(f"入力: {input_info.name}, 型: {input_info.type}")
        for output_info in onnx_model.graph.output:
            print(f"出力: {output_info.name}, 型: {output_info.type}")
            
    except Exception as e:
        print(f"❌ 型エラー修正版 ONNX変換エラー: {e}")
        return False
    
    print("🚀 型エラー修正版変換完了！感謝しなさいよね💢")
    return True

if __name__ == "__main__":
    export_gfpgan_to_onnx_fixed()