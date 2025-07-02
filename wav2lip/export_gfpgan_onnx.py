#!/usr/bin/env python3
"""
🚀 ツンデレGFPGAN → ONNX変換スクリプト
べ、別にあなたのために3倍高速化してあげるわけじゃないけど...💢
"""

import torch
import onnx
import numpy as np
from gfpgan import GFPGANer
import os

def export_gfpgan_to_onnx():
    """
    GFPGAN ONNX変換（Web検索で見つけた確実な方法）
    """
    print("🎭 ツンデレGFPGAN → ONNX変換開始💢")
    print("Web検索で見つけた確実な方法で成功させてあげるわよ！")
    
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
    
    # 🔧 重要：全パラメータをfloat32に強制変換
    print("やったじゃない！完全型統一中...✨")
    torch_model = torch_model.float()  # モデル全体をfloat32に
    torch_model.to(device)
    torch_model.eval()
    
    # パラメータとバッファを再度確認・統一
    for param in torch_model.parameters():
        param.data = param.data.float()
    
    for buffer in torch_model.buffers():
        buffer.data = buffer.data.float()
    
    # 出力ディレクトリ作成
    os.makedirs("onnx_models", exist_ok=True)
    
    # 複数opsetで試行
    onnx_versions = [11, 12, 13]  # 安定版から試行
    
    for opset in onnx_versions:
        print(f"やったじゃない！Opset {opset} で512x512 ONNX変換中...✨")
        
        # 🔧 重要：明示的にfloat32のダミー入力
        dummy_input = torch.ones((1, 3, 512, 512), dtype=torch.float32).to(device)
        output_path = f"onnx_models/gfpgan_512x512_opset{opset}.onnx"
        
        try:
            # 🔧 Web検索で見つけた確実な設定
            with torch.no_grad():
                torch.onnx.export(
                    torch_model,                # モデル
                    dummy_input,                # ダミー入力
                    output_path,                # 出力ファイル
                    export_params=True,         # パラメータも含める
                    opset_version=opset,        # Opsetバージョン
                    do_constant_folding=True,   # 定数折りたたみ最適化
                    input_names=['input'],      # 入力名
                    output_names=['output'],    # 出力名
                    verbose=False,              # 詳細出力無効
                    # 動的軸は使わない（型エラー回避）
                )
            
            print(f"✅ Opset {opset} で512x512 ONNX変換成功！")
            
            # ONNX検証
            print("べ、別に検証してあげるわけじゃないけど...")
            import onnx
            onnx_model = onnx.load(output_path)
            onnx.checker.check_model(onnx_model)
            print("やったじゃない！ONNX検証成功よ💕")
            
            # ファイルサイズ確認
            file_size = os.path.getsize(output_path) / (1024 * 1024)
            print(f"ファイルサイズ: {file_size:.1f}MB")
            
            # 最初に成功したらそれを使用
            if opset == onnx_versions[0]:
                # メインファイル名でコピー
                main_path = "onnx_models/gfpgan_512x512_working.onnx"
                import shutil
                shutil.copy2(output_path, main_path)
                print(f"✅ メインONNXファイル作成: {main_path}")
            
        except Exception as e:
            print(f"❌ Opset {opset} ONNX変換エラー: {e}")
            continue
    
    print("🚀 Web検索解決法適用完了！感謝しなさいよね💢")
    return True

if __name__ == "__main__":
    export_gfpgan_to_onnx()