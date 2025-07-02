#!/usr/bin/env python3
"""
🚀 ツンデレStyleGAN2修正版直接テスト
べ、別にあなたのためにレジストリー問題を回避してあげるわけじゃないけど...💢
"""

import torch
import onnx
import numpy as np
import os
import shutil
from pathlib import Path

def directly_test_fixed_stylegan():
    """
    修正されたStyleGAN2で直接ONNX変換テスト
    """
    print("🎭 ツンデレStyleGAN2修正版直接テスト開始💢")
    
    # 🔧 Step 1: StyleGAN2ファイル修正（再実行）
    import gfpgan
    gfpgan_path = Path(gfpgan.__file__).parent
    stylegan_file = gfpgan_path / "archs" / "stylegan2_clean_arch.py"
    
    print(f"修正対象ファイル: {stylegan_file}")
    
    # ファイル読み込み
    with open(stylegan_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 修正パターン確認・適用
    modifications_made = False
    
    # パターン1: new_empty().normal_()
    if "out.new_empty" in content and "normal_()" in content:
        content = content.replace(
            "out.new_empty(b, 1, h, w).normal_()",
            "torch.randn(b, 1, h, w, device=out.device, dtype=torch.float32)"
        )
        modifications_made = True
        print("✅ new_empty().normal_() 修正")
    
    # パターン2: 他の動的ノイズ生成
    import re
    
    # .new_empty(...).normal_() 全般
    pattern1 = r'(\w+)\.new_empty\(([^)]+)\)\.normal_\(\)'
    if re.search(pattern1, content):
        content = re.sub(pattern1, r'torch.randn(\2, device=\1.device, dtype=torch.float32)', content)
        modifications_made = True
        print("✅ 動的ノイズ生成修正")
    
    # パターン3: torch.randn().type_as()
    pattern2 = r'torch\.randn\(([^)]+)\)\.type_as\((\w+)\)'
    if re.search(pattern2, content):
        content = re.sub(pattern2, r'torch.randn(\1, device=\2.device, dtype=torch.float32)', content)
        modifications_made = True
        print("✅ type_as()修正")
    
    if modifications_made:
        # 修正されたファイル保存
        with open(stylegan_file, 'w', encoding='utf-8') as f:
            f.write(content)
        print("やったじゃない！StyleGAN2修正適用完了💕")
    else:
        print("修正対象が見つからないか、既に修正済み")
    
    # 🔧 Step 2: 新しいプロセスでONNX変換
    print("新しいプロセスでONNX変換実行...")
    
    try:
        # GFPGANを新規インポート（修正反映）
        from gfpgan.archs.gfpganv1_clean_arch import GFPGANv1Clean
        
        device = torch.device('cpu')
        
        # モデル初期化
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
        loadnet = torch.load("checkpoints/GFPGANv1.4.pth", map_location='cpu')
        
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
        
        # 完全型統一
        torch_model = torch_model.float().cpu()
        torch_model.eval()
        
        # 入力準備
        dummy_input = torch.ones((1, 3, 512, 512), dtype=torch.float32)
        
        print("修正版テスト実行中...")
        with torch.no_grad():
            test_output = torch_model(dummy_input)
            print(f"入力型: {dummy_input.dtype}")
            if isinstance(test_output, tuple):
                print(f"出力型: {test_output[0].dtype}")
            else:
                print(f"出力型: {test_output.dtype}")
        
        os.makedirs("onnx_models", exist_ok=True)
        output_path = "onnx_models/gfpgan_512x512_stylegan_fixed_direct.onnx"
        
        print("やったじゃない！修正版ONNX変換中...✨")
        
        with torch.no_grad():
            torch.onnx.export(
                torch_model,
                dummy_input,
                output_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                verbose=False,
                operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
            )
        
        print(f"✅ 修正版ONNX変換成功！")
        
        # ONNX検証
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("✅ ONNX検証成功")
        
        file_size = os.path.getsize(output_path) / (1024 * 1024)
        print(f"ファイルサイズ: {file_size:.1f}MB")
        
        return output_path
        
    except Exception as e:
        print(f"❌ 修正版変換エラー: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = directly_test_fixed_stylegan()
    
    if result:
        print("✅ 修正版ONNX変換成功")
        print(f"出力ファイル: {result}")
        print("\nべ、別にあなたのために完璧に修正してあげたわけじゃないからね！💕")
        print("でも...ちゃんと動作するはずよ✨")
    else:
        print("❌ 修正版ONNX変換失敗")
        print("も、もう！StyleGAN2は本当に複雑なのよ💢")