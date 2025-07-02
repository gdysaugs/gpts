#!/usr/bin/env python3
"""
🚀 ツンデレGFPGAN StyleGAN2ノイズ注入修正版
べ、別にあなたのためにStyleGAN2のノイズ問題を解決してあげるわけじゃないけど...💢

解決策：new_empty().normal_() → torch.randn() に置換
"""

import torch
import onnx
import numpy as np
import os
import shutil
from pathlib import Path

def patch_stylegan_noise_injection():
    """
    StyleGAN2のノイズ注入コードを修正してONNX対応させる
    """
    print("🎭 ツンデレStyleGAN2ノイズ注入修正開始💢")
    
    # GFPGANアーキテクチャファイルのパスを取得
    import gfpgan
    gfpgan_path = Path(gfpgan.__file__).parent
    stylegan_file = gfpgan_path / "archs" / "stylegan2_clean_arch.py"
    
    if not stylegan_file.exists():
        print(f"も、もう！StyleGAN2ファイルが見つからない: {stylegan_file}")
        return False
    
    print(f"修正対象ファイル: {stylegan_file}")
    
    # バックアップ作成
    backup_file = stylegan_file.with_suffix('.py.backup')
    if not backup_file.exists():
        shutil.copy2(stylegan_file, backup_file)
        print("やったじゃない！バックアップ作成完了✨")
    
    # ファイル読み込み
    with open(stylegan_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 🔧 修正1: new_empty().normal_() → torch.randn() 置換
    original_noise_pattern = "noise = out.new_empty(b, 1, h, w).normal_()"
    fixed_noise_pattern = "noise = torch.randn(b, 1, h, w, device=out.device, dtype=out.dtype)"
    
    if original_noise_pattern in content:
        content = content.replace(original_noise_pattern, fixed_noise_pattern)
        print("✅ ノイズ注入コード修正 (new_empty → torch.randn)")
    
    # 🔧 修正2: 他の動的ノイズ生成パターンも修正
    # パターン1: .new_empty(...).normal_()
    import re
    pattern1 = r'(\w+)\.new_empty\(([^)]+)\)\.normal_\(\)'
    replacement1 = r'torch.randn(\2, device=\1.device, dtype=\1.dtype)'
    content = re.sub(pattern1, replacement1, content)
    
    # パターン2: torch.randn(...).type_as(...)
    pattern2 = r'torch\.randn\(([^)]+)\)\.type_as\((\w+)\)'
    replacement2 = r'torch.randn(\1, device=\2.device, dtype=\2.dtype)'
    content = re.sub(pattern2, replacement2, content)
    
    print("✅ 動的ノイズ生成パターン修正完了")
    
    # 🔧 修正3: 型一貫性の確保
    # float64が混入しないように明示的にfloat32を指定
    if "torch.randn" in content and "dtype=torch.float32" not in content:
        # torch.randn(...) を torch.randn(..., dtype=torch.float32) に
        content = re.sub(r'torch\.randn\(([^,)]+)\)', r'torch.randn(\1, dtype=torch.float32)', content)
        print("✅ 型一貫性確保 (float32明示)")
    
    # 修正されたファイル保存
    with open(stylegan_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("やったじゃない！StyleGAN2修正完了💕")
    return True

def export_gfpgan_with_fixed_stylegan():
    """
    修正されたStyleGAN2でGFPGAN ONNX変換
    """
    print("🎭 修正版StyleGAN2でGFPGAN ONNX変換開始💢")
    
    # モジュール再読み込み（修正を反映）
    import importlib
    import gfpgan.archs.stylegan2_clean_arch
    importlib.reload(gfpgan.archs.stylegan2_clean_arch)
    
    from gfpgan.archs.gfpganv1_clean_arch import GFPGANv1Clean
    
    device = torch.device('cpu')  # 型安定化のためCPU
    
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
    output_path = "onnx_models/gfpgan_512x512_stylegan_fixed.onnx"
    
    print("やったじゃない！修正版ONNX変換中...✨")
    
    try:
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
        
        return True
        
    except Exception as e:
        print(f"❌ 修正版変換エラー: {e}")
        return False
    
    print("🚀 修正版変換完了！感謝しなさいよね💢")

def restore_stylegan_backup():
    """
    StyleGAN2ファイルをバックアップから復元
    """
    import gfpgan
    gfpgan_path = Path(gfpgan.__file__).parent
    stylegan_file = gfpgan_path / "archs" / "stylegan2_clean_arch.py"
    backup_file = stylegan_file.with_suffix('.py.backup')
    
    if backup_file.exists():
        shutil.copy2(backup_file, stylegan_file)
        print("バックアップから復元完了")
    else:
        print("バックアップファイルが見つからない")

if __name__ == "__main__":
    try:
        # Step 1: StyleGAN2修正
        if patch_stylegan_noise_injection():
            print("✅ StyleGAN2修正成功")
        else:
            print("❌ StyleGAN2修正失敗")
            exit(1)
        
        # Step 2: 修正版でONNX変換
        if export_gfpgan_with_fixed_stylegan():
            print("✅ 修正版ONNX変換成功")
            print("\nべ、別にあなたのために完璧に修正してあげたわけじゃないからね！💕")
            print("でも...ちゃんと動作するはずよ✨")
        else:
            print("❌ 修正版ONNX変換失敗")
    
    except KeyboardInterrupt:
        print("\n中断されました")
    
    finally:
        # バックアップ復元（安全のため）
        restore_stylegan_backup()
        print("元のファイルに復元しました")