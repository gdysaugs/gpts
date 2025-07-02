#!/usr/bin/env python3
"""
🎭 ツンデレCodeFormer 高速統合版
べ、別にあなたのためにCodeFormerを実装してあげるわけじゃないけど...💢

CodeFormerを使用した顔画質向上モジュール
langzizhixin/Wav2Lip-CodeFormer と sczhou/CodeFormer を参考にした実装
"""

import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import onnxruntime as ort
import numpy as np
import cv2
import os
import sys
import math
from pathlib import Path
from typing import Optional, Tuple

warnings.filterwarnings("ignore")


# CodeFormer関連のimport
try:
    from basicsr.utils import imwrite, img2tensor, tensor2img
    from basicsr.utils.download_util import load_file_from_url
    from basicsr.utils.registry import ARCH_REGISTRY
    from facelib.utils.face_restoration_helper import FaceRestoreHelper
    from facelib.utils.misc import is_gray
    
    # gpu_is_availableが見つからない場合の代替
    try:
        from basicsr.utils.misc import gpu_is_available, get_device
    except ImportError:
        # フォールバック：torch版を使用
        def gpu_is_available():
            return torch.cuda.is_available()
        
        def get_device():
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    CODEFORMER_AVAILABLE = True
    print("🚀 CodeFormer依存関係OK！")
except ImportError as e:
    print(f"❌ CodeFormer依存関係エラー: {e}")
    try:
        # フォールバック：GFPGANインポート
        from gfpgan import GFPGANer
        GFPGAN_AVAILABLE = True
        CODEFORMER_AVAILABLE = False
        print("🚀 GFPGANフォールバック利用可能")
    except ImportError:
        CODEFORMER_AVAILABLE = False
        GFPGAN_AVAILABLE = False
        print("❌ CodeFormer/GFPGAN両方が利用不可")

# CodeFormerモデルURL
pretrain_model_url = {
    'restoration': 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth'
}

def load_codeformer():
    """
    CodeFormerモデル読み込み（GPU最適化）
    """
    if not CODEFORMER_AVAILABLE:
        print("❌ CodeFormer利用不可：依存関係が不足")
        # GFPGANフォールバックを試行
        if 'GFPGAN_AVAILABLE' in globals() and GFPGAN_AVAILABLE:
            return load_gfpgan_fallback()
        return {'type': 'none', 'model': None}
    
    try:
        print("🚀 CodeFormer初期化中...（GPU最適化）💕")
        
        # デバイス設定
        device = get_device()
        print(f"デバイス: {device}")
        
        # CodeFormerネットワーク初期化
        net = ARCH_REGISTRY.get('CodeFormer')(
            dim_embd=512, 
            codebook_size=1024, 
            n_head=8, 
            n_layers=9, 
            connect_list=['32', '64', '128', '256']
        ).to(device)
        
        # モデルダウンロード/ロード
        model_dir = '/app/checkpoints/CodeFormer'
        os.makedirs(model_dir, exist_ok=True)
        
        ckpt_path = load_file_from_url(
            url=pretrain_model_url['restoration'], 
            model_dir=model_dir, 
            progress=True,
            file_name='codeformer.pth'
        )
        
        if not os.path.exists(ckpt_path):
            # フォールバック：ローカルパス
            fallback_paths = [
                '/app/checkpoints/CodeFormer/codeformer.pth',
                '/app/checkpoints/codeformer.pth',
                '/app/models/CodeFormer/codeformer.pth',
                'checkpoints/CodeFormer/codeformer.pth',
                'checkpoints/codeformer.pth',
                'models/CodeFormer/codeformer.pth'
            ]
            for path in fallback_paths:
                if os.path.exists(path):
                    ckpt_path = path
                    break
            else:
                print("❌ CodeFormerモデルが見つからない、GFPGANフォールバック")
                return load_gfpgan_fallback()
        
        # チェックポイント読み込み
        checkpoint = torch.load(ckpt_path, map_location=device)
        net.load_state_dict(checkpoint['params_ema'])
        net.eval()
        
        # 顔検出ヘルパー初期化（軽量モード）
        face_helper = FaceRestoreHelper(
            upscale_factor=1,
            face_size=512,
            crop_ratio=(1, 1),
            det_model='retinaface_resnet50',
            save_ext='png',
            use_parse=True,
            device=device
        )
        
        print("やったじゃない！CodeFormer初期化完了よ✨")
        
        return {
            'type': 'codeformer',
            'model': net,
            'face_helper': face_helper,
            'device': device
        }
        
    except Exception as e:
        print(f"❌ CodeFormer初期化失敗: {e}")
        print("フォールバック：GFPGAN使用")
        return load_gfpgan_fallback()

def load_gfpgan_fallback():
    """
    GFPGANフォールバック初期化
    """
    try:
        from gfpgan import GFPGANer
        print("🚀 GFPGANフォールバック初期化中...💢")
        
        gfpgan_model = GFPGANer(
            model_path="/app/checkpoints/GFPGANv1.4.pth",
            upscale=1,
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=None,
        )
        
        print("やったじゃない！GFPGANフォールバック成功よ✨")
        return {'type': 'gfpgan_fallback', 'model': gfpgan_model}
        
    except Exception as e:
        print(f"❌ GFPGANフォールバックも失敗: {e}")
        print("も、もう！どちらも使えないじゃない💢")
        return {'type': 'none', 'model': None}


def enhance_with_codeformer(image, run_params, fidelity_weight=0.7):
    """
    CodeFormerで顔画質向上（GFPGAN代替）
    """
    if run_params['type'] != 'codeformer':
        if run_params['type'] == 'gfpgan_fallback':
            print("GFPGANフォールバック使用")
            return enhance_gfpgan_fallback(image, run_params['model'])
        else:
            print("CodeFormer未初期化：元画像を返却")
            return image
    
    try:
        net = run_params['model']
        face_helper = run_params['face_helper']
        device = run_params['device']
        
        # 元のサイズ保存
        original_h, original_w = image.shape[:2]
        
        # 顔検出
        face_helper.clean_all()
        
        # OpenCVのBGR形式をRGB形式に変換
        if len(image.shape) == 3:
            if is_gray(image):
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            input_img = image
        
        face_helper.read_image(input_img)
        face_helper.get_face_landmarks_5(only_center_face=False, resize=640, eye_dist_threshold=5)
        face_helper.align_warp_face()
        
        # 顔が検出されない場合は元画像を返す
        if len(face_helper.cropped_faces) == 0:
            print("顔が検出されなかった：元画像を返却")
            return image
        
        # 検出された全ての顔を処理
        for idx, cropped_face in enumerate(face_helper.cropped_faces):
            # 前処理：テンソル変換と正規化
            cropped_face_t = img2tensor(cropped_face / 255., bgr2rgb=True, float32=True)
            # CodeFormerの正規化：[-1, 1]
            cropped_face_t = cropped_face_t.unsqueeze(0).to(device)
            # 正規化
            cropped_face_t = (cropped_face_t - 0.5) / 0.5
            
            try:
                # CodeFormer推論（GPU最適化）
                with torch.no_grad():
                    with torch.cuda.amp.autocast(enabled=device.type == 'cuda'):
                        output = net(cropped_face_t, w=fidelity_weight, adain=True)[0]
                        # 正規化を戻す：[-1, 1] -> [0, 1]
                        output = (output + 1) / 2
                        restored_face = tensor2img(output, rgb2bgr=True, min_max=(0, 1))
                        del output
                        torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"CodeFormer推論エラー: {e}")
                restored_face = cropped_face
            
            face_helper.add_restored_face(restored_face, cropped_face)
        
        # 結果をオリジナル画像に貼り付け
        face_helper.get_inverse_affine(None)
        restored_img = face_helper.paste_faces_to_input_image()
        
        # 元のサイズに戻す
        if restored_img.shape[:2] != (original_h, original_w):
            restored_img = cv2.resize(restored_img, (original_w, original_h), interpolation=cv2.INTER_LANCZOS4)
        
        # RGB -> BGR 変換（OpenCV形式）
        if len(restored_img.shape) == 3:
            restored_img = cv2.cvtColor(restored_img, cv2.COLOR_RGB2BGR)
        
        return restored_img
        
    except Exception as e:
        print(f"❌ CodeFormer処理エラー: {e}")
        return image

def enhance_gfpgan_fallback(image, gfpgan_model):
    """
    GFPGANフォールバック処理（CodeFormer代替）
    """






    try:
        # GFPGAN処理
        _, _, output = gfpgan_model.enhance(
            image, has_aligned=False, only_center_face=False, paste_back=True
        )
        
        if output is None:
            print("も、もう！GFPGAN処理に失敗したから元画像を返すわよ💢")
            return image
        
        return output
        
    except Exception as e:
        print(f"GFPGANフォールバック処理エラー: {e}")
        return image


def upscale_codeformer(image, properties, fidelity_weight=0.7):
    """
    CodeFormer画質向上処理（互換性関数）
    """
    return enhance_with_codeformer(image, properties, fidelity_weight)


# 後方互換性のためのエイリアス
def load_sr():
    """
    後方互換性のためのエイリアス（GFPGAN -> CodeFormer）
    """
    return load_codeformer()


def upscale(image, properties):
    """
    後方互換性のためのエイリアス（GFPGAN -> CodeFormer）
    """
    return enhance_with_codeformer(image, properties)


def normalize(tensor, mean, std, inplace=False):
    """テンソル正規化（CodeFormer用）"""
    if not inplace:
        tensor = tensor.clone()
    
    dtype = tensor.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=dtype, device=tensor.device)
    tensor.sub_(mean[None, :, None, None]).div_(std[None, :, None, None])
    return tensor


if __name__ == "__main__":
    # テスト実行
    print("🎭 CodeFormerモジュールテスト開始")
    
    # モデル読み込みテスト
    run_params = load_codeformer()
    if run_params['type'] == 'codeformer':
        print("✅ CodeFormer読み込み成功")
    elif run_params['type'] == 'gfpgan_fallback':
        print("✅ GFPGANフォールバック成功")
    else:
        print("❌ 顔画質向上モジュールが利用できない")
    
    print("🎭 CodeFormerモジュールテスト完了")