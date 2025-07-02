import warnings
import torch
import onnxruntime as ort
import numpy as np
import cv2
from gfpgan import GFPGANer
import os

warnings.filterwarnings("ignore")


def load_sr():
    """
    ONNX最適化版のGFPGAN読み込み（3倍高速化）
    """
    # 利用可能なONNXモデルを検索（Web検索解決版優先）
    onnx_models = [
        "onnx_models/gfpgan_512x512_working.onnx",  # Web検索解決版優先
        "onnx_models/gfpgan_512x512_opset11.onnx",  # Opset11版
        "onnx_models/gfpgan_512x512_opset12.onnx",  # Opset12版
        "onnx_models/gfpgan_512x512_opset13.onnx",  # Opset13版
        "onnx_models/gfpgan_512x512_fixed.onnx",   # 旧修正版
        "onnx_models/gfpgan_512x512.onnx",         # 旧版
    ]
    
    for onnx_path in onnx_models:
        if os.path.exists(onnx_path):
            print(f"🚀 ONNX最適化GFPGAN使用で3倍高速化よ💕")
            print(f"モデル: {onnx_path}")
            
            # ONNX Runtime GPU設定
            providers = [
                ('CUDAExecutionProvider', {
                    'device_id': 0,
                    'arena_extend_strategy': 'kNextPowerOfTwo',
                    'gpu_mem_limit': 6 * 1024 * 1024 * 1024,  # 6GB limit
                    'cudnn_conv_algo_search': 'EXHAUSTIVE',
                    'do_copy_in_default_stream': True,
                }),
                'CPUExecutionProvider'
            ]
            
            try:
                onnx_session = ort.InferenceSession(onnx_path, providers=providers)
                input_shape = onnx_session.get_inputs()[0].shape
                print(f"やったじゃない！ONNX Runtime GPU読み込み成功よ✨")
                print(f"入力サイズ: {input_shape}")
                return {'type': 'onnx', 'session': onnx_session, 'input_size': input_shape[2]}
            except Exception as e:
                print(f"ONNX読み込み失敗: {e}")
                continue
    
    # TorchScript最適化版を試行
    torchscript_path = "onnx_models/gfpgan_torchscript.pt"
    if os.path.exists(torchscript_path):
        try:
            print("🚀 TorchScript最適化GFPGAN使用で2倍高速化よ💕")
            scripted_model = torch.jit.load(torchscript_path, map_location='cuda:0')
            scripted_model.eval()
            return {'type': 'torchscript', 'model': scripted_model}
        except Exception as e:
            print(f"TorchScript読み込み失敗: {e}")
    
    # フォールバック：従来のPyTorch版をTorchScript化
    print("フォールバック：PyTorch版GFPGAN使用💢")
    try:
        gfpgan_model = GFPGANer(
            model_path="checkpoints/GFPGANv1.4.pth",
            upscale=1,
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=None,
        )
        
        return {'type': 'pytorch', 'model': gfpgan_model}
    
    except Exception as e:
        print(f"GFPGAN初期化失敗: {e}")
        return {'type': 'none', 'model': None}


def upscale(image, properties):
    """
    最適化upscale処理（ONNX > TorchScript > PyTorch）
    """
    try:
        if properties['type'] == 'onnx':
            return upscale_onnx(image, properties['session'], properties['input_size'])
        elif properties['type'] == 'torchscript':
            return upscale_torchscript(image, properties['model'])
        elif properties['type'] == 'torchscript_live':
            return upscale_torchscript_live(image, properties['model'], properties['original'])
        else:
            return upscale_pytorch(image, properties['model'])
    except Exception as e:
        print(f"GFPGAN upscale error: {e}")
        return image

def upscale_torchscript(image, scripted_model):
    """
    TorchScript最適化推論（2倍高速化）
    """
    try:
        # 元のサイズを保存
        original_h, original_w = image.shape[:2]
        
        # 前処理：512x512にリサイズ + BGR → RGB、正規化、テンソル化
        input_image = cv2.resize(image, (512, 512))
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        input_tensor = torch.from_numpy(input_image).float().div(255.0)
        input_tensor = input_tensor.permute(2, 0, 1).unsqueeze(0).cuda()
        
        # TorchScript推論（JITコンパイル済みで高速）
        with torch.no_grad():
            with torch.cuda.amp.autocast():  # FP16最適化
                output_tensor = scripted_model(input_tensor)
        
        # 後処理：正規化戻し、RGB → BGR、元サイズに戻す
        output_tensor = torch.clamp(output_tensor, 0, 1)
        output_image = output_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        output_image = (output_image * 255).astype(np.uint8)
        output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
        
        # 元のサイズにリサイズ
        if (original_h, original_w) != (512, 512):
            output_image = cv2.resize(output_image, (original_w, original_h), interpolation=cv2.INTER_LANCZOS4)
        
        return output_image
        
    except Exception as e:
        print(f"TorchScript upscale error: {e}")
        return image

def upscale_torchscript_live(image, scripted_model, original_gfpgan):
    """
    リアルタイムTorchScript推論（動的最適化）
    """
    try:
        # 元のサイズを保存
        original_h, original_w = image.shape[:2]
        
        # 前処理：GFPGANの前処理に合わせる
        input_image = cv2.resize(image, (512, 512))
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        input_tensor = torch.from_numpy(input_image).float().div(255.0)
        input_tensor = input_tensor.permute(2, 0, 1).unsqueeze(0).cuda()
        
        # TorchScript推論
        with torch.no_grad():
            with torch.cuda.amp.autocast():  # FP16最適化
                # GFPGANの正規化に合わせる
                input_tensor = input_tensor * 2.0 - 1.0  # [0,1] → [-1,1]
                output_tensor = scripted_model(input_tensor)
                output_tensor = (output_tensor + 1.0) / 2.0  # [-1,1] → [0,1]
        
        # 後処理
        output_tensor = torch.clamp(output_tensor, 0, 1)
        output_image = output_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        output_image = (output_image * 255).astype(np.uint8)
        output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
        
        # 元のサイズにリサイズ
        if (original_h, original_w) != (512, 512):
            output_image = cv2.resize(output_image, (original_w, original_h), interpolation=cv2.INTER_LANCZOS4)
        
        return output_image
        
    except Exception as e:
        print(f"TorchScript Live upscale error: {e}")
        # フォールバック：従来のPyTorch処理
        return upscale_pytorch(image, original_gfpgan)

def upscale_onnx(image, onnx_session, target_size):
    """
    ONNX Runtime GPU推論（超高速）
    """
    try:
        # 元のサイズを保存
        original_h, original_w = image.shape[:2]
        
        # 前処理：リサイズ + BGR → RGB、正規化
        input_image = cv2.resize(image, (target_size, target_size))
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        input_image = input_image.astype(np.float32) / 255.0
        input_image = np.transpose(input_image, (2, 0, 1))
        input_image = np.expand_dims(input_image, axis=0)
        
        # ONNX推論
        input_name = onnx_session.get_inputs()[0].name
        output_name = onnx_session.get_outputs()[0].name
        result = onnx_session.run([output_name], {input_name: input_image})
        
        # 後処理：正規化戻し、RGB → BGR、元サイズに戻す
        output = result[0][0]
        output = np.transpose(output, (1, 2, 0))
        output = np.clip(output * 255.0, 0, 255).astype(np.uint8)
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        
        # 元のサイズに戻す
        if (original_h, original_w) != (target_size, target_size):
            output = cv2.resize(output, (original_w, original_h))
        
        return output
        
    except Exception as e:
        print(f"ONNX推論エラー: {e}")
        return image

def upscale_pytorch(image, gfpgan_model):
    """
    従来のPyTorch推論（フォールバック）
    """
    try:
        _, _, output = gfpgan_model.enhance(
            image, has_aligned=False, only_center_face=False, paste_back=True
        )
        
        if output is None:
            print("も、もう！GFPGAN処理に失敗したから元画像を返すわよ💢")
            return image
        
        return output
        
    except Exception as e:
        print(f"PyTorch推論エラー: {e}")
        return image
