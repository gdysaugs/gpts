import warnings
from gfpgan import GFPGANer

warnings.filterwarnings("ignore")


def load_sr():
    run_params = GFPGANer(
        model_path="checkpoints/GFPGANv1.4.pth",
        upscale=1,
        arch="clean",
        channel_multiplier=2,
        bg_upsampler=None,
    )
    return run_params


def upscale(image, properties):
    try:
        # GFPGAN設定を修正：has_aligned=False, only_center_face=False に変更
        _, _, output = properties.enhance(
            image, has_aligned=False, only_center_face=False, paste_back=True
        )
        
        # outputがNoneの場合は元画像を返す
        if output is None:
            print("も、もう！GFPGAN処理に失敗したから元画像を返すわよ💢")
            return image
        
        return output
        
    except Exception as e:
        print(f"GFPGAN upscale error: {e}")
        return image  # エラー時は元画像を返す
