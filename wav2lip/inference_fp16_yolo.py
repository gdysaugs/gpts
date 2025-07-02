print("\rloading torch       ", end="")
import torch
import torch.cuda.amp as amp  # 🚀 FP16用追加

print("\rloading numpy       ", end="")
import numpy as np

print("\rloading Image       ", end="")
from PIL import Image

print("\rloading argparse    ", end="")
import argparse

print("\rloading configparser", end="")
import configparser

print("\rloading math        ", end="")
import math

print("\rloading os          ", end="")
import os

print("\rloading subprocess  ", end="")
import subprocess

print("\rloading pickle      ", end="")
import pickle

print("\rloading cv2         ", end="")
import cv2

print("\rloading audio       ", end="")
import audio

print("\rloading YOLO11     ", end="")
from ultralytics import YOLO

print("\rloading re          ", end="")
import re

print("\rloading partial     ", end="")
from functools import partial

print("\rloading tqdm        ", end="")
from tqdm import tqdm

print("\rloading warnings    ", end="")
import warnings

warnings.filterwarnings(
    "ignore", category=UserWarning, module="torchvision.transforms.functional_tensor"
)
print("\rloading upscale     ", end="")
from enhance import upscale

print("\rloading load_sr     ", end="")
from enhance import load_sr

print("\rloading load_model  ", end="")
from easy_functions import load_model, g_colab

print("\rimports loaded!     ")

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
gpu_id = 0 if torch.cuda.is_available() else -1

# 💕 ツンデレFP16チェック
if device == 'cpu':
    print('ふん！GPUが見つからないからCPUで動かすけど...すっごく遅いわよ！')
    USE_FP16 = False
elif device == 'cuda':
    print(f'べ、別に嬉しいわけじゃないけど...CUDA GPU使ってあげるわ！ {torch.cuda.get_device_name()}')
    if torch.cuda.get_device_capability()[0] >= 7:
        print('やるじゃない！Tensor Core対応だからFP16で高速化してあげる💕')
        USE_FP16 = True
    else:
        print('古いGPUね...FP16は使えないわ')
        USE_FP16 = False
else:
    USE_FP16 = False
parser = argparse.ArgumentParser(
    description="Inference code to lip-sync videos in the wild using Wav2Lip models"
)

parser.add_argument(
    "--checkpoint_path",
    type=str,
    help="Name of saved checkpoint to load weights from",
    required=True,
)

parser.add_argument(
    "--segmentation_path",
    type=str,
    default="checkpoints/face_segmentation.pth",
    help="Name of saved checkpoint of segmentation network",
    required=False,
)

parser.add_argument(
    "--face",
    type=str,
    help="Filepath of video/image that contains faces to use",
    required=True,
)
parser.add_argument(
    "--audio",
    type=str,
    help="Filepath of video/audio file to use as raw audio source",
    required=True,
)
parser.add_argument(
    "--outfile",
    type=str,
    help="Video path to save result. See default for an e.g.",
    default="results/result_voice.mp4",
)

parser.add_argument(
    "--static",
    type=bool,
    help="If True, then use only first video frame for inference",
    default=False,
)
parser.add_argument(
    "--fps",
    type=float,
    help="Can be specified only if input is a static image (default: 25)",
    default=25.0,
    required=False,
)

parser.add_argument(
    "--pads",
    nargs="+",
    type=int,
    default=[0, 10, 0, 0],
    help="Padding (top, bottom, left, right). Please adjust to include chin at least",
)

parser.add_argument(
    "--wav2lip_batch_size", type=int, help="Batch size for Wav2Lip model(s)", default=1
)

parser.add_argument(
    "--out_height",
    default=480,
    type=int,
    help="Output video height. Best results are obtained at 480 or 720",
)

parser.add_argument(
    "--crop",
    nargs="+",
    type=int,
    default=[0, -1, 0, -1],
    help="Crop video to a smaller region (top, bottom, left, right). Applied after resize_factor and rotate arg. "
    "Useful if multiple face present. -1 implies the value will be auto-inferred based on height, width",
)

parser.add_argument(
    "--box",
    nargs="+",
    type=int,
    default=[-1, -1, -1, -1],
    help="Specify a constant bounding box for the face. Use only as a last resort if the face is not detected."
    "Also, might work only if the face is not moving around much. Syntax: (top, bottom, left, right).",
)

parser.add_argument(
    "--rotate",
    default=False,
    action="store_true",
    help="Sometimes videos taken from a phone can be flipped 90deg. If true, will flip video right by 90deg."
    "Use if you get a flipped result, despite feeding a normal looking video",
)

parser.add_argument(
    "--nosmooth",
    type=str,
    default=False,
    help="Prevent smoothing face detections over a short temporal window",
)

parser.add_argument(
    "--no_seg",
    default=False,
    action="store_true",
    help="Prevent using face segmentation",
)

parser.add_argument(
    "--no_sr", default=False, action="store_true", help="Prevent using super resolution"
)

parser.add_argument(
    "--sr_model",
    type=str,
    default="gfpgan",
    help="Name of upscaler - gfpgan or RestoreFormer",
    required=False,
)

parser.add_argument(
    "--fullres",
    default=3,
    type=int,
    help="used only to determine if full res is used so that no resizing needs to be done if so",
)

parser.add_argument(
    "--debug_mask",
    type=str,
    default=False,
    help="Makes background grayscale to see the mask better",
)

parser.add_argument(
    "--preview_settings", type=str, default=False, help="Processes only one frame"
)

parser.add_argument(
    "--mouth_tracking",
    type=str,
    default=False,
    help="Tracks the mouth in every frame for the mask",
)

parser.add_argument(
    "--enable_gfpgan",
    action="store_true",
    help="Enable GFPGAN post-processing for face enhancement",
)

parser.add_argument(
    "--gfpgan_weight",
    type=float,
    default=0.8,
    help="GFPGAN enhancement weight (0.0-1.0)",
)

parser.add_argument(
    "--mask_dilation",
    default=150,
    type=float,
    help="size of mask around mouth",
    required=False,
)

parser.add_argument(
    "--mask_feathering",
    default=151,
    type=int,
    help="amount of feathering of mask around mouth",
    required=False,
)

parser.add_argument(
    "--quality",
    type=str,
    help="Choose between Fast, Improved and Enhanced",
    default="Fast",
)

# べ、別にあなたのためにdlib predictor修正してあげるわけじゃないんだからね！
# import dlib
# predictor = dlib.shape_predictor(os.path.join("checkpoints", "predictor.pkl"))
predictor = None  # dlibの代わりにNoneで無効化

# ふん！mouth_detectorもdlibで代用してあげるわよ！
# mouth_detector = dlib.get_frontal_face_detector()
mouth_detector = None  # dlibの代わりにNoneで無効化

# creating variables to prevent failing when a face isn't detected
kernel = last_mask = x = y = w = h = None

g_colab = g_colab()

if not g_colab:
  # Load the config file
  config = configparser.ConfigParser()
  config.read('config.ini')

  # Get the value of the "preview_window" variable
  preview_window = config.get('OPTIONS', 'preview_window')

all_mouth_landmarks = []

model = yolo_detector = None
current_adaptive_params = None  # 角度適応パラメータ保存用

def do_load(checkpoint_path):
    global model, yolo_detector
    print("べ、別に急いでモデルをロードしてあげるわけじゃないけど...")
    model = load_model(checkpoint_path)
    
    # 💕 FP16変換
    if USE_FP16 and device == 'cuda':
        print("ふん！FP16で高速化してあげるわよ💕")
        model = model.half()
    
    # 🚀 YOLOv8n-Face初期化（下向き顔対応）
    print("YOLOv8n-Face初期化中...下向き顔でも完璧に検出するわよ！")
    try:
        # 複数の顔検出モデルを試行
        face_model_sources = [
            "https://github.com/akanametov/yolov8-face/releases/download/v0.0.0/yolov8n-face.pt",
            "https://github.com/deepcam-cn/yolov5-face/releases/download/v0.0.0/yolov8n-face.pt", 
            "yolov8n-face.pt"  # ローカルファイル
        ]
        
        yolo_detector = None
        for model_source in face_model_sources:
            try:
                if model_source.startswith("http"):
                    import requests
                    face_model_path = "yolov8n-face.pt"
                    if not os.path.exists(face_model_path):
                        print(f"顔専用モデルをダウンロード中: {model_source}")
                        response = requests.get(model_source, timeout=30)
                        response.raise_for_status()
                        with open(face_model_path, 'wb') as f:
                            f.write(response.content)
                        print("ダウンロード完了よ！")
                    model_source = face_model_path
                
                yolo_detector = YOLO(model_source)
                print(f"やったじゃない！YOLOv8n-Face専用モデル使用よ✨: {model_source}")
                break
                
            except Exception as e:
                print(f"モデル {model_source} の読み込み失敗: {e}")
                continue
        
        if yolo_detector is None:
            raise Exception("全ての顔専用モデルの読み込みに失敗")
        
    except Exception as e:
        print(f"顔専用モデルのダウンロードに失敗: {e}")
        print("フォールバック：通常のYOLOv8nで妥協してあげる...")
        yolo_detector = YOLO("yolov8n.pt")
    
    if device == 'cuda':
        yolo_detector.to(device)
    print("やったじゃない！顔検出準備完了よ💕")

def detect_face_angle(image):
    """顔角度検出（下向き、上向き、横向き、正面）"""
    try:
        results = yolo_detector.predict(image, verbose=False, save=False, show=False, conf=0.3)
        
        if not results or len(results[0].boxes) == 0:
            return None, "no_face"
        
        # 最も信頼度の高い顔を選択
        boxes = results[0].boxes
        best_idx = torch.argmax(boxes.conf).item()
        best_box = boxes.xyxy[best_idx].cpu().numpy()
        
        x1, y1, x2, y2 = map(int, best_box)
        face_width = x2 - x1
        face_height = y2 - y1
        
        # 顔の縦横比から角度を推定
        aspect_ratio = face_width / face_height
        
        # 顔の位置から角度を判定
        img_height, img_width = image.shape[:2]
        face_center_y = (y1 + y2) / 2
        face_relative_y = face_center_y / img_height
        
        # 角度分類
        if aspect_ratio > 1.2:  # 横に広い = 横向き
            angle_type = "side_facing"
        elif face_relative_y < 0.35:  # 上部 = 下向き
            angle_type = "downward"
        elif face_relative_y > 0.65:  # 下部 = 上向き
            angle_type = "upward"
        else:  # 中央 = 正面
            angle_type = "frontal"
        
        print(f"検出角度: {angle_type} (縦横比: {aspect_ratio:.2f}, Y位置: {face_relative_y:.2f})")
        return angle_type, face_relative_y
        
    except Exception as e:
        print(f"角度検出エラー: {e}")
        return "frontal", 0.5

def get_adaptive_mask_params(angle_type, face_relative_y):
    """角度に応じたマスクパラメータを返す"""
    if angle_type == "downward":
        # 下向き顔：鼻を含まない狭いマスク
        return {
            'mask_dilation': 60,   # 非常に小さいマスク
            'mask_feathering': 80,
            'mouth_crop_factor': 0.6,  # 顔の下部60%のみ
            'description': "下向き顔用狭マスク💢"
        }
    elif angle_type == "upward":
        # 上向き顔用
        return {
            'mask_dilation': 100,
            'mask_feathering': 120,
            'mouth_crop_factor': 0.8,
            'description': "上向き顔用設定"
        }
    elif angle_type == "side_facing":
        # 横向き顔用
        return {
            'mask_dilation': 90,
            'mask_feathering': 110,
            'mouth_crop_factor': 0.7,
            'description': "横向き顔用設定"
        }
    else:  # frontal
        # 正面顔用（標準設定）
        return {
            'mask_dilation': 120,
            'mask_feathering': 130,
            'mouth_crop_factor': 1.0,
            'description': "正面顔用標準設定"
        }

def face_rect(images):
    """YOLOv8-Face専用モデルを使った正確な顔検出（角度適応版）"""
    prev_ret = None
    
    # 最初のフレームで角度検出
    if len(images) > 0:
        angle_type, face_relative_y = detect_face_angle(images[0])
        adaptive_params = get_adaptive_mask_params(angle_type, face_relative_y)
        print(f"適用設定: {adaptive_params['description']}")
        
        # グローバル変数として設定を保存（後でマスク処理で使用）
        global current_adaptive_params
        current_adaptive_params = adaptive_params
    
    for image in images:
        try:
            # YOLOv8-Face専用モデルで顔検出
            results = yolo_detector.predict(image, verbose=False, save=False, show=False, conf=0.3)
            
            best_face = None
            best_conf = 0
            largest_area = 0
            
            for result in results:
                if result.boxes is not None and len(result.boxes) > 0:
                    for box in result.boxes:
                        # 顔専用モデルなので全検出が顔
                        conf = float(box.conf.cpu().numpy())
                        if conf > 0.3:  # 顔検出の信頼度（低めに設定）
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            
                            # 面積で最大の顔を選択（メイン被写体）
                            area = (x2 - x1) * (y2 - y1)
                            
                            if area > largest_area and area > 100:  # 最小面積チェック
                                largest_area = area
                                best_conf = conf
                                best_face = (int(x1), int(y1), int(x2), int(y2))
            
            if best_face:
                prev_ret = best_face
            
        except Exception as e:
            print(f"顔検出エラー: {e}")
            # エラー時は前回の結果を使用
            pass
        
        yield prev_ret

def create_tracked_mask(img, original_img):
    global kernel, last_mask, x, y, w, h  # Add last_mask to global variables

    # Convert color space from BGR to RGB if necessary
    cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
    cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB, original_img)

    # Detect face
    faces = mouth_detector(img)
    if len(faces) == 0:
        if last_mask is not None:
            last_mask = cv2.resize(last_mask, (img.shape[1], img.shape[0]))
            mask = last_mask  # use the last successful mask
        else:
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
            return img, None
    else:
        face = faces[0]
        shape = predictor(img, face)

        # Get points for mouth
        mouth_points = np.array(
            [[shape.part(i).x, shape.part(i).y] for i in range(48, 68)]
        )

        # Calculate bounding box dimensions
        x, y, w, h = cv2.boundingRect(mouth_points)

        # Set kernel size as a fraction of bounding box size
        kernel_size = int(max(w, h) * args.mask_dilation)
        # if kernel_size % 2 == 0:  # Ensure kernel size is odd
        # kernel_size += 1

        # Create kernel
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        # Create binary mask for mouth
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(mask, mouth_points, 255)

        last_mask = mask  # Update last_mask with the new mask

    # Dilate the mask
    dilated_mask = cv2.dilate(mask, kernel)

    # Calculate distance transform of dilated mask
    dist_transform = cv2.distanceTransform(dilated_mask, cv2.DIST_L2, 5)

    # Normalize distance transform
    cv2.normalize(dist_transform, dist_transform, 0, 255, cv2.NORM_MINMAX)

    # Convert normalized distance transform to binary mask and convert it to uint8
    _, masked_diff = cv2.threshold(dist_transform, 50, 255, cv2.THRESH_BINARY)
    masked_diff = masked_diff.astype(np.uint8)

    # make sure blur is an odd number
    blur = args.mask_feathering
    if blur % 2 == 0:
        blur += 1
    # Set blur size as a fraction of bounding box size
    blur = int(max(w, h) * blur)  # 10% of bounding box size
    if blur % 2 == 0:  # Ensure blur size is odd
        blur += 1
    masked_diff = cv2.GaussianBlur(masked_diff, (blur, blur), 0)

    # Convert numpy arrays to PIL Images
    input1 = Image.fromarray(img)
    input2 = Image.fromarray(original_img)

    # Convert mask to single channel where pixel values are from the alpha channel of the current mask
    mask = Image.fromarray(masked_diff)

    # Ensure images are the same size
    assert input1.size == input2.size == mask.size

    # Paste input1 onto input2 using the mask
    input2.paste(input1, (0, 0), mask)

    # Convert the final PIL Image back to a numpy array
    input2 = np.array(input2)

    # input2 = cv2.cvtColor(input2, cv2.COLOR_BGR2RGB)
    cv2.cvtColor(input2, cv2.COLOR_BGR2RGB, input2)

    return input2, mask


def create_mask(img, original_img):
    global kernel, last_mask, x, y, w, h # Add last_mask to global variables

    # Convert color space from BGR to RGB if necessary
    cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
    cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB, original_img)

    if last_mask is not None:
        last_mask = np.array(last_mask)  # Convert PIL Image to numpy array
        last_mask = cv2.resize(last_mask, (img.shape[1], img.shape[0]))
        mask = last_mask  # use the last successful mask
        mask = Image.fromarray(mask)

    else:
        # Detect face
        faces = mouth_detector(img)
        if len(faces) == 0:
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
            return img, None
        else:
            face = faces[0]
            shape = predictor(img, face)

            # Get points for mouth
            mouth_points = np.array(
                [[shape.part(i).x, shape.part(i).y] for i in range(48, 68)]
            )

            # Calculate bounding box dimensions
            x, y, w, h = cv2.boundingRect(mouth_points)

            # Set kernel size as a fraction of bounding box size
            kernel_size = int(max(w, h) * args.mask_dilation)
            # if kernel_size % 2 == 0:  # Ensure kernel size is odd
            # kernel_size += 1

            # Create kernel
            kernel = np.ones((kernel_size, kernel_size), np.uint8)

            # Create binary mask for mouth
            mask = np.zeros(img.shape[:2], dtype=np.uint8)
            cv2.fillConvexPoly(mask, mouth_points, 255)

            # Dilate the mask
            dilated_mask = cv2.dilate(mask, kernel)

            # Calculate distance transform of dilated mask
            dist_transform = cv2.distanceTransform(dilated_mask, cv2.DIST_L2, 5)

            # Normalize distance transform
            cv2.normalize(dist_transform, dist_transform, 0, 255, cv2.NORM_MINMAX)

            # Convert normalized distance transform to binary mask and convert it to uint8
            _, masked_diff = cv2.threshold(dist_transform, 50, 255, cv2.THRESH_BINARY)
            masked_diff = masked_diff.astype(np.uint8)

            if not args.mask_feathering == 0:
                blur = args.mask_feathering
                # Set blur size as a fraction of bounding box size
                blur = int(max(w, h) * blur)  # 10% of bounding box size
                if blur % 2 == 0:  # Ensure blur size is odd
                    blur += 1
                masked_diff = cv2.GaussianBlur(masked_diff, (blur, blur), 0)

            # Convert mask to single channel where pixel values are from the alpha channel of the current mask
            mask = Image.fromarray(masked_diff)

            last_mask = mask  # Update last_mask with the final mask after dilation and feathering

    # Convert numpy arrays to PIL Images
    input1 = Image.fromarray(img)
    input2 = Image.fromarray(original_img)

    # Resize mask to match image size
    # mask = Image.fromarray(mask)
    mask = mask.resize(input1.size)

    # Ensure images are the same size
    assert input1.size == input2.size == mask.size

    # Paste input1 onto input2 using the mask
    input2.paste(input1, (0, 0), mask)

    # Convert the final PIL Image back to a numpy array
    input2 = np.array(input2)

    # input2 = cv2.cvtColor(input2, cv2.COLOR_BGR2RGB)
    cv2.cvtColor(input2, cv2.COLOR_BGR2RGB, input2)

    return input2, mask


def get_smoothened_boxes(boxes, T):
    for i in range(len(boxes)):
        if i + T > len(boxes):
            window = boxes[len(boxes) - T :]
        else:
            window = boxes[i : i + T]
        boxes[i] = np.mean(window, axis=0)
    return boxes
            
def face_detect(images, results_file="last_detected_face.pkl"):
    # If results file exists, load it and return
    if os.path.exists(results_file):
        print("Using face detection data from last input")
        with open(results_file, "rb") as f:
            return pickle.load(f)

    results = []
    pady1, pady2, padx1, padx2 = args.pads
    
    tqdm_partial = partial(tqdm, position=0, leave=True)
    for image, (rect) in tqdm_partial(
        zip(images, face_rect(images)),
        total=len(images),
        desc="detecting face in every frame",
        ncols=100,
    ):
        if rect is None:
            cv2.imwrite(
                "temp/faulty_frame.jpg", image
            )  # check this frame where the face was not detected.
            raise ValueError(
                "Face not detected! Ensure the video contains a face in all the frames."
            )

        y1 = max(0, rect[1] - pady1)
        y2 = min(image.shape[0], rect[3] + pady2)
        x1 = max(0, rect[0] - padx1)
        x2 = min(image.shape[1], rect[2] + padx2)

        results.append([x1, y1, x2, y2])


    boxes = np.array(results)
    if str(args.nosmooth) == "False":
        boxes = get_smoothened_boxes(boxes, T=5)
    results = [
        [image[y1:y2, x1:x2], (y1, y2, x1, x2)]
        for image, (x1, y1, x2, y2) in zip(images, boxes)
    ]

    # Save results to file
    with open(results_file, "wb") as f:
        pickle.dump(results, f)

    return results


def datagen(frames, mels):
    img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []
    print("\r" + " " * 100, end="\r")
    if args.box[0] == -1:
        if not args.static:
            face_det_results = face_detect(frames)  # BGR2RGB for CNN face detection
        else:
            face_det_results = face_detect([frames[0]])
    else:
        print("Using the specified bounding box instead of face detection...")
        y1, y2, x1, x2 = args.box
        face_det_results = [[f[y1:y2, x1:x2], (y1, y2, x1, x2)] for f in frames]

    for i, m in enumerate(mels):
        idx = 0 if args.static else i % len(frames)
        frame_to_save = frames[idx].copy()
        face, coords = face_det_results[idx].copy()

        face = cv2.resize(face, (args.img_size, args.img_size))

        img_batch.append(face)
        mel_batch.append(m)
        frame_batch.append(frame_to_save)
        coords_batch.append(coords)

        if len(img_batch) >= args.wav2lip_batch_size:
            img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

            img_masked = img_batch.copy()
            img_masked[:, args.img_size // 2 :] = 0

            img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.0
            mel_batch = np.reshape(
                mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1]
            )

            yield img_batch, mel_batch, frame_batch, coords_batch
            img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

    if len(img_batch) > 0:
        img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

        img_masked = img_batch.copy()
        img_masked[:, args.img_size // 2 :] = 0

        img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.0
        mel_batch = np.reshape(
            mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1]
        )

        yield img_batch, mel_batch, frame_batch, coords_batch


mel_step_size = 16

def _load(checkpoint_path):
    if device != "cpu":
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(
            checkpoint_path, map_location=lambda storage, loc: storage
        )
    return checkpoint


def main():
    args.img_size = 96
    frame_number = 11

    if os.path.isfile(args.face) and args.face.split(".")[1] in ["jpg", "png", "jpeg"]:
        args.static = True

    if not os.path.isfile(args.face):
        raise ValueError("--face argument must be a valid path to video/image file")

    elif args.face.split(".")[1] in ["jpg", "png", "jpeg"]:
        full_frames = [cv2.imread(args.face)]
        fps = args.fps

    else:
        if args.fullres != 1:
            print("Resizing video...")
        video_stream = cv2.VideoCapture(args.face)
        fps = video_stream.get(cv2.CAP_PROP_FPS)

        full_frames = []
        while 1:
            still_reading, frame = video_stream.read()
            if not still_reading:
                video_stream.release()
                break

            if args.fullres != 1:
                aspect_ratio = frame.shape[1] / frame.shape[0]
                frame = cv2.resize(
                    frame, (int(args.out_height * aspect_ratio), args.out_height)
                )

            if args.rotate:
                frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)

            y1, y2, x1, x2 = args.crop
            if x2 == -1:
                x2 = frame.shape[1]
            if y2 == -1:
                y2 = frame.shape[0]

            frame = frame[y1:y2, x1:x2]

            full_frames.append(frame)

    if not args.audio.endswith(".wav"):
        print("Converting audio to .wav")
        subprocess.check_call(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                args.audio,
                "temp/temp.wav",
            ]
        )
        args.audio = "temp/temp.wav"

    print("analysing audio...")
    wav = audio.load_wav(args.audio, 16000)
    mel = audio.melspectrogram(wav)

    if np.isnan(mel.reshape(-1)).sum() > 0:
        raise ValueError(
            "Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again"
        )

    mel_chunks = []

    mel_idx_multiplier = 80.0 / fps
    i = 0
    while 1:
        start_idx = int(i * mel_idx_multiplier)
        if start_idx + mel_step_size > len(mel[0]):
            mel_chunks.append(mel[:, len(mel[0]) - mel_step_size :])
            break
        mel_chunks.append(mel[:, start_idx : start_idx + mel_step_size])
        i += 1

    full_frames = full_frames[: len(mel_chunks)]
    if str(args.preview_settings) == "True":
        full_frames = [full_frames[0]]
        mel_chunks = [mel_chunks[0]]
    print(str(len(full_frames)) + " frames to process")
    batch_size = args.wav2lip_batch_size
    if str(args.preview_settings) == "True":
        gen = datagen(full_frames, mel_chunks)
    else:
        gen = datagen(full_frames.copy(), mel_chunks)

    for i, (img_batch, mel_batch, frames, coords) in enumerate(
        tqdm(
            gen,
            total=int(np.ceil(float(len(mel_chunks)) / batch_size)),
            desc="Processing Wav2Lip",
            ncols=100,
        )
    ):
        if i == 0:
            if not args.quality == "Fast":
                print(
                    f"mask size: {args.mask_dilation}, feathering: {args.mask_feathering}"
                )
                if not args.quality == "Improved":
                    print("Loading", args.sr_model)
                    run_params = load_sr()

            print("Starting...")
            frame_h, frame_w = full_frames[0].shape[:-1]
            # tempディレクトリ作成
            os.makedirs("temp", exist_ok=True)
            
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter("temp/result.mp4", fourcc, fps, (frame_w, frame_h))

        img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
        mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)

        # 💕 FP16最適化（モデル推論のみ、座標とリサイズは必ずFP32）
        with torch.no_grad():
            if USE_FP16 and device == 'cuda':
                # 入力データのみFP16変換（元のテンソルは保持）
                img_batch_fp16 = img_batch.half()
                mel_batch_fp16 = mel_batch.half()
                pred = model(mel_batch_fp16, img_batch_fp16)
            else:
                pred = model(mel_batch, img_batch)

        # 出力は必ずFP32に戻して精度保持
        pred = pred.cpu().float().numpy().transpose(0, 2, 3, 1) * 255.0

        for idx, (p, f, c) in enumerate(zip(pred, frames, coords)):
            # cv2.imwrite('temp/f.jpg', f)

            # 座標を確実にint型で保持（FP16精度問題対策）
            y1, y2, x1, x2 = int(c[0]), int(c[1]), int(c[2]), int(c[3])
            
            # 座標の妥当性チェック
            frame_h, frame_w = f.shape[:2]
            
            # 座標をフレーム範囲内に修正
            y1 = max(0, min(y1, frame_h - 1))
            y2 = max(y1 + 1, min(y2, frame_h))
            x1 = max(0, min(x1, frame_w - 1))
            x2 = max(x1 + 1, min(x2, frame_w))
            
            # デバッグ用（最初の数フレームのみ）
            if idx < 3:
                print(f"Frame {idx}: y1={y1}, y2={y2}, x1={x1}, x2={x2}, frame shape={f.shape}")

            if (
                str(args.debug_mask) == "True"
            ):  # makes the background black & white so you can see the mask better
                f = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
                f = cv2.cvtColor(f, cv2.COLOR_GRAY2BGR)

            # サイズを確実に正の整数で計算
            width = max(1, x2 - x1)
            height = max(1, y2 - y1)
            
            # 予測結果のサイズチェック
            if width < 10 or height < 10:
                print(f"Warning: Very small face region detected: {width}x{height}")
                continue  # スキップ
                
            try:
                p = cv2.resize(p.astype(np.uint8), (width, height))
                cf = f[y1:y2, x1:x2]
            except Exception as e:
                print(f"Resize error at frame {idx}: {e}")
                print(f"p shape: {p.shape}, target size: ({width}, {height})")
                continue

            if args.quality == "Enhanced":
                try:
                    p = upscale(p, run_params)
                except Exception as e:
                    print(f"Upscale error at frame {idx}: {e}")

            if args.quality in ["Enhanced", "Improved"]:
                try:
                    if str(args.mouth_tracking) == "True":
                        p, last_mask = create_tracked_mask(p, cf)
                    else:
                        p, last_mask = create_mask(p, cf)
                except Exception as e:
                    print(f"Mask creation error at frame {idx}: {e}")

            # 最終的なサイズチェック
            if p.shape[:2] != (height, width):
                print(f"Size mismatch at frame {idx}: p.shape={p.shape}, expected=({height}, {width})")
                try:
                    p = cv2.resize(p, (width, height))
                except:
                    continue
                    
            # 安全に代入
            try:
                f[y1:y2, x1:x2] = p
            except Exception as e:
                print(f"Assignment error at frame {idx}: {e}")
                print(f"f slice shape: {f[y1:y2, x1:x2].shape}, p shape: {p.shape}")
                continue

            if not g_colab:
                # Display the frame (DISABLED for headless operation)
                # if preview_window == "Face":
                #     cv2.imshow("face preview - press Q to abort", p)
                # elif preview_window == "Full":
                #     cv2.imshow("full preview - press Q to abort", f)
                # elif preview_window == "Both":
                #     cv2.imshow("face preview - press Q to abort", p)
                #     cv2.imshow("full preview - press Q to abort", f)

                # key = cv2.waitKey(1) & 0xFF
                # if key == ord('q'):
                #     exit()  # Exit the loop when 'Q' is pressed
                pass  # べ、別にGUI機能を無効化してあげるわけじゃないんだからね！

            if str(args.preview_settings) == "True":
                cv2.imwrite("temp/preview.jpg", f)
                if not g_colab:
                    cv2.imshow("preview - press Q to close", f)
                    if cv2.waitKey(-1) & 0xFF == ord('q'):
                        exit()  # Exit the loop when 'Q' is pressed

            else:
                out.write(f)

    # Close the window(s) when done
    cv2.destroyAllWindows()

    out.release()

    if str(args.preview_settings) == "False":
        print("converting to final video")

        subprocess.check_call([
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            "temp/result.mp4",
            "-i",
            args.audio,
            "-c:v",
            "libx264",
            args.outfile
        ])
        
        # GFPGAN後処理（オプション）
        if hasattr(args, 'enable_gfpgan') and args.enable_gfpgan:
            apply_gfpgan_postprocess(args.outfile, getattr(args, 'gfpgan_weight', 0.8))
        
    print("完了よ！感謝しなさいよね💕")

if __name__ == "__main__":
    args = parser.parse_args()
    do_load(args.checkpoint_path)
    main()
