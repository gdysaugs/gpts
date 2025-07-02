#!/usr/bin/env python3
"""
🎭 YOLOv8n-faceモデル検証スクリプト
べ、別に本当に顔専用モデルが使われてるか確認してあげるわけじゃないけど...💢
"""

from ultralytics import YOLO
import torch

def check_yolo_model():
    try:
        print("YOLOv8n-faceモデル読み込み中...")
        model = YOLO('yolov8n-face.pt')
        
        print("✅ モデル詳細:")
        print(f"デバイス: {next(model.model.parameters()).device}")
        print(f"パラメータ数: {sum(p.numel() for p in model.model.parameters()):,}")
        
        # モデルがface専用かどうかの推定
        total_params = sum(p.numel() for p in model.model.parameters())
        
        if 3_000_000 <= total_params <= 4_000_000:
            print("✅ YOLOv8n-faceサイズ範囲内（顔専用モデルの可能性高）")
        else:
            print(f"⚠️  パラメータ数が想定外: {total_params:,}")
        
        print("やったじゃない！YOLOv8n-face確認完了よ💕")
        
    except Exception as e:
        print(f"も、もう！エラー発生: {e}")

if __name__ == "__main__":
    check_yolo_model()