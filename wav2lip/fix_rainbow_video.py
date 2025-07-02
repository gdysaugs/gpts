#!/usr/bin/env python3
"""
🎭 ツンデレ虹色動画修正スクリプト
べ、別にあなたのために色空間バグを直してあげるわけじゃないんだからね！💢

YUV/RGB色空間変換問題とONNX処理による色空間破損を修正
"""

import cv2
import numpy as np
import subprocess
import os
import argparse

def fix_colorspace_opencv(input_video, output_video):
    """OpenCVを使用した色空間修正"""
    print("べ、別にOpenCVで色空間を修正してあげるわけじゃないけど...💕")
    
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print(f"❌ 動画が開けないじゃない: {input_video}")
        return False
    
    # 動画情報取得
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"📊 動画情報: {width}x{height}, {fps}fps, {frame_count}フレーム")
    
    # 出力設定
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    print("色空間修正中...")
    
    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        
        # 色空間修正のトリック
        # 1. YUVとして読み込まれた可能性があるフレームをRGBに変換
        if frame.shape[2] == 3:  # BGR画像の場合
            # 異常な色合いの場合は色チャンネルを交換
            b, g, r = cv2.split(frame)
            
            # 虹色問題：通常はBGR→RGBの順序問題
            # パターン1: BGR → RGB
            frame_rgb = cv2.merge([r, g, b])
            
            # パターン2: 色チャンネル入れ替え（虹色修正）
            # 青と赤が入れ替わってる場合が多い
            frame_fixed = cv2.merge([b, g, r])  # 元に戻す
            
            # より高度な修正: YUV → RGB変換エラーの修正
            try:
                # フレームの値範囲を確認
                frame_mean = np.mean(frame)
                if frame_mean > 128:  # 明るすぎる場合
                    frame = np.clip(frame * 0.8, 0, 255).astype(np.uint8)
                
                # 色の彩度を正常化
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 0.7, 0, 255)  # 彩度を下げる
                frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                
            except:
                pass  # エラー時は元フレームを使用
        
        out.write(frame)
    
    cap.release()
    out.release()
    
    print(f"✅ OpenCV修正完了: {output_video}")
    return True

def fix_colorspace_ffmpeg(input_video, output_video):
    """FFmpegを使用した色空間修正"""
    print("べ、別にFFmpegで色空間を修正してあげるわけじゃないけど...💢")
    
    # 複数の修正方法を試行
    fix_commands = [
        # パターン1: 基本的な色空間変換
        [
            "ffmpeg", "-y", "-i", input_video,
            "-vf", "colorspace=bt709:iall=bt601-6-625:fast=1",
            "-c:v", "libx264", "-preset", "fast",
            "-c:a", "copy", output_video
        ],
        
        # パターン2: YUV420p強制 + 色補正
        [
            "ffmpeg", "-y", "-i", input_video,
            "-vf", "format=yuv420p,eq=contrast=0.9:brightness=0.05:saturation=0.8",
            "-c:v", "libx264", "-preset", "fast",
            "-c:a", "copy", output_video
        ],
        
        # パターン3: RGB24強制 + 再エンコード
        [
            "ffmpeg", "-y", "-i", input_video,
            "-vf", "format=rgb24,format=yuv420p",
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-c:a", "copy", output_video
        ]
    ]
    
    for i, cmd in enumerate(fix_commands, 1):
        print(f"🔧 修正パターン {i} 試行中...")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ FFmpeg修正成功 (パターン {i}): {output_video}")
                return True
            else:
                print(f"⚠️ パターン {i} 失敗: {result.stderr}")
        except Exception as e:
            print(f"❌ パターン {i} エラー: {e}")
    
    print("❌ FFmpeg修正失敗")
    return False

def analyze_video_colorspace(video_path):
    """動画の色空間情報を分析"""
    print(f"📊 動画分析中: {video_path}")
    
    cmd = [
        "ffprobe", "-v", "quiet", "-show_entries", 
        "stream=color_space,color_primaries,color_transfer,pix_fmt",
        "-of", "csv=p=0", video_path
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"🔍 色空間情報: {result.stdout.strip()}")
        else:
            print("⚠️ 色空間情報取得失敗")
    except:
        print("❌ ffprobe実行エラー")
    
    # OpenCVでも確認
    cap = cv2.VideoCapture(video_path)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print(f"📊 フレーム形状: {frame.shape}")
            print(f"📊 フレーム値範囲: {frame.min()}-{frame.max()}")
            print(f"📊 平均値: R={frame[:,:,2].mean():.1f}, G={frame[:,:,1].mean():.1f}, B={frame[:,:,0].mean():.1f}")
        cap.release()

def main():
    parser = argparse.ArgumentParser(description="🎭 ツンデレ虹色動画修正")
    parser.add_argument("--input", type=str, required=True, help="入力動画")
    parser.add_argument("--output", type=str, help="出力動画")
    parser.add_argument("--method", type=str, default="ffmpeg", 
                       choices=["ffmpeg", "opencv", "both"], help="修正方法")
    parser.add_argument("--analyze", action="store_true", help="色空間分析のみ")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"❌ 入力ファイルが見つからない: {args.input}")
        return
    
    if args.analyze:
        analyze_video_colorspace(args.input)
        return
    
    if not args.output:
        base_name = os.path.splitext(args.input)[0]
        args.output = f"{base_name}_fixed.mp4"
    
    print("ふん！虹色動画修正開始よ...💕")
    
    # 分析
    analyze_video_colorspace(args.input)
    
    success = False
    
    if args.method in ["ffmpeg", "both"]:
        success = fix_colorspace_ffmpeg(args.input, args.output)
    
    if not success and args.method in ["opencv", "both"]:
        success = fix_colorspace_opencv(args.input, args.output)
    
    if success:
        print(f"✨ 修正完了！感謝しなさいよね💕")
        print(f"📁 出力: {args.output}")
        
        # 修正後の確認
        print("\n修正後の分析:")
        analyze_video_colorspace(args.output)
    else:
        print("も、もう！修正に失敗したわよ💢")

if __name__ == "__main__":
    main()