#!/usr/bin/env python3
"""
並列処理パフォーマンステスト
"""
import time
import subprocess
import argparse

def test_processing(script_path, args_list):
    """処理時間計測"""
    start_time = time.time()
    
    cmd = ["python", script_path] + args_list
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    return processing_time, result.returncode == 0

def main():
    parser = argparse.ArgumentParser(description="並列処理パフォーマンステスト")
    parser.add_argument("--checkpoint_path", default="checkpoints/wav2lip_gan.pth")
    parser.add_argument("--face", default="input/target_video.mp4") 
    parser.add_argument("--audio", default="input/reference_audio.wav")
    parser.add_argument("--out_height", type=int, default=720)
    args = parser.parse_args()
    
    print("🎭 並列処理パフォーマンステスト開始")
    
    # テスト設定
    tests = [
        {
            "name": "スタンドアロン版（順次処理）",
            "script": "inference_fp16_yolo_codeformer_standalone.py",
            "args": [
                "--checkpoint_path", args.checkpoint_path,
                "--face", args.face,
                "--audio", args.audio,
                "--outfile", "output/test_standalone.mp4",
                "--out_height", str(args.out_height),
                "--fidelity_weight", "0.7"
            ]
        },
        {
            "name": "並列処理版（マルチプロセス）",
            "script": "inference_fp16_yolo_codeformer_parallel.py",
            "args": [
                "--checkpoint_path", args.checkpoint_path,
                "--face", args.face,
                "--audio", args.audio,
                "--outfile", "output/test_parallel.mp4",
                "--out_height", str(args.out_height),
                "--fidelity_weight", "0.7",
                "--num_workers", "4"
            ]
        },
        {
            "name": "並列処理版（GPUバッチ）",
            "script": "inference_fp16_yolo_codeformer_parallel.py",
            "args": [
                "--checkpoint_path", args.checkpoint_path,
                "--face", args.face,
                "--audio", args.audio,
                "--outfile", "output/test_gpu_batch.mp4",
                "--out_height", str(args.out_height),
                "--fidelity_weight", "0.7",
                "--use_gpu_batch",
                "--batch_size", "8"
            ]
        }
    ]
    
    results = []
    
    for test in tests:
        print(f"\n⚡ テスト: {test['name']}")
        print("処理中...")
        
        time_taken, success = test_processing(test['script'], test['args'])
        
        if success:
            print(f"✅ 成功: {time_taken:.1f}秒")
            results.append((test['name'], time_taken))
        else:
            print(f"❌ 失敗")
    
    # 結果まとめ
    print("\n📊 パフォーマンス比較結果:")
    print("-" * 50)
    for name, time_taken in sorted(results, key=lambda x: x[1]):
        print(f"{name}: {time_taken:.1f}秒")
    
    if results:
        fastest = min(results, key=lambda x: x[1])
        slowest = max(results, key=lambda x: x[1])
        speedup = slowest[1] / fastest[1]
        print(f"\n⚡ 最速: {fastest[0]} ({fastest[1]:.1f}秒)")
        print(f"🐌 最遅: {slowest[0]} ({slowest[1]:.1f}秒)")
        print(f"🚀 高速化率: {speedup:.1f}倍")

if __name__ == "__main__":
    main()