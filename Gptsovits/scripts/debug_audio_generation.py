#!/usr/bin/env python3
"""
音声生成のデバッグスクリプト
分割処理と音声データの詳細分析
"""

import os
import sys
import numpy as np
import soundfile as sf
import matplotlib
matplotlib.use('Agg')  # GUI不要
import matplotlib.pyplot as plt

def analyze_audio_file(file_path):
    """音声ファイルの詳細分析"""
    print(f"\n=== 音声ファイル分析: {file_path} ===")
    
    if not os.path.exists(file_path):
        print(f"ファイルが見つかりません: {file_path}")
        return False
        
    try:
        # 音声データ読み込み
        data, sr = sf.read(file_path)
        
        print(f"サンプリングレート: {sr} Hz")
        print(f"データ長: {len(data)} samples")
        print(f"音声時間: {len(data)/sr:.3f} 秒")
        print(f"データ型: {data.dtype}")
        print(f"データ形状: {data.shape}")
        
        # 統計情報
        print(f"\n--- 統計情報 ---")
        print(f"最大値: {np.max(data):.6f}")
        print(f"最小値: {np.min(data):.6f}")
        print(f"平均値: {np.mean(data):.6f}")
        print(f"標準偏差: {np.std(data):.6f}")
        print(f"RMS: {np.sqrt(np.mean(data**2)):.6f}")
        
        # 無音部分の検出
        silence_threshold = 0.001
        non_silent_samples = np.abs(data) > silence_threshold
        non_silent_count = np.sum(non_silent_samples)
        silent_count = len(data) - non_silent_count
        
        print(f"\n--- 無音分析 (閾値: {silence_threshold}) ---")
        print(f"非無音サンプル数: {non_silent_count}")
        print(f"無音サンプル数: {silent_count}")
        print(f"非無音率: {non_silent_count/len(data)*100:.2f}%")
        
        # 連続無音区間の検出
        if len(data) > 0:
            silent_regions = []
            in_silence = True
            silence_start = 0
            
            for i, sample in enumerate(np.abs(data) <= silence_threshold):
                if sample and not in_silence:  # 無音開始
                    in_silence = True
                    silence_start = i
                elif not sample and in_silence:  # 無音終了
                    in_silence = False
                    silence_duration = (i - silence_start) / sr
                    if silence_duration > 0.01:  # 10ms以上の無音
                        silent_regions.append((silence_start/sr, i/sr, silence_duration))
            
            print(f"\n--- 長い無音区間 (>10ms) ---")
            if silent_regions:
                for start, end, duration in silent_regions[:5]:  # 最初の5個
                    print(f"  {start:.3f}s - {end:.3f}s (長さ: {duration:.3f}s)")
                if len(silent_regions) > 5:
                    print(f"  ... 他 {len(silent_regions)-5} 個")
            else:
                print("  長い無音区間なし")
        
        # エネルギー分布
        if len(data) > 100:
            # 100サンプルごとのRMS
            window_size = 100
            num_windows = len(data) // window_size
            rms_values = []
            
            for i in range(num_windows):
                start = i * window_size
                end = start + window_size
                window_rms = np.sqrt(np.mean(data[start:end]**2))
                rms_values.append(window_rms)
            
            print(f"\n--- エネルギー分布 ({window_size}サンプル窓) ---")
            print(f"最大RMS: {np.max(rms_values):.6f}")
            print(f"平均RMS: {np.mean(rms_values):.6f}")
            print(f"RMS標準偏差: {np.std(rms_values):.6f}")
            
            # 低エネルギー窓の割合
            low_energy_threshold = np.max(rms_values) * 0.1
            low_energy_count = np.sum(np.array(rms_values) < low_energy_threshold)
            print(f"低エネルギー窓 (<10%): {low_energy_count}/{num_windows} ({low_energy_count/num_windows*100:.1f}%)")
        
        return True
        
    except Exception as e:
        print(f"分析エラー: {e}")
        return False

def test_different_cuts():
    """異なる分割方法での音声生成テスト"""
    print("\n=== 分割方法比較テスト ===")
    
    # テストケース
    cut_methods = [
        ("No slice", "不切"),
        ("Cut by punctuation", "按标点符号切"),
        ("Combine 50 characters", "凑50字一切"),
        ("Combine four sentences", "凑四句一切")
    ]
    
    ref_audio = "/app/input/reference_5sec.wav"
    ref_text = "おはようございます"
    target_text = "こんにちは、これはテスト音声です。"
    
    for en_name, zh_name in cut_methods:
        print(f"\n--- {en_name} ({zh_name}) ---")
        output_file = f"/app/output/test_{en_name.replace(' ', '_').lower()}.wav"
        
        # Docker実行コマンド作成（簡略版 - 実際はsubprocessで実行）
        cmd = f"""
docker run --gpus all --rm \\
  -v $(pwd)/input:/app/input \\
  -v $(pwd)/output:/app/output \\
  -v $(pwd)/logs:/app/logs \\
  -v $(pwd)/scripts:/app/scripts \\
  gpt-sovits:v4 python -c "
import sys
sys.path.insert(0, '/app')
sys.path.insert(0, '/app/GPT_SoVITS')
from GPT_SoVITS.inference_webui import get_tts_wav
import soundfile as sf

result = get_tts_wav(
    ref_wav_path='{ref_audio}',
    prompt_text='{ref_text}',
    prompt_language='Japanese',
    text='{target_text}',
    text_language='Japanese',
    how_to_cut='{zh_name}',
    top_k=15,
    top_p=0.8,
    temperature=0.7,
    ref_free=False
)

for item in result:
    if isinstance(item, tuple) and len(item) >= 2:
        sr, audio = item[0], item[1]
        sf.write('{output_file}', audio, sr)
        print(f'生成完了: {len(audio)} samples, {len(audio)/sr:.3f}s')
        break
"
"""
        print(f"実行コマンド例:")
        print(cmd.strip())

if __name__ == "__main__":
    # 既存の音声ファイルを分析
    audio_files = [
        "/home/adama/.claude/projects/Gptsovits/output/cloned_voice.wav",
        "/home/adama/.claude/projects/Gptsovits/output/long_cloned_voice.wav"
    ]
    
    for audio_file in audio_files:
        analyze_audio_file(audio_file)
    
    # 分割方法のテスト方法を表示
    test_different_cuts()