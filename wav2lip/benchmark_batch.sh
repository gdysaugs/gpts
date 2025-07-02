#!/bin/bash

# べ、別にベンチマークテストしてあげるわけじゃないけど...💢
echo "🎭 ツンデレWav2Lip バッチサイズベンチマーク開始💕"

BATCH_SIZES=(1 2 4 8 16)

for batch in "${BATCH_SIZES[@]}"; do
    echo ""
    echo "======================"
    echo "🚀 バッチサイズ ${batch} テスト開始"
    echo "======================"
    
    start_time=$(date +%s)
    
    docker run --gpus all --rm --privileged \
      -v /usr/lib/wsl:/usr/lib/wsl -e LD_LIBRARY_PATH=/usr/lib/wsl/lib \
      -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output \
      -v $(pwd)/checkpoints:/app/checkpoints -v $(pwd):/app/host \
      --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
      wav2lip-optimized:v2 bash -c "
      rm -f last_detected_face.pkl temp/face_detection_cache.pkl
      mkdir -p temp && 
      cp /app/host/inference_fp16_yolo_gfpgan_correct.py /app/inference_fp16_yolo_gfpgan_correct.py &&
      cp /app/host/enhance.py /app/enhance.py &&
      python /app/inference_fp16_yolo_gfpgan_correct.py \
      --checkpoint_path /app/checkpoints/wav2lip_gan.pth \
      --face /app/input/target_video.mp4 \
      --audio /app/input/reference_audio.wav \
      --outfile /app/output/result_batch_${batch}.mp4 \
      --out_height 720 \
      --batch_size ${batch} \
      --enable_gfpgan"
    
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    
    echo "✅ バッチサイズ ${batch}: ${duration}秒完了"
    echo "${batch},${duration}" >> benchmark_results.csv
done

echo ""
echo "🎉 全テスト完了！結果:"
echo "バッチサイズ,処理時間(秒)"
cat benchmark_results.csv
echo ""
echo "ふん！どれが一番速かったか確認しなさいよ💕"