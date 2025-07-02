"""
Wav2Lip + YOLO11 統合処理エンジン
ツンデレ口調での高品質口パク動画生成システム
"""

import cv2
import numpy as np
import torch
import librosa
from typing import List, Dict, Tuple, Optional
import logging
from pathlib import Path
import time

# Easy-Wav2Lipモジュール
from inference import load_model, main as wav2lip_inference
from audio import load_wav
import face_detection

# 自作モジュール
from yolo_detector import TsundereYOLODetector
from face_aligner import TsundereFaceAligner, calculate_face_quality_score

# ツンデレログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TsundereWav2LipYOLOEngine:
    """ツンデレなWav2Lip+YOLO統合エンジン"""
    
    def __init__(self,
                 wav2lip_model_path: str = "/app/models/wav2lip/wav2lip_gan.pth",
                 yolo_model_path: str = "/app/models/yolo/yolo11n.pt",
                 device: str = "cuda",
                 tsundere_mode: bool = True):
        
        self.tsundere_mode = tsundere_mode
        self.device = device
        
        if self.tsundere_mode:
            logger.info("ふん！最強のWav2Lip+YOLOシステムを初期化してあげるわよ...")
        
        # YOLO検出器初期化
        self.yolo_detector = TsundereYOLODetector(
            model_path=yolo_model_path,
            device=device,
            tsundere_mode=tsundere_mode
        )
        
        # 顔整列器初期化
        self.face_aligner = TsundereFaceAligner(tsundere_mode=tsundere_mode)
        
        # Wav2Lipモデル読み込み
        try:
            self.wav2lip_model = load_model(wav2lip_model_path)
            if self.tsundere_mode:
                logger.info("べ、別にあなたのためじゃないけど...Wav2Lipモデルの準備できたわよ！")
        except Exception as e:
            if self.tsundere_mode:
                logger.error(f"な、何よ！Wav2Lipモデルの読み込みに失敗したじゃない！: {e}")
            raise
        
        # RTX 3050最適化適用
        self._optimize_for_rtx3050()
    
    def _optimize_for_rtx3050(self):
        """RTX 3050用最適化"""
        if self.tsundere_mode:
            logger.info("ふん！RTX 3050の最適化なんて朝飯前よ...")
        
        try:
            # YOLO最適化
            self.yolo_detector.optimize_for_rtx3050()
            
            # Wav2Lip最適化
            if torch.cuda.is_available():
                self.wav2lip_model = self.wav2lip_model.half()  # FP16
                
            if self.tsundere_mode:
                logger.info("べ、別にすごくないけど...RTX 3050最適化完了よ！")
                
        except Exception as e:
            if self.tsundere_mode:
                logger.warning(f"最適化で少し問題があったけど...まあ動くからいいわ: {e}")
    
    def process_video_with_audio(self,
                                video_path: str,
                                audio_path: str,
                                output_path: str,
                                quality: str = "Enhanced",
                                target_face_id: Optional[int] = None) -> bool:
        """
        動画と音声から口パク動画を生成
        
        Args:
            video_path: 入力動画パス
            audio_path: 入力音声パス
            output_path: 出力動画パス
            quality: 品質設定 (Fast/Improved/Enhanced)
            target_face_id: 対象顔ID（None=最大顔自動選択）
            
        Returns:
            bool: 処理成功フラグ
        """
        if self.tsundere_mode:
            logger.info("ふん！また口パク動画を作ってって言うのね...")
            logger.info("べ、別にあなたのためじゃないんだからね！でもちゃんと作ってあげるわ！")
        
        try:
            start_time = time.time()
            
            # 動画読み込み
            video_stream = cv2.VideoCapture(video_path)
            fps = video_stream.get(cv2.CAP_PROP_FPS)
            total_frames = int(video_stream.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if self.tsundere_mode:
                logger.info(f"動画情報: {total_frames}フレーム, {fps}FPS")
            
            # 音声読み込み
            audio, sr = librosa.load(audio_path, sr=16000)
            
            # フレーム処理用リスト
            frames = []
            face_detections = {}
            
            # 1. 全フレームの顔検出（バッチ処理）
            if self.tsundere_mode:
                logger.info("ステップ1: 全フレームの顔検出中...")
            
            frame_idx = 0
            while True:
                ret, frame = video_stream.read()
                if not ret:
                    break
                
                frames.append(frame)
                
                # YOLO顔検出
                faces = self.yolo_detector.detect_faces(frame)
                face_detections[frame_idx] = faces
                
                frame_idx += 1
                
                if frame_idx % 100 == 0 and self.tsundere_mode:
                    logger.info(f"処理中... {frame_idx}/{total_frames}フレーム")
            
            video_stream.release()
            
            # 2. 対象顔の選択
            target_face = self._select_target_face(face_detections, target_face_id)
            
            if target_face is None:
                if self.tsundere_mode:
                    logger.error("あら？対象となる顔が見つからないじゃない！ちゃんとした動画を使いなさいよ！")
                return False
            
            # 3. Wav2Lip処理
            if self.tsundere_mode:
                logger.info("ステップ2: Wav2Lipで口パク生成中...")
            
            processed_frames = self._apply_wav2lip_to_frames(
                frames, face_detections, target_face, audio, quality
            )
            
            # 4. 動画出力
            if self.tsundere_mode:
                logger.info("ステップ3: 動画出力中...")
            
            success = self._write_output_video(
                processed_frames, audio_path, output_path, fps
            )
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            if success and self.tsundere_mode:
                logger.info(f"ふん！{processing_time:.1f}秒で完璧な口パク動画を作ってあげたわよ！")
                logger.info("感謝しなさいよね！")
            
            return success
            
        except Exception as e:
            if self.tsundere_mode:
                logger.error(f"も、もう！処理中にエラーが出たじゃない！: {e}")
            return False
    
    def _select_target_face(self, 
                           face_detections: Dict[int, List[Dict]], 
                           target_face_id: Optional[int] = None) -> Optional[Dict]:
        """対象顔を選択"""
        
        if target_face_id is not None:
            # 指定IDの顔を検索
            for frame_faces in face_detections.values():
                for face in frame_faces:
                    if face.get('track_id') == target_face_id:
                        return face
        
        # 自動選択：最も頻繁に検出される最大の顔
        face_stats = {}
        
        for frame_faces in face_detections.values():
            for face in frame_faces:
                bbox = face['bbox']
                area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                confidence = face['confidence']
                
                # 顔の特徴でグループ化（簡易版）
                center_x = (bbox[0] + bbox[2]) // 2
                center_y = (bbox[1] + bbox[3]) // 2
                face_key = (center_x // 50, center_y // 50, area // 1000)  # 粗いグルーピング
                
                if face_key not in face_stats:
                    face_stats[face_key] = {
                        'count': 0,
                        'total_area': 0,
                        'total_confidence': 0,
                        'sample_face': face
                    }
                
                face_stats[face_key]['count'] += 1
                face_stats[face_key]['total_area'] += area
                face_stats[face_key]['total_confidence'] += confidence
        
        if not face_stats:
            return None
        
        # 最適な顔を選択（出現回数 * 平均面積 * 平均信頼度）
        best_face = None
        best_score = 0
        
        for stats in face_stats.values():
            avg_area = stats['total_area'] / stats['count']
            avg_confidence = stats['total_confidence'] / stats['count']
            score = stats['count'] * avg_area * avg_confidence
            
            if score > best_score:
                best_score = score
                best_face = stats['sample_face']
        
        if self.tsundere_mode and best_face:
            logger.info(f"ふん！最適な顔を選択してあげたわよ。スコア: {best_score:.0f}")
        
        return best_face
    
    def _apply_wav2lip_to_frames(self,
                                frames: List[np.ndarray],
                                face_detections: Dict[int, List[Dict]],
                                target_face: Dict,
                                audio: np.ndarray,
                                quality: str) -> List[np.ndarray]:
        """フレームにWav2Lipを適用"""
        
        processed_frames = []
        
        # 音声の特徴量抽出
        mel_chunks = self._extract_mel_chunks(audio)
        
        for frame_idx, frame in enumerate(frames):
            try:
                # 対応する音声チャンクを取得
                mel_chunk_idx = min(frame_idx, len(mel_chunks) - 1)
                mel_chunk = mel_chunks[mel_chunk_idx]
                
                # フレーム内の対象顔を検索
                frame_faces = face_detections.get(frame_idx, [])
                matching_face = self._find_matching_face(frame_faces, target_face)
                
                if matching_face:
                    # 顔整列
                    aligned_face = self.face_aligner.align_face_for_wav2lip(
                        frame, matching_face['bbox']
                    )
                    
                    # Wav2Lip推論
                    generated_face = self._run_wav2lip_inference(
                        aligned_face, mel_chunk
                    )
                    
                    # フレームに合成
                    processed_frame = self._composite_face_to_frame(
                        frame, generated_face, matching_face['bbox'], quality
                    )
                    
                    processed_frames.append(processed_frame)
                else:
                    # 顔が見つからない場合は元フレームを使用
                    processed_frames.append(frame)
                    
            except Exception as e:
                if self.tsundere_mode:
                    logger.warning(f"フレーム{frame_idx}の処理でエラー: {e}")
                processed_frames.append(frame)
        
        return processed_frames
    
    def _extract_mel_chunks(self, audio: np.ndarray) -> List[np.ndarray]:
        """音声からMELスペクトログラムチャンクを抽出"""
        
        # Wav2Lip標準の設定
        fps = 25  # Wav2Lip標準FPS
        mel_step_size = 16
        
        mel = librosa.feature.melspectrogram(y=audio, sr=16000, n_mels=80)
        mel = np.log(mel + 1e-8)
        
        # フレーム単位でチャンク分割
        mel_chunks = []
        mel_idx_multiplier = 80. / fps
        
        frame_count = int(len(audio) / 16000 * fps)
        
        for i in range(frame_count):
            start_idx = int(i * mel_idx_multiplier)
            end_idx = start_idx + mel_step_size
            
            if end_idx <= mel.shape[1]:
                mel_chunk = mel[:, start_idx:end_idx]
            else:
                # パディング
                mel_chunk = np.zeros((80, mel_step_size))
                available_frames = mel.shape[1] - start_idx
                if available_frames > 0:
                    mel_chunk[:, :available_frames] = mel[:, start_idx:]
            
            mel_chunks.append(mel_chunk)
        
        return mel_chunks
    
    def _find_matching_face(self, frame_faces: List[Dict], target_face: Dict) -> Optional[Dict]:
        """フレーム内で対象顔にマッチする顔を検索"""
        
        if not frame_faces:
            return None
        
        target_bbox = target_face['bbox']
        target_center = ((target_bbox[0] + target_bbox[2]) // 2,
                        (target_bbox[1] + target_bbox[3]) // 2)
        target_area = (target_bbox[2] - target_bbox[0]) * (target_bbox[3] - target_bbox[1])
        
        best_match = None
        best_score = float('inf')
        
        for face in frame_faces:
            bbox = face['bbox']
            center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            
            # 距離とサイズの差でスコア計算
            distance = np.sqrt((center[0] - target_center[0])**2 + 
                             (center[1] - target_center[1])**2)
            area_diff = abs(area - target_area) / target_area
            
            score = distance + area_diff * 100
            
            if score < best_score:
                best_score = score
                best_match = face
        
        return best_match
    
    def _run_wav2lip_inference(self, face_image: np.ndarray, mel_chunk: np.ndarray) -> np.ndarray:
        """Wav2Lip推論実行"""
        
        # 入力準備
        face_tensor = torch.FloatTensor(face_image).permute(2, 0, 1).unsqueeze(0)
        mel_tensor = torch.FloatTensor(mel_chunk).unsqueeze(0)
        
        if self.device == "cuda":
            face_tensor = face_tensor.cuda()
            mel_tensor = mel_tensor.cuda()
        
        # 推論実行
        with torch.no_grad():
            pred = self.wav2lip_model(mel_tensor, face_tensor)
        
        # 後処理
        pred = pred.squeeze(0).permute(1, 2, 0).cpu().numpy()
        pred = np.clip(pred, 0, 1) * 255
        
        return pred.astype(np.uint8)
    
    def _composite_face_to_frame(self, 
                                frame: np.ndarray, 
                                generated_face: np.ndarray, 
                                bbox: Tuple[int, int, int, int],
                                quality: str) -> np.ndarray:
        """生成された顔をフレームに合成"""
        
        x1, y1, x2, y2 = bbox
        
        # 生成された顔をバウンディングボックスサイズにリサイズ
        resized_face = cv2.resize(generated_face, (x2 - x1, y2 - y1))
        
        # 品質設定に応じて合成
        result_frame = frame.copy()
        
        if quality == "Fast":
            # 直接置換
            result_frame[y1:y2, x1:x2] = resized_face
            
        elif quality == "Improved":
            # フェザーマスクで合成
            mask = self._create_feather_mask(x2 - x1, y2 - y1)
            
            for c in range(3):
                result_frame[y1:y2, x1:x2, c] = (
                    resized_face[:, :, c] * mask +
                    frame[y1:y2, x1:x2, c] * (1 - mask)
                )
        
        elif quality == "Enhanced":
            # GFPGAN + フェザーマスク（実装は簡略化）
            # 実際の実装ではGFPGANで顔を高画質化してから合成
            enhanced_face = self._enhance_face_quality(resized_face)
            mask = self._create_feather_mask(x2 - x1, y2 - y1)
            
            for c in range(3):
                result_frame[y1:y2, x1:x2, c] = (
                    enhanced_face[:, :, c] * mask +
                    frame[y1:y2, x1:x2, c] * (1 - mask)
                )
        
        return result_frame
    
    def _create_feather_mask(self, width: int, height: int, feather_size: int = 5) -> np.ndarray:
        """フェザーマスクを作成"""
        
        mask = np.ones((height, width), dtype=np.float32)
        
        # エッジからフェザーサイズ分を徐々に透明に
        for i in range(feather_size):
            alpha = (i + 1) / feather_size
            mask[i, :] *= alpha
            mask[height-1-i, :] *= alpha
            mask[:, i] *= alpha
            mask[:, width-1-i] *= alpha
        
        return mask
    
    def _enhance_face_quality(self, face_image: np.ndarray) -> np.ndarray:
        """顔画質向上（GFPGAN簡易版）"""
        
        # 実際の実装ではGFPGANを使用
        # ここでは簡易的なシャープニング
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        enhanced = cv2.filter2D(face_image, -1, kernel)
        
        # 元画像と合成（過度なシャープニングを避ける）
        enhanced = cv2.addWeighted(face_image, 0.7, enhanced, 0.3, 0)
        
        return enhanced
    
    def _write_output_video(self,
                           frames: List[np.ndarray],
                           audio_path: str,
                           output_path: str,
                           fps: float) -> bool:
        """出力動画を書き込み"""
        
        try:
            height, width = frames[0].shape[:2]
            
            # 一時動画ファイル作成
            temp_video_path = output_path.replace('.mp4', '_temp.mp4')
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))
            
            for frame in frames:
                video_writer.write(frame)
            
            video_writer.release()
            
            # FFmpegで音声合成
            import subprocess
            
            cmd = [
                'ffmpeg', '-y',
                '-i', temp_video_path,
                '-i', audio_path,
                '-c:v', 'libx264',
                '-c:a', 'aac',
                '-strict', '-2',
                output_path
            ]
            
            subprocess.run(cmd, check=True, capture_output=True)
            
            # 一時ファイル削除
            Path(temp_video_path).unlink()
            
            return True
            
        except Exception as e:
            if self.tsundere_mode:
                logger.error(f"動画出力でエラーが出たじゃない！: {e}")
            return False


def test_integration():
    """統合システムのテスト"""
    print("🎭 ツンデレWav2Lip+YOLO統合システムテスト開始！")
    
    # エンジン初期化
    engine = TsundereWav2LipYOLOEngine(tsundere_mode=True)
    
    print("✅ 統合システム初期化完了！")
    print("ふん！完璧なシステムができたじゃない！")


if __name__ == "__main__":
    test_integration()