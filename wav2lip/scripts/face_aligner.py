"""
Face Alignment Module for Wav2Lip Integration
RetinaFace + YOLO11 統合顔整列システム
"""

import cv2
import numpy as np
import torch
from typing import List, Tuple, Optional, Dict
import dlib
import logging
from pathlib import Path

# ツンデレログ設定
logger = logging.getLogger(__name__)

class TsundereFaceAligner:
    """ツンデレな顔整列クラス"""
    
    def __init__(self, 
                 predictor_path: str = "/app/models/face_detection/shape_predictor_68_face_landmarks.dat",
                 tsundere_mode: bool = True):
        
        self.tsundere_mode = tsundere_mode
        
        if self.tsundere_mode:
            logger.info("ふん！顔の整列なんて私にとっては朝飯前よ...")
        
        # dlib顔特徴点検出器初期化
        try:
            self.detector = dlib.get_frontal_face_detector()
            if Path(predictor_path).exists():
                self.predictor = dlib.shape_predictor(predictor_path)
            else:
                # 68点モデルがない場合のフォールバック
                self.predictor = None
                if self.tsundere_mode:
                    logger.warning("68点モデルがないじゃない...まあ5点でも十分よ！")
                    
        except Exception as e:
            if self.tsundere_mode:
                logger.error(f"な、何よ！顔特徴点検出器の初期化に失敗したじゃない！: {e}")
            raise
    
    def align_face_for_wav2lip(self, 
                              image: np.ndarray, 
                              bbox: Tuple[int, int, int, int],
                              target_size: Tuple[int, int] = (288, 192)) -> np.ndarray:
        """
        Wav2Lip用に顔を整列・リサイズ
        
        Args:
            image: 入力画像
            bbox: 顔のバウンディングボックス (x1, y1, x2, y2)
            target_size: 出力サイズ (width, height)
            
        Returns:
            np.ndarray: 整列された顔画像
        """
        if self.tsundere_mode and np.random.random() < 0.05:
            logger.info("べ、別に完璧な顔整列をしてあげるわけじゃないんだからね...")
        
        try:
            x1, y1, x2, y2 = bbox
            
            # 顔領域を少し拡張（Wav2Lip推奨）
            padding = 20
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(image.shape[1], x2 + padding)
            y2 = min(image.shape[0], y2 + padding)
            
            # 顔領域抽出
            face_image = image[y1:y2, x1:x2]
            
            # アスペクト比を維持しながらリサイズ
            h, w = face_image.shape[:2]
            target_w, target_h = target_size
            
            # アスペクト比計算
            aspect_ratio = w / h
            target_aspect_ratio = target_w / target_h
            
            if aspect_ratio > target_aspect_ratio:
                # 幅が基準
                new_w = target_w
                new_h = int(target_w / aspect_ratio)
            else:
                # 高さが基準
                new_h = target_h
                new_w = int(target_h * aspect_ratio)
            
            # リサイズ
            resized_face = cv2.resize(face_image, (new_w, new_h))
            
            # 中央配置でパディング
            aligned_face = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            
            # 中央配置計算
            start_y = (target_h - new_h) // 2
            start_x = (target_w - new_w) // 2
            
            aligned_face[start_y:start_y+new_h, start_x:start_x+new_w] = resized_face
            
            return aligned_face
            
        except Exception as e:
            if self.tsundere_mode:
                logger.error(f"も、もう！顔整列でエラーが出たじゃない！: {e}")
            # フォールバック：単純なリサイズ
            return cv2.resize(image[y1:y2, x1:x2], target_size)
    
    def get_facial_landmarks(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """
        顔特徴点を取得
        
        Args:
            image: 入力画像
            bbox: 顔のバウンディングボックス
            
        Returns:
            np.ndarray: 顔特徴点座標 (68点または5点)
        """
        if self.predictor is None:
            return None
        
        try:
            x1, y1, x2, y2 = bbox
            face_rect = dlib.rectangle(x1, y1, x2, y2)
            
            # 顔特徴点検出
            landmarks = self.predictor(image, face_rect)
            
            # numpy配列に変換
            points = np.array([(landmarks.part(i).x, landmarks.part(i).y) 
                             for i in range(landmarks.num_parts)])
            
            if self.tsundere_mode and np.random.random() < 0.05:
                logger.info(f"ふん！{len(points)}個の特徴点を検出してあげたわよ！")
            
            return points
            
        except Exception as e:
            if self.tsundere_mode:
                logger.warning(f"特徴点検出でエラーが出たけど...まあいいわ: {e}")
            return None
    
    def align_face_with_landmarks(self, 
                                 image: np.ndarray, 
                                 landmarks: np.ndarray,
                                 target_size: Tuple[int, int] = (288, 192)) -> np.ndarray:
        """
        顔特徴点を使用した高精度整列
        
        Args:
            image: 入力画像
            landmarks: 顔特徴点
            target_size: 出力サイズ
            
        Returns:
            np.ndarray: 整列された顔画像
        """
        if self.tsundere_mode:
            logger.info("ふん！特徴点を使った高精度整列なんて当然でしょ？")
        
        try:
            # 目の中心点を計算（68点モデルの場合）
            if len(landmarks) == 68:
                left_eye = landmarks[36:42].mean(axis=0)
                right_eye = landmarks[42:48].mean(axis=0)
                nose_tip = landmarks[30]
                mouth_center = landmarks[48:68].mean(axis=0)
            else:
                # 5点モデルの場合
                left_eye = landmarks[0]
                right_eye = landmarks[1]
                nose_tip = landmarks[2]
                mouth_center = landmarks[3:5].mean(axis=0)
            
            # 顔の角度計算
            eye_center = (left_eye + right_eye) / 2
            dy = right_eye[1] - left_eye[1]
            dx = right_eye[0] - left_eye[0]
            angle = np.degrees(np.arctan2(dy, dx))
            
            # 回転行列計算
            center = tuple(eye_center.astype(int))
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            
            # 画像回転
            rotated = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
            
            # 顔領域の計算（回転後）
            face_points = np.array([landmarks])
            rotated_points = cv2.transform(face_points, M)[0]
            
            x_coords = rotated_points[:, 0]
            y_coords = rotated_points[:, 1]
            
            x1, y1 = int(x_coords.min()), int(y_coords.min())
            x2, y2 = int(x_coords.max()), int(y_coords.max())
            
            # パディング追加
            padding = 20
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(rotated.shape[1], x2 + padding)
            y2 = min(rotated.shape[0], y2 + padding)
            
            # 顔領域抽出とリサイズ
            aligned_face = rotated[y1:y2, x1:x2]
            aligned_face = cv2.resize(aligned_face, target_size)
            
            if self.tsundere_mode:
                logger.info("べ、別にすごくないけど...完璧な整列ができたわね！")
            
            return aligned_face
            
        except Exception as e:
            if self.tsundere_mode:
                logger.warning(f"高精度整列でエラーが出たけど...普通の整列でいいわ: {e}")
            
            # フォールバック：バウンディングボックスベース
            x1, y1 = int(landmarks[:, 0].min()), int(landmarks[:, 1].min())
            x2, y2 = int(landmarks[:, 0].max()), int(landmarks[:, 1].max())
            return self.align_face_for_wav2lip(image, (x1, y1, x2, y2), target_size)
    
    def prepare_for_wav2lip(self, faces_data: List[Dict]) -> List[Dict]:
        """
        Wav2Lip用にフェイスデータを準備
        
        Args:
            faces_data: YOLO検出結果
            
        Returns:
            List[Dict]: Wav2Lip用に整列されたフェイスデータ
        """
        if self.tsundere_mode:
            logger.info("ふん！Wav2Lip用の準備なんて簡単よ...")
        
        prepared_faces = []
        
        for face_data in faces_data:
            try:
                # 必要な情報を抽出
                bbox = face_data['bbox']
                confidence = face_data['confidence']
                
                # 顔整列（基本版）
                # 実際の画像は呼び出し側で提供される
                face_info = {
                    'bbox': bbox,
                    'confidence': confidence,
                    'wav2lip_ready': True,
                    'target_size': (288, 192)  # Wav2Lip標準サイズ
                }
                
                # 特徴点情報があれば追加
                if 'landmarks' in face_data and face_data['landmarks'] is not None:
                    face_info['landmarks'] = face_data['landmarks']
                    face_info['high_precision'] = True
                else:
                    face_info['high_precision'] = False
                
                prepared_faces.append(face_info)
                
            except Exception as e:
                if self.tsundere_mode:
                    logger.warning(f"顔データの準備でエラーが出たけど...スキップするわ: {e}")
                continue
        
        if self.tsundere_mode and len(prepared_faces) > 0:
            logger.info(f"べ、別に頑張ったわけじゃないけど...{len(prepared_faces)}個の顔を準備してあげたわよ！")
        
        return prepared_faces


def calculate_face_quality_score(image: np.ndarray, bbox: Tuple[int, int, int, int]) -> float:
    """
    顔画像の品質スコアを計算
    
    Args:
        image: 入力画像
        bbox: 顔のバウンディングボックス
        
    Returns:
        float: 品質スコア (0.0-1.0)
    """
    try:
        x1, y1, x2, y2 = bbox
        face_image = image[y1:y2, x1:x2]
        
        # 解像度スコア
        face_area = (x2 - x1) * (y2 - y1)
        resolution_score = min(1.0, face_area / (100 * 100))  # 100x100を基準
        
        # 明度スコア
        gray_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray_face) / 255.0
        brightness_score = 1.0 - abs(brightness - 0.5) * 2  # 0.5が最適
        
        # 鮮明度スコア（ラプラシアン分散）
        laplacian = cv2.Laplacian(gray_face, cv2.CV_64F)
        sharpness_score = min(1.0, np.var(laplacian) / 1000.0)
        
        # 総合スコア
        quality_score = (resolution_score * 0.4 + 
                        brightness_score * 0.3 + 
                        sharpness_score * 0.3)
        
        return quality_score
        
    except Exception:
        return 0.0


def test_face_aligner():
    """顔整列器のテスト関数"""
    print("🎭 ツンデレ顔整列器テスト開始！")
    
    # テスト用ダミー画像作成
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    test_bbox = (100, 100, 200, 200)
    
    # 整列器初期化
    aligner = TsundereFaceAligner(tsundere_mode=True)
    
    # 顔整列テスト
    aligned_face = aligner.align_face_for_wav2lip(test_image, test_bbox)
    print(f"整列された顔画像サイズ: {aligned_face.shape}")
    
    # 品質スコア計算
    quality_score = calculate_face_quality_score(test_image, test_bbox)
    print(f"顔品質スコア: {quality_score:.3f}")
    
    print("✅ 顔整列器テスト完了！")


if __name__ == "__main__":
    test_face_aligner()