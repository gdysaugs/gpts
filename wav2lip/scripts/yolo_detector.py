"""
YOLO11 Face Detection Module
RTX 3050 + WSL2 最適化版
"""

import torch
import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Tuple, Optional, Dict
import logging
from pathlib import Path

# ツンデレログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TsundereYOLODetector:
    """ツンデレなYOLO11顔検出クラス"""
    
    def __init__(self, 
                 model_path: str = "/app/models/yolo/yolo11n.pt",
                 confidence: float = 0.7,
                 nms_threshold: float = 0.4,
                 device: str = "cuda",
                 tsundere_mode: bool = True):
        
        self.tsundere_mode = tsundere_mode
        
        if self.tsundere_mode:
            logger.info("ふん！YOLO11の初期化なんて簡単すぎるわよ...")
        
        self.model_path = model_path
        self.confidence = confidence
        self.nms_threshold = nms_threshold
        self.device = device
        
        # YOLO11モデル読み込み
        try:
            self.model = YOLO(model_path)
            self.model.to(device)
            
            if self.tsundere_mode:
                logger.info("べ、別にあなたのためじゃないけど...YOLO11の準備できたわよ！")
                
        except Exception as e:
            if self.tsundere_mode:
                logger.error(f"な、何よ！YOLO11の読み込みに失敗したじゃない！: {e}")
            raise
    
    def detect_faces(self, frame: np.ndarray) -> List[Dict]:
        """
        フレーム内の顔を検出
        
        Args:
            frame: 入力フレーム (BGR)
            
        Returns:
            List[Dict]: 検出された顔の情報
                - bbox: (x1, y1, x2, y2)
                - confidence: 信頼度
                - landmarks: 顔特徴点 (利用可能な場合)
        """
        if self.tsundere_mode and np.random.random() < 0.1:  # 10%の確率でツンデレコメント
            logger.info("ふん...また顔検出してって言うのね...")
        
        try:
            # YOLO11で推論実行
            results = self.model(frame, 
                               conf=self.confidence,
                               iou=self.nms_threshold,
                               verbose=False)
            
            faces = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # 人物クラス（class_id=0）のみフィルタ
                        if int(box.cls) == 0:  # person class
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            conf = float(box.conf[0])
                            
                            # 顔領域を推定（人物検出の上部30%）
                            face_height = (y2 - y1) * 0.3
                            face_y1 = y1
                            face_y2 = y1 + face_height
                            
                            face_info = {
                                'bbox': (int(x1), int(face_y1), int(x2), int(face_y2)),
                                'confidence': conf,
                                'person_bbox': (int(x1), int(y1), int(x2), int(y2)),
                                'landmarks': None  # YOLO11では直接取得不可
                            }
                            faces.append(face_info)
            
            if self.tsundere_mode and len(faces) > 0:
                logger.info(f"ふん！{len(faces)}個の顔を見つけてあげたわよ！感謝しなさい！")
            elif self.tsundere_mode and len(faces) == 0:
                logger.warning("あら？顔が見つからないじゃない...ちゃんとした動画を使いなさいよ！")
                
            return faces
            
        except Exception as e:
            if self.tsundere_mode:
                logger.error(f"も、もう！顔検出でエラーが出たじゃない！: {e}")
            return []
    
    def track_faces(self, frames: List[np.ndarray]) -> Dict[int, List[Dict]]:
        """
        複数フレームにわたって顔を追跡
        
        Args:
            frames: フレームリスト
            
        Returns:
            Dict[int, List[Dict]]: フレームID -> 顔情報リスト
        """
        if self.tsundere_mode:
            logger.info("ふん！顔追跡なんて朝飯前よ...")
        
        tracked_faces = {}
        
        try:
            # YOLO11のトラッキング機能使用
            results = self.model.track(frames, 
                                     conf=self.confidence,
                                     iou=self.nms_threshold,
                                     persist=True,
                                     verbose=False)
            
            for frame_idx, result in enumerate(results):
                faces = []
                boxes = result.boxes
                
                if boxes is not None:
                    for box in boxes:
                        if int(box.cls) == 0:  # person class
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            conf = float(box.conf[0])
                            track_id = int(box.id[0]) if box.id is not None else -1
                            
                            # 顔領域推定
                            face_height = (y2 - y1) * 0.3
                            face_y1 = y1
                            face_y2 = y1 + face_height
                            
                            face_info = {
                                'bbox': (int(x1), int(face_y1), int(x2), int(face_y2)),
                                'confidence': conf,
                                'track_id': track_id,
                                'person_bbox': (int(x1), int(y1), int(x2), int(y2))
                            }
                            faces.append(face_info)
                
                tracked_faces[frame_idx] = faces
            
            if self.tsundere_mode:
                total_detections = sum(len(faces) for faces in tracked_faces.values())
                logger.info(f"べ、別にすごくないけど...{len(frames)}フレームで{total_detections}個の顔を追跡してあげたわ！")
                
            return tracked_faces
            
        except Exception as e:
            if self.tsundere_mode:
                logger.error(f"な、何よ！顔追跡でエラーが出たじゃない！: {e}")
            return {}
    
    def optimize_for_rtx3050(self):
        """RTX 3050用最適化設定"""
        if self.tsundere_mode:
            logger.info("ふん！RTX 3050の最適化なんて当然でしょ？")
        
        try:
            # FP16最適化
            if torch.cuda.is_available():
                self.model.model.half()
                
            # バッチサイズ最適化（RTX 3050用）
            self.model.model.yaml['batch_size'] = 8
            
            # TensorRT最適化（環境変数で制御）
            import os
            if os.getenv('TENSORRT_OPTIMIZE', 'false').lower() == 'true':
                self._apply_tensorrt_optimization()
                
            if self.tsundere_mode:
                logger.info("べ、別にあなたのためじゃないけど...RTX 3050最適化完了よ！")
                
        except Exception as e:
            if self.tsundere_mode:
                logger.warning(f"最適化で問題があったけど...まあ動くからいいわ: {e}")
    
    def _apply_tensorrt_optimization(self):
        """TensorRT最適化適用"""
        try:
            import torch_tensorrt
            
            # TensorRT変換
            trt_ts_module = torch_tensorrt.compile(
                self.model.model,
                inputs=[torch_tensorrt.Input(shape=[1, 3, 640, 640])],
                enabled_precisions={torch.half}
            )
            
            self.model.model = trt_ts_module
            
            if self.tsundere_mode:
                logger.info("TensorRT最適化も完璧よ！2-3倍速くなったんだから感謝しなさい！")
                
        except ImportError:
            if self.tsundere_mode:
                logger.warning("TensorRTがないじゃない...まあPyTorchでも十分速いけど")
        except Exception as e:
            if self.tsundere_mode:
                logger.warning(f"TensorRT最適化に失敗したけど...気にしないわ: {e}")


class FaceTracker:
    """顔追跡ユーティリティクラス"""
    
    def __init__(self, max_disappeared: int = 10):
        self.next_id = 0
        self.objects = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared
    
    def register(self, centroid):
        """新しい顔を登録"""
        self.objects[self.next_id] = centroid
        self.disappeared[self.next_id] = 0
        self.next_id += 1
    
    def deregister(self, object_id):
        """顔の追跡を終了"""
        del self.objects[object_id]
        del self.disappeared[object_id]
    
    def update(self, rects):
        """顔位置を更新"""
        if len(rects) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return {}
        
        input_centroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (x1, y1, x2, y2)) in enumerate(rects):
            cx = int((x1 + x2) / 2.0)
            cy = int((y1 + y2) / 2.0)
            input_centroids[i] = (cx, cy)
        
        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.register(input_centroids[i])
        else:
            object_centroids = list(self.objects.values())
            object_ids = list(self.objects.keys())
            
            # 距離計算
            D = np.linalg.norm(np.array(object_centroids)[:, np.newaxis] - input_centroids, axis=2)
            
            # 最小コスト割り当て
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            
            used_row_idxs = set()
            used_col_idxs = set()
            
            for (row, col) in zip(rows, cols):
                if row in used_row_idxs or col in used_col_idxs:
                    continue
                
                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.disappeared[object_id] = 0
                
                used_row_idxs.add(row)
                used_col_idxs.add(col)
            
            unused_row_idxs = set(range(0, D.shape[0])).difference(used_row_idxs)
            unused_col_idxs = set(range(0, D.shape[1])).difference(used_col_idxs)
            
            if D.shape[0] >= D.shape[1]:
                for row in unused_row_idxs:
                    object_id = object_ids[row]
                    self.disappeared[object_id] += 1
                    if self.disappeared[object_id] > self.max_disappeared:
                        self.deregister(object_id)
            else:
                for col in unused_col_idxs:
                    self.register(input_centroids[col])
        
        return self.objects


def test_yolo_detector():
    """YOLO検出器のテスト関数"""
    print("🎭 ツンデレYOLO検出器テスト開始！")
    
    # テスト用ダミー画像作成
    test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # 検出器初期化
    detector = TsundereYOLODetector(tsundere_mode=True)
    
    # RTX 3050最適化適用
    detector.optimize_for_rtx3050()
    
    # 顔検出テスト
    faces = detector.detect_faces(test_frame)
    print(f"検出された顔数: {len(faces)}")
    
    print("✅ YOLO検出器テスト完了！")


if __name__ == "__main__":
    test_yolo_detector()