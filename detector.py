"""
YOLO 检测器模块 — YOLO11n 99类TCM检测
============================================
基于 Ultralytics 框架，已加载训练好的 99 类中医药检测模型。
当前使用 YOLO11n (2.65M params)，通过 Ultralytics YOLO 接口推理。
============================================
"""

import cv2
import numpy as np
import time
import random
from config import (
    YOLO_MODEL_PATH,
    CONFIDENCE_THRESHOLD,
    NMS_IOU_THRESHOLD,
    INPUT_SIZE,
    CLASS_NAMES,
    BOX_COLORS
)


# ============================================================
#  A) 抽象检测器接口
# ============================================================

class BaseDetector:
    """所有检测器的基类，定义统一接口。"""

    def load_model(self):
        """加载模型权重。"""
        raise NotImplementedError

    def detect(self, frame: np.ndarray) -> list:
        """
        对一帧图像进行推理。
        Args:
            frame: BGR 格式的 numpy 数组 (H, W, 3)
        Returns:
            detections: list of dict，每个 dict 包含:
                {
                    "class_id": int,
                    "class_name": str,
                    "confidence": float,
                    "bbox": (x1, y1, x2, y2),
                }
        """
        raise NotImplementedError


# ============================================================
#  B) 模拟检测器 — 仅用于无模型时测试流程
# ============================================================

class DummyDetector(BaseDetector):
    """
    模拟检测器：每帧随机画 0~3 个假检测框。
    仅用于验证摄像头到UI链路是否通畅。
    """

    def __init__(self, class_names=None):
        self.class_names = class_names or CLASS_NAMES
        self._loaded = False

    def load_model(self):
        self._loaded = True
        print("[DummyDetector] 模拟模型已加载")

    def detect(self, frame: np.ndarray):
        if not self._loaded:
            self.load_model()
        h, w = frame.shape[:2]
        detections = []
        num_objects = random.randint(0, 3)
        for _ in range(num_objects):
            cls_id = random.randint(0, len(self.class_names) - 1)
            x1 = random.randint(10, w - 100)
            y1 = random.randint(10, h - 100)
            x2 = x1 + random.randint(50, 150)
            y2 = y1 + random.randint(50, 150)
            conf = round(random.uniform(CONFIDENCE_THRESHOLD, 0.99), 2)
            detections.append({
                "class_id": cls_id,
                "class_name": self.class_names[cls_id],
                "confidence": conf,
                "bbox": (x1, y1, x2, y2),
            })
        return detections


# ============================================================
#  C) YOLO 真实检测器 — YOLO11n 99类TCM
# ============================================================

class YOLOv26Detector(BaseDetector):
    """
    YOLO11n 99类中医药检测器。
    模型: runs/detect/yolo100/train_20260721_135809/weights/best.pt
    通过 Ultralytics YOLO 接口加载和推理。
    """

    def __init__(self, model_path=YOLO_MODEL_PATH, class_names=None,
                 conf_thresh=CONFIDENCE_THRESHOLD,
                 nms_thresh=NMS_IOU_THRESHOLD,
                 input_size=INPUT_SIZE):
        self.model_path = model_path
        self.class_names = class_names or CLASS_NAMES
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.input_size = input_size
        self.model = None
        self._loaded = False

    def load_model(self):
        """加载 YOLO11n 模型。"""
        from ultralytics import YOLO
        self.model = YOLO(self.model_path)
        self._loaded = True
        print(f"[YOLO11n] 模型已加载: {self.model_path}")

    def detect(self, frame: np.ndarray):
        """对一帧图像进行推理，返回检测结果列表。"""
        if not self._loaded:
            self.load_model()
        results = self.model(frame, verbose=False)[0]
        return self._parse_ultralytics_result(results, frame.shape)

    # ------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------

    def _parse_ultralytics_result(self, result, frame_shape):
        """解析 Ultralytics YOLO 输出为统一格式。"""
        h, w = frame_shape[:2]
        detections = []
        if result.boxes is not None:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                if conf < self.conf_thresh:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append({
                    "class_id": cls_id,
                    "class_name": self.class_names[cls_id]
                                  if cls_id < len(self.class_names)
                                  else f"class_{cls_id}",
                    "confidence": round(conf, 3),
                    "bbox": (int(x1), int(y1), int(x2), int(y2)),
                })
        return detections

    def _preprocess(self, frame):
        """预处理: resize, normalize。"""
        inp_w, inp_h = self.input_size
        blob = cv2.resize(frame, (inp_w, inp_h))
        blob = blob.astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)
        blob = np.expand_dims(blob, axis=0)
        return blob

    def _postprocess(self, outputs, frame_shape):
        """后处理: 解析ONNX输出。"""
        return []


# ============================================================
#  D) 画框工具函数
# ============================================================

def draw_detections(frame: np.ndarray, detections: list) -> np.ndarray:
    """在图像上绘制检测框、类别名和置信度（白底黑字标签）。"""
    annotated = frame.copy()
    h, w = annotated.shape[:2]

    for det in detections:
        cls_id = det["class_id"]
        cls_name = det["class_name"]
        conf = det["confidence"]
        x1, y1, x2, y2 = det["bbox"]

        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1))
        y2 = max(0, min(y2, h - 1))

        color = BOX_COLORS[cls_id % len(BOX_COLORS)]

        # 矩形框
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # 标签文字（白底黑字）
        label = f"{cls_name} {conf:.2f}"
        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
        )
        cv2.rectangle(
            annotated,
            (x1, y1 - text_h - baseline - 4),
            (x1 + text_w, y1),
            (255, 255, 255),
            -1,
        )
        cv2.putText(
            annotated, label,
            (x1, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2,
        )

    return annotated


# ============================================================
#  E) 工厂函数
# ============================================================

def create_detector(use_dummy=True) -> BaseDetector:
    """
    创建检测器实例。
    Args:
        use_dummy: True → 模拟检测器
                   False → YOLO11n 真实检测器（默认）
    """
    if use_dummy:
        return DummyDetector()
    else:
        return YOLOv26Detector()