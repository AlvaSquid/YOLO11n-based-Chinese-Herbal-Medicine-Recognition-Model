"""
TCM 中医药 99类 YOLO 检测模型训练 (CPU)
batch=8, epochs=40
输出到 yolo100/ 目录下
"""

import os
from datetime import datetime

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(BASE)

from ultralytics import YOLO

DATA_YAML = "yolo100/data.yaml"
OUTPUT_DIR = "yolo100"
RUN_NAME = datetime.now().strftime("train_%Y%m%d_%H%M%S")

print("=" * 55)
print("  TCM 99类中医药检测 训练")
print(f"  模型: yolo11n.pt")
print(f"  数据: {DATA_YAML}  (99类)")
print(f"  设备: CUDA (RTX 4050)  |  Epochs: 40  |  Batch: 8")
print(f"  输出: {OUTPUT_DIR}/{RUN_NAME}/")
print("=" * 55)

if __name__ == "__main__":
    model = YOLO("yolo11n.pt")
    model.train(
        data=DATA_YAML,
        epochs=40,
        batch=8,
        imgsz=640,
        device=0,
        project=OUTPUT_DIR,
        name=RUN_NAME,
        exist_ok=True,
        patience=10,
        save=True,
        save_period=10,
        plots=True,
        val=True,
        workers=0,   # 避免 Windows spawn 多进程问题
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        degrees=5.0, translate=0.1, scale=0.5,
        fliplr=0.5, mosaic=1.0, mixup=0.05,
    )

    best_pt = os.path.join(OUTPUT_DIR, RUN_NAME, "weights", "best.pt")
    print(f"\n[完成] 最佳模型: {best_pt}")
    print(f"[提示] 测试: python yolo100/test.py")
