"""
主入口 - 启动实时 YOLO11n 99类TCM检测系统
====================================
用法:
    python main.py              # 使用真实模型（99类TCM）
    python main.py --list       # 列出可用摄像头
    python main.py --camera 1   # 指定摄像头索引
"""

import argparse
import sys
from config import CAMERA_INDEX
from camera import CameraStream, list_available_cameras
from detector import create_detector
from ui import DetectionUI


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO11n 99类TCM实时检测")
    parser.add_argument(
        "--camera", type=int, default=CAMERA_INDEX,
        help=f"摄像头索引 (默认: {CAMERA_INDEX})"
    )
    parser.add_argument(
        "--list", action="store_true",
        help="列出所有可用摄像头并退出"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.list:
        print("正在扫描摄像头...")
        list_available_cameras(max_index=10)
        sys.exit(0)

    # 创建检测器（默认真实模型）
    detector = create_detector(use_dummy=False)
    detector.load_model()

    # 启动摄像头
    cam = CameraStream(camera_index=args.camera)
    cam.start()

    # 启动 GUI
    ui = DetectionUI(cam, detector)
    try:
        ui.run()
    finally:
        cam.release()


if __name__ == "__main__":
    main()