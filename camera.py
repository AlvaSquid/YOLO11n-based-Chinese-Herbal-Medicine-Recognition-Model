"""
摄像头捕获模块 — 支持 USB 手机摄像头 (DroidCam / IP Webcam)
"""

import cv2
import threading
import time
from config import CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS


class CameraStream:
    """
    使用独立线程循环读取摄像头帧，保证取帧不阻塞主线程。
    """

    def __init__(self, camera_index=CAMERA_INDEX,
                 width=CAMERA_WIDTH, height=CAMERA_HEIGHT, fps=CAMERA_FPS):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps

        self.cap = None
        self.frame = None
        self._running = False
        self._lock = threading.Lock()
        self._thread = None

    def start(self):
        """打开摄像头并启动取帧线程。"""
        self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        if not self.cap.isOpened():
            raise RuntimeError(
                f"无法打开摄像头 (索引={self.camera_index})。\n"
                f"请确认手机已通过 USB 连接，并启动了 DroidCam / IP Webcam 等应用。"
            )

        self._running = True
        self._thread = threading.Thread(target=self._grab_loop, daemon=True)
        self._thread.start()
        print(f"[Camera] 摄像头已启动: {self.width}x{self.height} @ index={self.camera_index}")

    def _grab_loop(self):
        """后台线程：持续抓取最新帧。"""
        while self._running:
            ret, frame = self.cap.read()
            if ret:
                with self._lock:
                    self.frame = frame
            else:
                time.sleep(0.005)  # 读失败时短暂休眠

    def read(self):
        """获取最新的摄像头帧 (线程安全)。"""
        with self._lock:
            if self.frame is None:
                return None
            return self.frame.copy()

    def release(self):
        """停止捕获并释放资源。"""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self.cap is not None:
            self.cap.release()
        print("[Camera] 摄像头已释放。")


def list_available_cameras(max_index=5):
    """打印系统中可用的摄像头索引，方便调试。"""
    available = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            available.append(i)
            print(f"  [摄像头 {i}] 可用")
            cap.release()
    if not available:
        print("  未检测到任何摄像头！")
    return available