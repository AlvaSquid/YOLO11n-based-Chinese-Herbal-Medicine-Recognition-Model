"""
GUI 界面模块 — 基于 Tkinter 的实时检测画面显示
包含目标种类筛选 + 命中绿色边框高亮
"""

import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import time
from config import (
    WINDOW_TITLE, UI_UPDATE_INTERVAL_MS, CLASS_NAMES, CHINESE_CLASS_NAMES
)


PASS_GREEN = "#00FF00"       # 命中绿
BORDER_DEFAULT = "#555555"   # 默认灰
BORDER_WIDTH = 6             # 外框宽度（像素）


class DetectionUI:
    """实时检测画面 GUI，支持目标种类筛选与命中绿框。"""

    def __init__(self, camera_stream, detector,
                 window_title=WINDOW_TITLE,
                 update_interval_ms=UI_UPDATE_INTERVAL_MS):
        self.camera = camera_stream
        self.detector = detector
        self.update_interval = update_interval_ms

        self._running = False
        self._fps_counter = FPSMeter(window=10)

        self.root = tk.Tk()

        # ---- 选中的目标 & 命中状态 (必须在 root 之后创建) ----
        self._target_cn = tk.StringVar()
        self._match_color = BORDER_DEFAULT  # 外框颜色
        self._match_text = tk.StringVar(value="目标: 未选中")
        self.root.title(window_title)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.configure(bg="#1e1e1e")

        # ---- 顶部控制栏 ----
        self.control_frame = ttk.Frame(self.root, padding=5)
        self.control_frame.pack(side=tk.TOP, fill=tk.X)

        self.btn_start = ttk.Button(
            self.control_frame, text="▶ 开始检测", command=self.start)
        self.btn_start.pack(side=tk.LEFT, padx=4)

        self.btn_stop = ttk.Button(
            self.control_frame, text="⏸ 暂停", command=self.stop, state=tk.DISABLED)
        self.btn_stop.pack(side=tk.LEFT, padx=4)

        self.btn_snapshot = ttk.Button(
            self.control_frame, text="📷 截图", command=self._save_snapshot)
        self.btn_snapshot.pack(side=tk.LEFT, padx=4)

        # 类别下拉框
        ttk.Label(self.control_frame, text="🎯 目标:").pack(side=tk.LEFT, padx=(10, 2))
        self.combo_target = ttk.Combobox(
            self.control_frame, textvariable=self._target_cn,
            values=["（不限）"] + list(CHINESE_CLASS_NAMES),
            state="readonly", width=10
        )
        self.combo_target.set("（不限）")
        self.combo_target.pack(side=tk.LEFT, padx=2)
        self.combo_target.bind("<<ComboboxSelected>>", self._on_target_change)

        # 命中指示
        self.lbl_match = ttk.Label(
            self.control_frame, textvariable=self._match_text,
            font=("Microsoft YaHei", 10, "bold"), foreground=BORDER_DEFAULT
        )
        self.lbl_match.pack(side=tk.LEFT, padx=10)

        self.lbl_fps = ttk.Label(self.control_frame, text="FPS: --", font=("Consolas", 10))
        self.lbl_fps.pack(side=tk.RIGHT, padx=10)

        self.lbl_status = ttk.Label(self.control_frame, text="状态: 就绪", foreground="gray")
        self.lbl_status.pack(side=tk.RIGHT, padx=10)

        # ---- 主画面容器（带可变色边框） ----
        self.border_frame = tk.Frame(
            self.root,
            bg=BORDER_DEFAULT,
            bd=0,
            highlightbackground=BORDER_DEFAULT,
            highlightthickness=BORDER_WIDTH,
            highlightcolor=BORDER_DEFAULT,
        )
        self.border_frame.pack(padx=6, pady=6)

        self.canvas = tk.Canvas(
            self.border_frame,
            width=self.camera.width,
            height=self.camera.height,
            bg="black",
            highlightthickness=0,
            bd=0,
        )
        self.canvas.pack()

        # ---- 底部检测结果列表 ----
        self.result_frame = ttk.LabelFrame(self.root, text="检测结果", padding=5)
        self.result_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=4, pady=4)

        self.result_text = tk.Text(
            self.result_frame, height=6, width=70,
            font=("Consolas", 9), state=tk.DISABLED,
            bg="#1e1e1e", fg="#d4d4d4",
            insertbackground="white"
        )
        self.result_text.pack(fill=tk.X)

        self._current_tk_image = None

    # ---- 控制方法 ----
    def start(self):
        if self._running:
            return
        self._running = True
        self.btn_start.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        self._set_status("运行中", "green")
        self.root.after(self.update_interval, self._update_loop)

    def stop(self):
        self._running = False
        self.btn_start.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)
        self._set_status("已暂停", "orange")

    def _on_close(self):
        self.stop()
        self.root.destroy()

    def _set_status(self, text, color="gray"):
        self.lbl_status.config(text=f"状态: {text}", foreground=color)

    def _on_target_change(self, event):
        """当下拉框选择变化时重置边框。"""
        val = self._target_cn.get()
        if val == "（不限）":
            self._match_text.set("目标: 未选中")
            self.lbl_match.config(foreground=BORDER_DEFAULT)
        else:
            self._match_text.set(f"目标: {val}")
            self.lbl_match.config(foreground="#FFA500")
        self._match_color = BORDER_DEFAULT
        self.border_frame.configure(highlightbackground=BORDER_DEFAULT)

    # ---- 主循环 ----
    def _update_loop(self):
        if not self._running:
            return

        frame = self.camera.read()
        if frame is None:
            self.root.after(self.update_interval, self._update_loop)
            return

        detections = self.detector.detect(frame)

        annotated = frame
        if detections:
            from detector import draw_detections
            annotated = draw_detections(frame, detections)

        fps = self._fps_counter.tick()
        self.lbl_fps.config(text=f"FPS: {fps:.1f}")

        # ------- 目标匹配 & 外框变色 -------
        target_cn = self._target_cn.get()
        matched = False
        if target_cn != "（不限）" and detections:
            for d in detections:
                # 查找中文名匹配的检测结果
                en_name = d["class_name"]
                for en, cn in zip(CLASS_NAMES, CHINESE_CLASS_NAMES):
                    if en == en_name and cn == target_cn:
                        matched = True
                        break
                if matched:
                    break

        new_color = PASS_GREEN if matched else BORDER_DEFAULT
        if new_color != self._match_color:
            self._match_color = new_color
            self.border_frame.configure(highlightbackground=new_color)
            if matched:
                self._match_text.set(f"✅ {target_cn} 命中!")
                self.lbl_match.config(foreground=PASS_GREEN)
            elif target_cn != "（不限）":
                self._match_text.set(f"🔍 {target_cn} 未命中")
                self.lbl_match.config(foreground="#FFA500")

        # ------- Canvas 更新 -------
        rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        self._current_tk_image = ImageTk.PhotoImage(image)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self._current_tk_image)

        self._update_result_list(detections)
        self.root.after(self.update_interval, self._update_loop)

    def _update_result_list(self, detections):
        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete(1.0, tk.END)
        if not detections:
            self.result_text.insert(tk.END, "（未检测到目标）\n")
        else:
            for i, det in enumerate(detections, 1):
                en = det["class_name"]
                conf = det["confidence"]
                x1, y1, x2, y2 = det["bbox"]
                # 找到对应的中文名
                cn = en
                for en_ref, cn_ref in zip(CLASS_NAMES, CHINESE_CLASS_NAMES):
                    if en_ref == en:
                        cn = cn_ref
                        break
                line = (
                    f"[{i}] {cn:<6s}({en})  conf={conf:.2f}  "
                    f"bbox=({x1},{y1})-({x2},{y2})\n"
                )
                self.result_text.insert(tk.END, line)
        self.result_text.config(state=tk.DISABLED)

    def _save_snapshot(self):
        frame = self.camera.read()
        if frame is None:
            messagebox.showwarning("截图", "当前无画面可截图。")
            return
        detections = self.detector.detect(frame)
        from detector import draw_detections
        annotated = draw_detections(frame, detections)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"screenshot_{timestamp}.jpg"
        cv2.imwrite(filename, annotated)
        messagebox.showinfo("截图", f"已保存到: {filename}")

    def run(self):
        self.root.mainloop()


class FPSMeter:
    def __init__(self, window=30):
        self.window = window
        self._ticks = []

    def tick(self) -> float:
        now = time.perf_counter()
        self._ticks.append(now)
        if len(self._ticks) > self.window:
            self._ticks.pop(0)
        if len(self._ticks) < 2:
            return 0.0
        elapsed = self._ticks[-1] - self._ticks[0]
        return (len(self._ticks) - 1) / elapsed if elapsed > 0 else 0.0