"""
TCM 99类中医药检测模型测试脚本
=============================
功能:
  1. 对测试集逐张推理
  2. 画框 + 中文类别标注
  3. 计算 mAP@50
  4. 输出报告到 yolo100/test/
"""

import os
import sys
import cv2
import numpy as np
import json
from datetime import datetime

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(BASE)

from ultralytics import YOLO

# ========== 配置 ==========
def find_best_model():
    candidates = []
    for root, dirs, files in os.walk("yolo100"):
        for f in files:
            if f == "best.pt":
                candidates.append((os.path.getmtime(os.path.join(root, f)), os.path.join(root, f)))
    candidates.sort(reverse=True)
    return candidates[0][1] if candidates else "yolo11n.pt"

MODEL_PATH = find_best_model()
TEST_IMAGE_DIR = r"C:\Users\Wing\Desktop\TCM.yolo26\test\images"
TEST_LABEL_DIR = r"C:\Users\Wing\Desktop\TCM.yolo26\test\labels"
OUTPUT_DIR = os.path.join("yolo100", "test")
IMAGE_DIR = os.path.join(OUTPUT_DIR, "images")

CONF_THRESHOLD = 0.25

# 中文映射（与 data.yaml 顺序对应）
CN_NAMES = {
    0: "相思豆", 1: "泽泻", 2: "砂仁", 3: "当归", 4: "羚羊角",
    5: "牛蒡子", 6: "大腹皮", 7: "槟榔", 8: "苦杏仁", 9: "紫草",
    10: "白术", 11: "枳壳", 12: "拳参", 13: "僵蚕", 14: "柴胡",
    15: "青果", 16: "鸡冠花", 17: "青箱子", 18: "木瓜", 19: "菊花",
    20: "川芎", 21: "蝉蜕", 22: "橘红", 23: "佛手", 24: "蛤壳",
    25: "薏苡仁", 26: "黄连", 27: "西红花", 28: "仙茅", 29: "姜黄",
    30: "菟丝子", 31: "锁阳", 32: "山药", 33: "淫羊藿", 34: "杜仲",
    35: "土鳖虫", 36: "款冬花", 37: "川贝母", 38: "五倍子", 39: "灵芝",
    40: "大蒜", 41: "蛤蚧", 42: "银杏叶", 43: "猪牙皂", 44: "皂角刺",
    45: "甘草", 46: "石膏", 47: "山楂", 48: "鱼腥草", 49: "白茅根",
    50: "旋覆花", 51: "核桃仁", 52: "大枣", 53: "沙姜", 54: "柿蒂",
    55: "水蛭", 56: "女贞子", 57: "百合", 58: "路路通", 59: "龙眼肉",
    60: "金银花", 61: "罗汉果", 62: "枸杞子", 63: "厚朴", 64: "辛夷",
    65: "桑葚", 66: "乌梅", 67: "莲子心", 68: "莲子", 69: "乳香",
    70: "木蝴蝶", 71: "牡蛎", 72: "胖大海", 73: "桃仁", 74: "苹拔茄",
    75: "荜茇", 76: "夏枯草", 77: "月季花", 78: "金樱子", 79: "三七",
    80: "红藤", 81: "天山雪莲", 82: "蜈蚣", 83: "蝎子", 84: "海马",
    85: "蛇蜕", 86: "豨莶草", 87: "三棱", 88: "天麻", 89: "蒲公英",
    90: "通草", 91: "川楝子", 92: "蒺藜", 93: "鳖甲", 94: "八角茴香",
    95: "草果", 96: "栀子", 97: "生姜", 98: "人参",
}
COLORS = [(0,255,0),(255,0,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255),(128,255,0),(255,128,0)] * 15


def draw_chinese_label(img, text, pos, color):
    from PIL import Image, ImageDraw, ImageFont
    font_paths = ["C:\\Windows\\Fonts\\msyh.ttc","C:\\Windows\\Fonts\\simhei.ttf","C:\\Windows\\Fonts\\simsun.ttc","C:\\Windows\\Fonts\\simkai.ttf"]
    font = None
    for fp in font_paths:
        if os.path.exists(fp):
            font = ImageFont.truetype(fp, size=14)
            break
    if font is None: return img

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil)
    x, y = pos
    bbox = draw.textbbox((0,0), text, font=font)
    tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
    cx1, cy1, cx2, cy2 = x, max(y-th-4, 0), x+tw+6, y+2
    draw.rectangle([cx1,cy1,cx2,cy2], fill=(0,0,0))
    draw.rectangle([cx1,cy1,cx2,cy2], outline=color, width=1)
    draw.text((x+3, y-th-3), text, fill=(color[2],color[1],color[0]), font=font)
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def load_gt(ldir):
    gt = {}
    for f in os.listdir(ldir):
        if not f.endswith(".txt"): continue
        s = os.path.splitext(f)[0]
        boxes = []
        with open(os.path.join(ldir,f)) as fh:
            for l in fh:
                if not l.strip(): continue
                p = l.split()
                boxes.append({"class":int(p[0]),"bbox":list(map(float,p[1:5]))})
        gt[s] = boxes
    return gt


def calc_iou(a, b):
    xa1, ya1 = a[0]-a[2]/2, a[1]-a[3]/2
    xa2, ya2 = a[0]+a[2]/2, a[1]+a[3]/2
    xb1, yb1 = b[0]-b[2]/2, b[1]-b[3]/2
    xb2, yb2 = b[0]+b[2]/2, b[1]+b[3]/2
    ix1, iy1 = max(xa1,xb1), max(ya1,yb1)
    ix2, iy2 = min(xa2,xb2), min(ya2,yb2)
    if ix1>=ix2 or iy1>=iy2: return 0
    inter = (ix2-ix1)*(iy2-iy1)
    return inter/((xa2-xa1)*(ya2-ya1)+(xb2-xb1)*(yb2-yb1)-inter)


def compute_ap(recalls, precisions):
    pairs = sorted(zip(recalls, precisions), key=lambda x: x[0])
    ap = sum(max([p for r,p in pairs if r>=t]+[0]) for t in np.arange(0,1.1,0.1))/11
    return ap


def main():
    print("="*55)
    print("  TCM 99类检测 — 测试")
    print(f"  模型: {MODEL_PATH}")
    print("="*55)

    model = YOLO(MODEL_PATH)
    gt_all = load_gt(TEST_LABEL_DIR)

    os.makedirs(IMAGE_DIR, exist_ok=True)

    all_preds, all_gts, details = [], [], []
    imgs = [f for f in os.listdir(TEST_IMAGE_DIR) if f.lower().endswith(('.jpg','.png')) and f!="desktop.ini"]

    print(f"  推理 {len(imgs)} 张图片...")
    for i, f in enumerate(imgs):
        img = cv2.imread(os.path.join(TEST_IMAGE_DIR, f))
        if img is None: continue
        h, w = img.shape[:2]
        stem = os.path.splitext(f)[0]
        results = model(img, verbose=False)[0]
        img_dets = []

        if results.boxes is not None:
            for box in results.boxes:
                cid = int(box.cls[0])
                conf = float(box.conf[0])
                if conf < CONF_THRESHOLD: continue
                x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
                x1,y1,x2,y2 = max(0,x1),max(0,y1),min(w-1,x2),min(h-1,y2)
                cn = CN_NAMES.get(cid, f"c{cid}")
                c = COLORS[cid%len(COLORS)]
                cv2.rectangle(img, (x1,y1), (x2,y2), c, 2)
                img = draw_chinese_label(img, f"{cn} {conf:.2f}", (x2, y1), c)
                img_dets.append({"class_id":cid,"class_cn":cn,"confidence":round(conf,4),"bbox_pixel":(x1,y1,x2,y2)})
                all_preds.append((stem, cid, conf, [(x1+x2)/2/w, (y1+y2)/2/h, (x2-x1)/w, (y2-y1)/h]))

        gts = gt_all.get(stem, [])
        for g in gts: all_gts.append((stem, g["class"], g["bbox"]))

        cv2.imwrite(os.path.join(IMAGE_DIR, f"{stem}_labeled.jpg"), img)
        details.append({"image":f, "num_gt":len(gts), "num_det":len(img_dets), "detections":img_dets})
        if (i+1)%20==0: print(f"  {i+1}/{len(imgs)}")

    # mAP
    class_aps = {}
    for cid in range(99):
        cp = [(s,conf,b) for s,ci,conf,b in all_preds if ci==cid]
        cg = {s:[b for ss,cc,b in all_gts if ss==s and cc==cid] for s in set(s for s,cc,b in all_gts if cc==cid)}
        if not sum(len(v) for v in cg.values()): continue
        cp.sort(key=lambda x:x[1], reverse=True)
        tp,fp = [],[]
        matched = {s:set() for s in cg}
        for s,conf,pb in cp:
            best_i,best_j = 0,-1
            for j,gb in enumerate(cg.get(s,[])):
                iou = calc_iou(pb,gb)
                if iou>best_i: best_i,best_j = iou,j
            if best_i>=0.5 and best_j not in matched[s]:
                tp.append(1); fp.append(0); matched[s].add(best_j)
            else: tp.append(0); fp.append(1)
        tpc, fpc = np.cumsum(tp), np.cumsum(fp)
        tg = sum(len(v) for v in cg.values())
        if tg==0: continue
        rec = tpc/tg
        prec = np.divide(tpc, tpc+fpc, out=np.zeros_like(tpc,dtype=float), where=(tpc+fpc)>0)
        ap = compute_ap(rec, prec)
        class_aps[CN_NAMES.get(cid,f"c{cid}")] = round(ap,4)

    mAP = round(np.mean(list(class_aps.values())),4) if class_aps else 0
    print(f"\n  mAP@50 = {mAP}")

    report = {"test_time":datetime.now().strftime("%Y-%m-%d %H:%M:%S"),"model":MODEL_PATH,"num_images":len(imgs),"mAP50":mAP,"per_class_AP":class_aps,"total_gt":len(all_gts),"total_pred":len(all_preds),"details":details}
    with open(os.path.join(OUTPUT_DIR,"test_results.json"),"w",encoding="utf-8") as fh: json.dump(report,fh,ensure_ascii=False,indent=2)
    with open(os.path.join(OUTPUT_DIR,"test_report.txt"),"w",encoding="utf-8") as fh:
        fh.write(f"TCM 99类检测 — 测试报告\nmAP@50 = {mAP}\n\n")
        for n,ap in sorted(class_aps.items(), key=lambda x:-x[1]): fh.write(f"  {n:<12s} AP={ap:.4f}\n")
    print(f"  报告: {OUTPUT_DIR}/test_report.txt")


if __name__ == "__main__":
    main()