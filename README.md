# YOLO 中医药(TCM)实时检测系统

基于 **YOLO11n**（Ultralytics 框架）训练的 **99类中医药(TCM)** 实时目标检测系统。支持 USB/笔记本摄像头实时推理、目标筛选命中高亮、截图保存。

---

## 项目结构

```
01Camera/
├── main.py             # 主入口，启动实时检测 GUI
├── config.py           # 所有参数集中管理（模型路径、99类名、相机等）
├── camera.py           # 摄像头捕获模块（独立线程取帧）
├── detector.py         # YOLO 检测器模块（Ultralytics 推理接口）
├── ui.py               # Tkinter GUI（实时画面 + 打框 + 目标筛选 + 命中绿框）
├── requirements.txt    # Python 依赖
│
├── train/              # 训练相关
│   ├── data.yaml       # 训练数据集配置（99类 TCM.yolo26）
│   ├── train.py        # 训练脚本（CUDA, batch=8, epochs=40）
│   └── test.py         # 训练后测试（mAP 评估）
│
├── runs/detect/yolo100/train_*/  # 训练输出（best.pt 模型在此）
├── result/             # 测试结果
│   ├── run_test.py     # 独立测试脚本（打框 + 中文标签 + 置信度）
│   ├── test_images/    # 带标注的测试结果图（989张）
│   ├── test_report.json
│   └── test_report.txt
│
└── models/             # 可放置自定义模型文件
```

---

## 快速开始

### 1. 环境准备

```bash
# 激活 BC conda 环境
conda activate BC

# 安装依赖（如未安装）
pip install -r requirements.txt
pip install ultralytics
```

### 2. 启动实时检测

```bash
conda activate BC
python main.py --camera 0
```

GUI 窗口弹出后，点击 **▶ 开始检测**，摄像头画面中出现中药时自动打框标注。

**目标筛选功能：** 从顶部 🎯 下拉框选择一种中药，当检测到该品种时，画面**外框变为绿色**并显示 `✅ X 命中!`。

📷 截图保存至项目根目录 `screenshot_*.jpg`。

---

## 模型信息

| 项目 | 详情 |
|------|------|
| **模型架构** | **YOLO11n** (Ultralytics 框架) |
| **任务类型** | 目标检测 (Detection) |
| **类别数** | **99 类中医药** |
| **训练框架** | Ultralytics 8.4.96 |
| **预训练权重** | yolo11n.pt（COCO 预训练） |
| **训练数据** | TCM.yolo26（Roboflow，8978 张训练图） |
| **训练参数** | CUDA RTX 4050, 40 epochs, batch=8, imgsz=640 |
| **模型文件** | `runs/detect/yolo100/train_20260721_135809/weights/best.pt` |

### 99 类中医药完整列表

| ID | 英文名 | 中文名 |
|----|--------|--------|
| 0 | AbriprecatoriiSemen | 相思豆 |
| 1 | AlismatisRhizoma | 泽泻 |
| 2 | AmomiFructus | 砂仁 |
| 3 | AngelicaRoot | 当归 |
| 4 | AntelopeHom | 羚羊角 |
| 5 | ArctiiFructus | 牛蒡子 |
| 6 | ArecaePericarpium | 大腹皮 |
| 7 | ArecaeSemen | 槟榔 |
| 8 | ArmeniacaeSemenAmarum | 苦杏仁 |
| 9 | ArnebiaeRadix | 紫草 |
| 10 | AtractylodesMacrocephala | 白术 |
| 11 | AurantiiFructusImmaturus | 枳壳 |
| 12 | BistortaeRhizoma | 拳参 |
| 13 | BombyxBatryticatus | 僵蚕 |
| 14 | BupleuriRadix | 柴胡 |
| 15 | CanariiFructus | 青果 |
| 16 | CelosiaeCristataeFlos | 鸡冠花 |
| 17 | CelosiaeSemen | 青箱子 |
| 18 | ChaenomelisFructus | 木瓜 |
| 19 | ChrysanthemiFlos | 菊花 |
| 20 | Chuanxiong | 川芎 |
| 21 | CicadaePeriostracum | 蝉蜕 |
| 22 | CitriGrandisExocarpium | 橘红 |
| 23 | CitriSarcodactylisFructus | 佛手 |
| 24 | ClamShell | 蛤壳 |
| 25 | CoicisSemen | 薏苡仁 |
| 26 | CoptidisRhizoma | 黄连 |
| 27 | CrociStigma | 西红花 |
| 28 | CurculiginisRhizoma | 仙茅 |
| 29 | CurcumaLonga | 姜黄 |
| 30 | CuscutaeSemen | 菟丝子 |
| 31 | CynomoriiHerba | 锁阳 |
| 32 | DioscoreaeRhizoma | 山药 |
| 33 | EpimediiFolium | 淫羊藿 |
| 34 | EucommiaBark | 杜仲 |
| 35 | EupolyphagaSinensis | 土鳖虫 |
| 36 | FarfaraeFlos | 款冬花 |
| 37 | FritillariaeCirrhosaeBulbus | 川贝母 |
| 38 | GallaChinensis | 五倍子 |
| 39 | Ganoderma | 灵芝 |
| 40 | Garlic | 大蒜 |
| 41 | Gecko | 蛤蚧 |
| 42 | GinkgoBiloba | 银杏叶 |
| 43 | GleditsiaeFructusAbnormalis | 猪牙皂 |
| 44 | GleditsiaeSpina | 皂角刺 |
| 45 | GlycyrrhizaUralensis | 甘草 |
| 46 | GypsumFibrosum | 石膏 |
| 47 | Hawthorn | 山楂 |
| 48 | HouttuyniaeHerba | 鱼腥草 |
| 49 | ImperataeRhizoma | 白茅根 |
| 50 | InulaeFlos | 旋覆花 |
| 51 | JuglandisSemen | 核桃仁 |
| 52 | JujubaeFructus | 大枣 |
| 53 | KaempferiaeRhizoma | 沙姜 |
| 54 | KakiCalyx | 柿蒂 |
| 55 | Leech | 水蛭 |
| 56 | LigustriLucidiFructus | 女贞子 |
| 57 | LiliiBulbus | 百合 |
| 58 | LiquidambarisFructus | 路路通 |
| 59 | LonganArillus | 龙眼肉 |
| 60 | LoniceraJaponica | 金银花 |
| 61 | Luohanguo | 罗汉果 |
| 62 | LyciiFructus | 枸杞子 |
| 63 | MagnoliaBark | 厚朴 |
| 64 | MagnoliaeFlos | 辛夷 |
| 65 | MoriFructus | 桑葚 |
| 66 | MumeFructus | 乌梅 |
| 67 | NelumbinisPlumula | 莲子心 |
| 68 | NelumbinisSemen | 莲子 |
| 69 | Olibanum | 乳香 |
| 70 | OroxylumIndicum | 木蝴蝶 |
| 71 | Oyster | 牡蛎 |
| 72 | Pangdahai | 胖大海 |
| 73 | PersicaeSemen | 桃仁 |
| 74 | PiperCubeba | 苹拔茄 |
| 75 | PiperisLongiFructus | 荜茇 |
| 76 | PrunellaeSpica | 夏枯草 |
| 77 | RosaeChinensisFlos | 月季花 |
| 78 | RosaeLaevigataeFructus | 金樱子 |
| 79 | Sanqi | 三七 |
| 80 | SargentodexaeCaulis | 红藤 |
| 81 | SaussureaeInvolucrataeHerba | 天山雪莲 |
| 82 | Scolopendra | 蜈蚣 |
| 83 | Scorpion | 蝎子 |
| 84 | Seahorse | 海马 |
| 85 | SerpentisPeriostracum | 蛇蜕 |
| 86 | SiegesbeckiaeHerba | 豨莶草 |
| 87 | SparganiiRhizoma | 三棱 |
| 88 | TallGastrodiae | 天麻 |
| 89 | TaraxaciHerba | 蒲公英 |
| 90 | TetrapanacisMedulla | 通草 |
| 91 | ToosendanFructus | 川楝子 |
| 92 | TribuliFructus | 蒺藜 |
| 93 | TrionycisCarapax | 鳖甲 |
| 94 | TruestarAnise | 八角茴香 |
| 95 | TsaokoFructus | 草果 |
| 96 | Zhizi | 栀子 |
| 97 | ZingiberisRhizomaRecens | 生姜 |
| 98 | ginseng | 人参 |

---

## 重新训练

### 训练新模型

```bash
conda activate BC
cd d:\YOLO_c2\01Camera
python train/train.py
```

默认使用 `yolo11n.pt` 预训练权重。修改 `train/train.py` 中参数可调整：
- `--epochs` 训练轮数（默认 40）
- `--batch` 批大小（默认 8）
- `device=0` CUDA GPU / `device="cpu"` CPU
- `--model yolo11s.pt` 换用更大模型

训练完成后，`best.pt` 保存在 `runs/detect/yolo100/train_*/weights/` 。

### 集成新模型

1. 将 `best.pt` 复制到 `models/` 目录
2. 修改 `config.py` 中的 `YOLO_MODEL_PATH` 和 `CLASS_NAMES`
3. 重启 `python main.py --camera 0`

---

## 测试与评估

### 对测试集批量推理（打框 + 中文标签）

```bash
python result/run_test.py
```

输出：
- `result/test_images/` — 989 张带标注框 + 中文标签图片
- `result/test_report.json` — JSON 格式详细报告
- `result/test_report.txt` — 可读文本报告

### 计算 mAP@50

```bash
python train/test.py
```

---

## 依赖

| 包 | 版本 |
|----|------|
| ultralytics | ≥8.4 |
| torch | ≥2.5 (CUDA 12.1) |
| opencv-python | ≥4.8 |
| numpy | ≥1.24 |
| Pillow | ≥10.0 |

---

## 常见问题

**Q: 提示"无法打开摄像头"？**
- 修改 `config.py` 中 `CAMERA_INDEX`（0=笔记本, 1=外接/USB摄像头）
- 运行 `python main.py --list` 扫描可用摄像头

**Q: 检测框标注看不清楚？**
- 白底黑字标签，已优化对比度
- 可在 `detector.py` 的 `draw_detections()` 中调整字体颜色和大小

**Q: 如何用手机摄像头？**
- 手机安装 DroidCam / Iriun Webcam
- USB 连接后在 `config.py` 设置 `CAMERA_INDEX`