"""
配置文件 — 集中管理所有可调参数
"""

# ========== 摄像头配置 ==========
# 如果你的手机通过 USB 使用 DroidCam / IP Webcam / Iriun 等 App 作为虚拟摄像头
# OpenCV 会将其识别为一个可用的摄像头索引 (通常为 0 或 1)
CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

# ========== YOLO 模型路径 ==========
# YOLO11n, 99类TCM检测，训练于 TCM.yolo26 数据集
YOLO_MODEL_PATH = "runs/detect/yolo100/train_20260721_135809/weights/best.pt"   # YOLO11n, 99类TCM
# YOLO_MODEL_PATH = "models/your_model.pt"   # 替换为你的模型

# ========== 推理配置 ==========
CONFIDENCE_THRESHOLD = 0.25         # 置信度阈值，低于此值不画框（降低以适应低精度模型）
NMS_IOU_THRESHOLD = 0.45           # NMS 的 IoU 阈值
INPUT_SIZE = (640, 640)            # 模型输入尺寸

# ========== 类别标签 ==========
# 请根据你的模型训练时的类别顺序修改此列表
CLASS_NAMES = [
    "AbriprecatoriiSemen","AlismatisRhizoma","AmomiFructus","AngelicaRoot","AntelopeHom",
    "ArctiiFructus","ArecaePericarpium","ArecaeSemen","ArmeniacaeSemenAmarum","ArnebiaeRadix",
    "AtractylodesMacrocephala","AurantiiFructusImmaturus","BistortaeRhizoma","BombyxBatryticatus","BupleuriRadix",
    "CanariiFructus","CelosiaeCristataeFlos","CelosiaeSemen","ChaenomelisFructus","ChrysanthemiFlos",
    "Chuanxiong","CicadaePeriostracum","CitriGrandisExocarpium","CitriSarcodactylisFructus","ClamShell",
    "CoicisSemen","CoptidisRhizoma","CrociStigma","CurculiginisRhizoma","CurcumaLonga",
    "CuscutaeSemen","CynomoriiHerba","DioscoreaeRhizoma","EpimediiFolium","EucommiaBark",
    "EupolyphagaSinensis","FarfaraeFlos","FritillariaeCirrhosaeBulbus","GallaChinensis","Ganoderma",
    "Garlic","Gecko","GinkgoBiloba","GleditsiaeFructusAbnormalis","GleditsiaeSpina",
    "GlycyrrhizaUralensis","GypsumFibrosum","Hawthorn","HouttuyniaeHerba","ImperataeRhizoma",
    "InulaeFlos","JuglandisSemen","JujubaeFructus","KaempferiaeRhizoma","KakiCalyx",
    "Leech","LigustriLucidiFructus","LiliiBulbus","LiquidambarisFructus","LonganArillus",
    "LoniceraJaponica","Luohanguo","LyciiFructus","MagnoliaBark","MagnoliaeFlos",
    "MoriFructus","MumeFructus","NelumbinisPlumula","NelumbinisSemen","Olibanum",
    "OroxylumIndicum","Oyster","Pangdahai","PersicaeSemen","PiperCubeba",
    "PiperisLongiFructus","PrunellaeSpica","RosaeChinensisFlos","RosaeLaevigataeFructus","Sanqi",
    "SargentodexaeCaulis","SaussureaeInvolucrataeHerba","Scolopendra","Scorpion","Seahorse",
    "SerpentisPeriostracum","SiegesbeckiaeHerba","SparganiiRhizoma","TallGastrodiae","TaraxaciHerba",
    "TetrapanacisMedulla","ToosendanFructus","TribuliFructus","TrionycisCarapax","TruestarAnise",
    "TsaokoFructus","Zhizi","ZingiberisRhizomaRecens","ginseng",
]

# ========== UI 配置 ==========
WINDOW_TITLE = "YOLOv26 实时检测"
UI_UPDATE_INTERVAL_MS = 30        # UI 刷新间隔（毫秒）

# 画框颜色 (B, G, R)，循环使用
BOX_COLORS = [
    (0, 255, 0),      # 绿色
    (255, 0, 0),      # 蓝色
    (0, 0, 255),      # 红色
    (255, 255, 0),    # 青色
    (255, 0, 255),    # 品红
    (0, 255, 255),    # 黄色
    (128, 255, 0),    # 浅绿
    (255, 128, 0),    # 橙色
]

# 中文类别名，用于 UI 下拉框显示
CHINESE_CLASS_NAMES = [
    "相思豆","泽泻","砂仁","当归","羚羊角","牛蒡子","大腹皮","槟榔","苦杏仁","紫草",
    "白术","枳壳","拳参","僵蚕","柴胡","青果","鸡冠花","青箱子","木瓜","菊花",
    "川芎","蝉蜕","橘红","佛手","蛤壳","薏苡仁","黄连","西红花","仙茅","姜黄",
    "菟丝子","锁阳","山药","淫羊藿","杜仲","土鳖虫","款冬花","川贝母","五倍子","灵芝",
    "大蒜","蛤蚧","银杏叶","猪牙皂","皂角刺","甘草","石膏","山楂","鱼腥草","白茅根",
    "旋覆花","核桃仁","大枣","沙姜","柿蒂","水蛭","女贞子","百合","路路通","龙眼肉",
    "金银花","罗汉果","枸杞子","厚朴","辛夷","桑葚","乌梅","莲子心","莲子","乳香",
    "木蝴蝶","牡蛎","胖大海","桃仁","苹拔茄","荜茇","夏枯草","月季花","金樱子","三七",
    "红藤","天山雪莲","蜈蚣","蝎子","海马","蛇蜕","豨莶草","三棱","天麻","蒲公英",
    "通草","川楝子","蒺藜","鳖甲","八角茴香","草果","栀子","生姜","人参",
]
