from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

# ─── Dataset ayarlari ────────────────────────────────────────────────────────
DATASET_NAME = "9050_3_Mdataset"
DRIVE_ZIP_PATH = "/content/drive/MyDrive/YOLO/datasets/9050_3_Mdataset.zip"
LOCAL_DATASET_PATH = str(REPO_ROOT / DATASET_NAME)
DATASET_SEARCH_ROOTS = (
    LOCAL_DATASET_PATH,
    str(REPO_ROOT / "TIB_dataset"),
    str(REPO_ROOT.parent / DATASET_NAME),
    "/content/dataset",
    "/content/9050_3_Mdataset",
)
DATASET_ZIP_FALLBACKS = (
    DRIVE_ZIP_PATH,
    "/content/drive/MyDrive/9050_3_Mdataset.zip",
    "/content/drive/MyDrive/datasets/9050_3_Mdataset.zip",
    "/content/drive/MyDrive/YOLO/9050_3_Mdataset.zip",
    str(REPO_ROOT / "9050_3_Mdataset.zip"),
    str(REPO_ROOT.parent / "9050_3_Mdataset.zip"),
    str(REPO_ROOT.parent / "dataset.zip"),
)
DATA_YAML_NAME = "data.yaml"
REQUIRE_TEST_SPLIT = False

# ─── Cikti yollari ───────────────────────────────────────────────────────────
DRIVE_OUTPUT_PATH = "/content/drive/MyDrive/yolo_runs"
DRIVE_MODEL_DIR = "/content/drive/MyDrive/ultralytics_models"
LOCAL_MODEL_DIR = "/content/ultralytics_models"

# ─── Model ───────────────────────────────────────────────────────────────────
# Resmi Ultralytics model adini ver. Ilk calismada indirilir, Drive'a kaydedilir,
# sonraki calismalarda ayni dosya tekrar kullanilir.
MODEL = "yolo26s.pt"
AUTO_DOWNLOAD_MODEL = True

# ─── Temel egitim parametreleri ──────────────────────────────────────────────
EPOCHS = 100
IMGSZ = 640
BATCH = 64          # L4 GPU (~22 GB) icin 640px batch-64 guvenli
WORKERS = 8
CACHE = "ram"       # 9050 gorseli RAM'e yukle, epoch basi disk I/O sifir
DEVICE = "0"
SAVE_PERIOD = -1    # Her epoch checkpoint alma; sadece best.pt + last.pt kalir
PROJECT = DRIVE_OUTPUT_PATH
RUN_NAME = "yolo26s_9050_v1"
RESUME = False
SINGLE_CLS = False  # Dataset zaten tek sinif (nc=1, names=[drone]); override gereksiz

# ─── Optimizer ───────────────────────────────────────────────────────────────
# optimizer="auto" kullanilirsa Ultralytics lr0/momentum degerlerini yok sayar.
# Onceki basarili kosuda sistem AdamW(lr=0.002, momentum=0.9) secmisti;
# ayni degerleri burada acikca sabitleriz.
OPTIMIZER = "AdamW"
LR0 = 0.002
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
WARMUP_EPOCHS = 3.0
WARMUP_MOMENTUM = 0.8
WARMUP_BIAS_LR = 0.1

# ─── Dogrulama / kayit / raporlama ───────────────────────────────────────────
PATIENCE = 20       # Early stopping: 20 epoch iyilesme olmazsa dur
AMP = True
PRETRAINED = True
VAL = True
PLOTS = True        # Egitim sonu loss/mAP/precision/recall grafikleri kaydedilir

# ─── Augmentation ────────────────────────────────────────────────────────────
CLOSE_MOSAIC = 10   # Son 10 epoch'ta mozaigi kapat, fine-tune stabilizasyonu
MOSAIC = 1.0
MIXUP = 0.0
CUTMIX = 0.0
TRANSLATE = 0.1
SCALE = 0.5
FLIPLR = 0.5
HSV_H = 0.015
HSV_S = 0.7
HSV_V = 0.4
RECT = False
MULTI_SCALE = 0.0
