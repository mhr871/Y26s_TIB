from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

DRIVE_ZIP_PATH = "/content/drive/MyDrive/YOLO/datasets/9050_3_Mdataset.zip"
LOCAL_DATASET_PATH = "/content/9050_3_Mdataset"
DRIVE_OUTPUT_PATH = "/content/drive/MyDrive/yolo_runs"
DRIVE_MODEL_DIR = "/content/drive/MyDrive/ultralytics_models"
LOCAL_MODEL_DIR = "/content/ultralytics_models"
DATASET_ZIP_FALLBACKS = (
    "/content/drive/MyDrive/9050_3_Mdataset.zip",
    "/content/drive/MyDrive/datasets/9050_3_Mdataset.zip",
    "/content/drive/MyDrive/YOLO/9050_3_Mdataset.zip",
)

# Resmi Ultralytics model adini ver. Ilk calismada indirilir, Drive'a kaydedilir,
# sonraki calismalarda ayni dosya tekrar kullanilir.
MODEL = "yolo26s.pt"
AUTO_DOWNLOAD_MODEL = True

EPOCHS = 100
IMGSZ = 640
BATCH = 16
WORKERS = 4
CACHE = False
DEVICE = "0"
SAVE_PERIOD = 1
PROJECT = DRIVE_OUTPUT_PATH
RUN_NAME = "yolo26s_UAV_v1"
RESUME = False
