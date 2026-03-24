DRIVE_ZIP_PATH     = "/content/drive/MyDrive/TIB_dataset.zip"
LOCAL_DATASET_PATH = "/content/TIB_dataset"
DRIVE_OUTPUT_PATH  = "/content/drive/MyDrive/yolo_runs"

MODEL      = "yolov8s.pt"
EPOCHS     = 100
IMGSZ      = 640
BATCH      = 16
WORKERS    = 4
CACHE      = False
DEVICE     = "0"
SAVE_PERIOD = 1
PROJECT    = DRIVE_OUTPUT_PATH
RUN_NAME   = "yolo26s_UAV_v1"
RESUME     = False
