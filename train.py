import os
import yaml
from google.colab import drive
from ultralytics import YOLO

import config as CFG


def mount_drive():
    drive.mount("/content/drive")


def build_yaml(dataset_path, output_path="/content/data.yaml"):
    cfg = {
        "path":  dataset_path,
        "train": "images/train",
        "val":   "images/val",
        "test":  "images/test",
        "nc":    1,
        "names": {0: "UAV"},
    }
    with open(output_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    return output_path


def main():
    mount_drive()

    os.makedirs(CFG.PROJECT, exist_ok=True)

    data_yaml = build_yaml(CFG.DRIVE_DATASET_PATH)

    model = YOLO(CFG.MODEL)

    model.train(
        data        = data_yaml,
        epochs      = CFG.EPOCHS,
        imgsz       = CFG.IMGSZ,
        batch       = CFG.BATCH,
        workers     = CFG.WORKERS,
        cache       = CFG.CACHE,
        device      = CFG.DEVICE,
        project     = CFG.PROJECT,
        name        = CFG.RUN_NAME,
        save_period = CFG.SAVE_PERIOD,
        resume      = CFG.RESUME,
        task        = "detect",
        single_cls  = True,
        exist_ok    = True,
    )


if __name__ == "__main__":
    main()
