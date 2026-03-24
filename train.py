import os
import shutil
import zipfile
from pathlib import Path

from ultralytics import YOLO

import config as CFG

try:
    from google.colab import drive
except ImportError:
    drive = None


def in_colab() -> bool:
    return drive is not None


def drive_is_mounted() -> bool:
    return Path("/content/drive/MyDrive").exists()


def can_mount_drive_in_this_process() -> bool:
    if not in_colab():
        return False
    try:
        from IPython import get_ipython
        ip = get_ipython()
    except Exception:
        return False
    return bool(ip and getattr(ip, "kernel", None))


def mount_drive_if_needed() -> None:
    if not in_colab():
        return
    if drive_is_mounted():
        return
    if not can_mount_drive_in_this_process():
        raise RuntimeError(
            "Google Drive bu '!python train.py' subprocess'i icinden mount edilemez.\n"
            "Colab'da once su komutu calistirin:\n"
            "from google.colab import drive; drive.mount('/content/drive')\n"
            "Ardindan tekrar '!python train.py' calistirin."
        )
    drive.mount("/content/drive")
    if not drive_is_mounted():
        raise RuntimeError("Google Drive mount tamamlanamadi.")


def find_dataset_root(search_root: Path) -> Path | None:
    direct_yaml = search_root / "data.yaml"
    if direct_yaml.exists():
        return search_root
    tib_yaml = search_root / "TIB_dataset" / "data.yaml"
    if tib_yaml.exists():
        return tib_yaml.parent
    return None


def resolve_zip_candidates() -> list[Path]:
    repo_root = Path(__file__).resolve().parent
    configured = [Path(CFG.DRIVE_ZIP_PATH)]
    configured.extend(Path(path) for path in getattr(CFG, "DATASET_ZIP_FALLBACKS", ()))
    configured.extend([
        repo_root / "TIB_dataset.zip",
        repo_root.parent / "TIB_dataset.zip",
    ])

    seen: set[str] = set()
    result: list[Path] = []
    for path in configured:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        result.append(path)
    return result


def extract_dataset() -> Path:
    dataset_root = Path(CFG.LOCAL_DATASET_PATH)
    existing_root = find_dataset_root(dataset_root)
    if existing_root is not None:
        return existing_root

    zip_candidates = resolve_zip_candidates()
    zip_path = next((path for path in zip_candidates if path.exists()), None)
    if zip_path is None:
        checked = "\n".join(str(path) for path in zip_candidates)
        raise FileNotFoundError(f"Dataset zip bulunamadi. Kontrol edilen yollar:\n{checked}")

    dataset_root.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(dataset_root.parent)
    extracted_root = find_dataset_root(dataset_root.parent)
    if extracted_root is None:
        raise FileNotFoundError(
            "Zip acildi ama data.yaml bulunamadi. Beklenen konumlar:\n"
            f"{dataset_root / 'data.yaml'}\n"
            f"{dataset_root.parent / 'data.yaml'}"
        )
    return extracted_root


def resolve_data_yaml(dataset_root: Path) -> str:
    data_yaml = dataset_root / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml bulunamadi: {data_yaml}")
    return str(data_yaml)


def unique_paths(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    unique: list[Path] = []
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def find_existing_model_paths(model_name: str) -> list[Path]:
    repo_root = Path(__file__).resolve().parent
    workspace_models = repo_root.parent / "models" / "pretrained"
    candidates = [
        Path(CFG.MODEL),
        Path.cwd() / model_name,
        repo_root / model_name,
        Path(CFG.LOCAL_MODEL_DIR) / model_name,
        Path(CFG.DRIVE_MODEL_DIR) / model_name,
        Path("/content/drive/MyDrive") / model_name,
        Path("/content/drive/MyDrive/models") / model_name,
    ]
    if workspace_models.exists():
        candidates.extend(workspace_models.rglob(model_name))
    return [path for path in unique_paths(candidates) if path.exists()]


def find_downloaded_model_path(model: YOLO, model_name: str) -> Path | None:
    for attr_name in ("ckpt_path", "model_name"):
        value = getattr(model, attr_name, None)
        if value:
            path = Path(str(value))
            if path.exists():
                return path

    search_roots = [
        Path.cwd(),
        Path(CFG.LOCAL_MODEL_DIR),
        Path("/content"),
        Path("/root/.cache/ultralytics"),
        Path.home() / ".cache" / "ultralytics",
    ]
    for root in search_roots:
        if not root.exists():
            continue
        direct = root / model_name
        if direct.exists():
            return direct
        matches = list(root.rglob(model_name))
        if matches:
            return matches[0]
    return None


def cache_model_in_drive(source_path: Path, model_name: str) -> Path:
    drive_dir = Path(CFG.DRIVE_MODEL_DIR)
    drive_dir.mkdir(parents=True, exist_ok=True)
    drive_path = drive_dir / model_name
    if not drive_path.exists():
        shutil.copy2(source_path, drive_path)
    return drive_path


def cache_model_locally(source_path: Path, model_name: str) -> Path:
    local_dir = Path(CFG.LOCAL_MODEL_DIR)
    local_dir.mkdir(parents=True, exist_ok=True)
    local_path = local_dir / model_name
    if not local_path.exists():
        shutil.copy2(source_path, local_path)
    return local_path


def resolve_model_path() -> str:
    model_name = Path(CFG.MODEL).name

    existing = find_existing_model_paths(model_name)
    if existing:
        source_path = existing[0]
        if in_colab():
            drive_path = cache_model_in_drive(source_path, model_name)
            return str(cache_model_locally(drive_path, model_name))
        return str(source_path)

    if not CFG.AUTO_DOWNLOAD_MODEL:
        raise FileNotFoundError(f"Pretrained model bulunamadi: {model_name}")

    print(f"Model indiriliyor: {CFG.MODEL}")
    downloaded_model = YOLO(CFG.MODEL)
    downloaded_path = find_downloaded_model_path(downloaded_model, model_name)
    if downloaded_path is None:
        raise FileNotFoundError(f"Model indirildi ama dosya yolu bulunamadi: {model_name}")

    if in_colab():
        drive_path = cache_model_in_drive(downloaded_path, model_name)
        return str(cache_model_locally(drive_path, model_name))
    return str(downloaded_path)


def main():
    mount_drive_if_needed()
    dataset_root = extract_dataset()
    data_yaml = resolve_data_yaml(dataset_root)
    model_path = resolve_model_path()
    os.makedirs(CFG.PROJECT, exist_ok=True)

    print(f"Dataset : {dataset_root}")
    print(f"data.yaml: {data_yaml}")
    print(f"Model   : {model_path}")
    print(f"Run     : {CFG.RUN_NAME}")

    model = YOLO(model_path)

    model.train(
        data=data_yaml,
        epochs=CFG.EPOCHS,
        imgsz=CFG.IMGSZ,
        batch=CFG.BATCH,
        workers=CFG.WORKERS,
        cache=CFG.CACHE,
        device=CFG.DEVICE,
        project=CFG.PROJECT,
        name=CFG.RUN_NAME,
        save_period=CFG.SAVE_PERIOD,
        resume=CFG.RESUME,
        task="detect",
        single_cls=True,
        exist_ok=True,
    )


if __name__ == "__main__":
    main()
