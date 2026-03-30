import os
import shutil
import zipfile
from pathlib import Path

from ultralytics import YOLO
import yaml

import config as CFG

try:
    from google.colab import drive
except ImportError:
    drive = None

IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


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


def expand_path(raw_path: str | Path) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(str(raw_path))))


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


def find_dataset_root(search_root: Path) -> Path | None:
    yaml_name = getattr(CFG, "DATA_YAML_NAME", "data.yaml")
    search_root = expand_path(search_root)
    if not search_root.exists():
        return None
    if search_root.is_file():
        return search_root.parent if search_root.name == yaml_name else None

    candidates = [search_root]
    frontier = [search_root]
    for _ in range(2):
        next_frontier: list[Path] = []
        for directory in frontier:
            try:
                child_dirs = [child for child in directory.iterdir() if child.is_dir()]
            except OSError:
                continue
            candidates.extend(child_dirs)
            next_frontier.extend(child_dirs)
        frontier = next_frontier

    for candidate in unique_paths(candidates):
        if (candidate / yaml_name).exists():
            return candidate
    return None


def resolve_dataset_search_roots() -> list[Path]:
    repo_root = Path(__file__).resolve().parent
    configured = [expand_path(CFG.LOCAL_DATASET_PATH)]
    configured.extend(
        expand_path(path) for path in getattr(CFG, "DATASET_SEARCH_ROOTS", ())
    )
    # Colab'da repo agacini tarama: Drive'daki zip birincil kaynak.
    # Repo agaci taramasi yalnizca lokal gelistirme icin kullanilir.
    if not in_colab():
        configured.extend([repo_root, repo_root.parent])
    return unique_paths(configured)


def find_existing_dataset_root() -> Path | None:
    for search_root in resolve_dataset_search_roots():
        dataset_root = find_dataset_root(search_root)
        if dataset_root is not None:
            return dataset_root
    return None


def resolve_zip_candidates() -> list[Path]:
    repo_root = Path(__file__).resolve().parent
    dataset_name = getattr(CFG, "DATASET_NAME", "dataset")
    configured = [expand_path(CFG.DRIVE_ZIP_PATH)]
    configured.extend(
        expand_path(path) for path in getattr(CFG, "DATASET_ZIP_FALLBACKS", ())
    )
    configured.extend([
        repo_root / f"{dataset_name}.zip",
        repo_root.parent / f"{dataset_name}.zip",
        repo_root.parent / "dataset.zip",
    ])
    return unique_paths(configured)


def extract_dataset() -> Path:
    existing_root = find_existing_dataset_root()
    if existing_root is not None:
        return existing_root

    dataset_root = expand_path(CFG.LOCAL_DATASET_PATH)
    zip_candidates = resolve_zip_candidates()
    zip_path = next((path for path in zip_candidates if path.exists()), None)
    if zip_path is None:
        checked = "\n".join(str(path) for path in zip_candidates)
        raise FileNotFoundError(f"Dataset zip bulunamadi. Kontrol edilen yollar:\n{checked}")

    dataset_root.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(dataset_root.parent)
    extracted_root = find_dataset_root(dataset_root) or find_dataset_root(dataset_root.parent)
    if extracted_root is None:
        raise FileNotFoundError(
            "Zip acildi ama data.yaml bulunamadi. Beklenen konumlar:\n"
            f"{dataset_root / getattr(CFG, 'DATA_YAML_NAME', 'data.yaml')}\n"
            f"{dataset_root.parent / getattr(CFG, 'DATA_YAML_NAME', 'data.yaml')}"
        )
    return extracted_root


def resolve_dataset_base(raw_cfg: dict, dataset_root: Path) -> Path:
    configured_base = raw_cfg.get("path")
    if not configured_base:
        return dataset_root.resolve()

    # Windows mutlak yolu (orn. "C:/...") Linux'ta is_absolute()=False doner
    # ve dataset_root ile birlestirilince bozuk yol olusur. Bu durumda
    # dataset_root'u kullan.
    import re
    if re.match(r"^[A-Za-z]:[/\\]", str(configured_base)):
        return dataset_root.resolve()

    base_path = expand_path(configured_base)
    if not base_path.is_absolute():
        base_path = (dataset_root / base_path).resolve()
    if not base_path.exists():
        raise FileNotFoundError(f"Dataset path bulunamadi: {base_path}")
    return base_path


def resolve_names_and_nc(raw_cfg: dict) -> tuple[int, list | dict]:
    names = raw_cfg.get("names")
    nc = raw_cfg.get("nc")

    if isinstance(names, list):
        nc = len(names) if nc is None else int(nc)
        return nc, names

    if isinstance(names, dict):
        nc = len(names) if nc is None else int(nc)
        return nc, names

    if nc is not None:
        nc = int(nc)
        return nc, {index: f"class_{index}" for index in range(nc)}

    return 1, {0: "object"}


def normalize_split_entry(split_name: str, split_value, dataset_base: Path):
    if isinstance(split_value, list):
        return [
            normalize_split_entry(split_name, item, dataset_base)
            for item in split_value
        ]

    if not isinstance(split_value, str) or not split_value.strip():
        raise ValueError(f"'{split_name}' alani bos veya gecersiz: {split_value!r}")

    raw_value = split_value.strip()
    split_path = expand_path(raw_value)
    if not split_path.is_absolute():
        split_path = (dataset_base / raw_value).resolve()
    else:
        split_path = split_path.resolve()

    if not split_path.exists():
        raise FileNotFoundError(f"Dataset yolu bulunamadi ({split_name}): {split_path}")

    try:
        return split_path.relative_to(dataset_base.resolve()).as_posix()
    except ValueError:
        return str(split_path)


def iter_split_entries(split_value):
    if isinstance(split_value, list):
        for item in split_value:
            yield from iter_split_entries(item)
        return
    yield split_value


def count_images(split_value, dataset_base: Path) -> int:
    total = 0
    for entry in iter_split_entries(split_value):
        split_path = Path(entry)
        if not split_path.is_absolute():
            split_path = dataset_base / split_path

        if split_path.is_file():
            if split_path.suffix.lower() in IMAGE_EXTENSIONS:
                total += 1
            continue

        for file_path in split_path.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS:
                total += 1
    return total


def format_class_names(names: list | dict) -> str:
    if isinstance(names, list):
        return ", ".join(str(name) for name in names)

    def sort_key(item):
        key = str(item[0])
        return (0, int(key)) if key.isdigit() else (1, key)

    return ", ".join(str(value) for _, value in sorted(names.items(), key=sort_key))


def resolve_data_yaml(dataset_root: Path) -> tuple[str, dict]:
    data_yaml = dataset_root / getattr(CFG, "DATA_YAML_NAME", "data.yaml")
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml bulunamadi: {data_yaml}")

    with open(data_yaml, "r", encoding="utf-8") as handle:
        raw_cfg = yaml.safe_load(handle) or {}

    if "train" not in raw_cfg or "val" not in raw_cfg:
        raise KeyError("data.yaml icinde en az 'train' ve 'val' alanlari olmali.")

    dataset_base = resolve_dataset_base(raw_cfg, dataset_root)
    nc, names = resolve_names_and_nc(raw_cfg)

    normalized_cfg = dict(raw_cfg)
    normalized_cfg["path"] = str(dataset_base)
    normalized_cfg["train"] = normalize_split_entry("train", raw_cfg["train"], dataset_base)
    normalized_cfg["val"] = normalize_split_entry("val", raw_cfg["val"], dataset_base)
    normalized_cfg["nc"] = nc
    normalized_cfg["names"] = names

    if "test" in raw_cfg and raw_cfg["test"] not in (None, ""):
        normalized_cfg["test"] = normalize_split_entry("test", raw_cfg["test"], dataset_base)
    elif getattr(CFG, "REQUIRE_TEST_SPLIT", False):
        raise FileNotFoundError("Dataset test split icermiyor ama REQUIRE_TEST_SPLIT=True.")
    else:
        normalized_cfg.pop("test", None)

    runtime_yaml = dataset_root / "data.runtime.yaml"
    with open(runtime_yaml, "w", encoding="utf-8") as handle:
        yaml.safe_dump(normalized_cfg, handle, sort_keys=False, allow_unicode=True)

    split_counts = {
        split: count_images(normalized_cfg[split], dataset_base)
        for split in ("train", "val", "test")
        if split in normalized_cfg
    }
    dataset_meta = {
        "dataset_base": dataset_base,
        "nc": nc,
        "names": names,
        "class_names": format_class_names(names),
        "split_counts": split_counts,
        "total_images": sum(split_counts.values()),
    }
    return str(runtime_yaml), dataset_meta


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
    data_yaml, dataset_meta = resolve_data_yaml(dataset_root)
    model_path = resolve_model_path()
    os.makedirs(CFG.PROJECT, exist_ok=True)

    print(f"Dataset : {dataset_root}")
    print(f"Data dir : {dataset_meta['dataset_base']}")
    print(f"data.yaml: {data_yaml}")
    print(f"Classes : {dataset_meta['nc']} -> {dataset_meta['class_names']}")
    print(f"Images  : {dataset_meta['split_counts']} (toplam={dataset_meta['total_images']})")
    print(f"Model   : {model_path}")
    print(f"Run     : {CFG.RUN_NAME}")

    model = YOLO(model_path)
    single_cls = bool(getattr(CFG, "SINGLE_CLS", False))
    if single_cls and dataset_meta["nc"] > 1:
        print("Uyari   : SINGLE_CLS=True oldugu icin tum siniflar tek sinifa indirgenecek.")

    model.train(
        data=data_yaml,
        task="detect",
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
        single_cls=single_cls,
        exist_ok=True,
        # ─── Optimizer ───────────────────────────────────────────────────────
        optimizer=CFG.OPTIMIZER,
        lr0=CFG.LR0,
        momentum=CFG.MOMENTUM,
        weight_decay=CFG.WEIGHT_DECAY,
        warmup_epochs=CFG.WARMUP_EPOCHS,
        warmup_momentum=CFG.WARMUP_MOMENTUM,
        warmup_bias_lr=CFG.WARMUP_BIAS_LR,
        # ─── Dogrulama / kayit / raporlama ───────────────────────────────────
        patience=CFG.PATIENCE,
        amp=CFG.AMP,
        pretrained=CFG.PRETRAINED,
        val=CFG.VAL,
        plots=CFG.PLOTS,
        # ─── Augmentation ────────────────────────────────────────────────────
        close_mosaic=CFG.CLOSE_MOSAIC,
        mosaic=CFG.MOSAIC,
        mixup=CFG.MIXUP,
        cutmix=CFG.CUTMIX,
        translate=CFG.TRANSLATE,
        scale=CFG.SCALE,
        fliplr=CFG.FLIPLR,
        hsv_h=CFG.HSV_H,
        hsv_s=CFG.HSV_S,
        hsv_v=CFG.HSV_V,
        rect=CFG.RECT,
        multi_scale=CFG.MULTI_SCALE,
    )


if __name__ == "__main__":
    main()
