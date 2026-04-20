import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from src.config import Config
from src.utils import list_images, check_dir_exists


@dataclass(frozen=True)
class SampleRecord:
    image_path: Path
    mask_path: Optional[Path]
    label: int
    defect_type: str
    split: str
    category: str


def _infer_defect_type_from_name(image_path: Path) -> str:
    """
    Infer the defect type from the parent folder of the image.
    For good images, return 'good'.
    """
    parent = image_path.parent.name.lower()
    return parent


def _make_mask_path(image_path: Path, gt_root: Path) -> Optional[Path]:
    """
    Map test defect image path to ground truth mask path.
    MVTec AD usually uses same filename with `_mask.png`.
    """
    defect_type = image_path.parent.name
    stem = image_path.stem

    candidates = [
        gt_root / defect_type / f"{stem}_mask.png",
        gt_root / defect_type / f"{stem}.png",
        gt_root / defect_type / f"{stem}_mask.bmp",
    ]

    for p in candidates:
        if p.exists():
            return p

    return None


def build_records(config: Config, split: str) -> List[SampleRecord]:
    """
    Build records for a given split.

    split:
        - 'train' -> only train/good
        - 'test'  -> test/good + all defect folders
    """
    category_root = config.category_root
    records: List[SampleRecord] = []

    if split == "train":
        train_good = category_root / "train" / "good"
        check_dir_exists(train_good)

        for img_path in list_images(train_good):
            records.append(
                SampleRecord(
                    image_path=img_path,
                    mask_path=None,
                    label=0,
                    defect_type="good",
                    split="train",
                    category=config.category,
                )
            )

    elif split == "test":
        test_root = category_root / "test"
        gt_root = category_root / "ground_truth"
        check_dir_exists(test_root)
        check_dir_exists(gt_root)

        for defect_folder in sorted([p for p in test_root.iterdir() if p.is_dir()]):
            defect_type = defect_folder.name.lower()
            for img_path in list_images(defect_folder):
                is_good = defect_type == "good"
                mask_path = None if is_good else _make_mask_path(img_path, gt_root)

                records.append(
                    SampleRecord(
                        image_path=img_path,
                        mask_path=mask_path,
                        label=0 if is_good else 1,
                        defect_type=defect_type,
                        split="test",
                        category=config.category,
                    )
                )
    else:
        raise ValueError("split must be either 'train' or 'test'.")

    return records


def records_to_dataframe(records: List[SampleRecord]) -> pd.DataFrame:
    """Convert list of records into a dataframe for inspection / logging."""
    rows: List[Dict[str, Any]] = []
    for r in records:
        rows.append(
            {
                "image_path": str(r.image_path),
                "mask_path": str(r.mask_path) if r.mask_path is not None else None,
                "label": r.label,
                "defect_type": r.defect_type,
                "split": r.split,
                "category": r.category,
            }
        )
    return pd.DataFrame(rows)


class MVTecBottleDataset(Dataset):
    """
    Dataset wrapper that returns:
        image_path, image, mask, label, defect_type
    """

    def __init__(self, config: Config, split: str):
        self.config = config
        self.split = split
        self.records = build_records(config, split)

        if len(self.records) == 0:
            raise RuntimeError(f"No samples found for split='{split}' in category='{config.category}'.")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        record = self.records[idx]

        image = Image.open(record.image_path).convert("RGB")

        mask = None
        if record.mask_path is not None and record.mask_path.exists():
            mask = Image.open(record.mask_path).convert("L")

        return {
            "image_path": str(record.image_path),
            "mask_path": str(record.mask_path) if record.mask_path is not None else None,
            "image": image,
            "mask": mask,
            "label": record.label,
            "defect_type": record.defect_type,
            "split": record.split,
            "category": record.category,
        }


def get_train_dataframe(config: Config) -> pd.DataFrame:
    records = build_records(config, "train")
    return records_to_dataframe(records)


def get_test_dataframe(config: Config) -> pd.DataFrame:
    records = build_records(config, "test")
    return records_to_dataframe(records)


def summarize_dataset(config: Config) -> Dict[str, Any]:
    """Return simple counts useful for debugging and logging."""
    train_df = get_train_dataframe(config)
    test_df = get_test_dataframe(config)

    summary = {
        "category": config.category,
        "train_total": int(len(train_df)),
        "test_total": int(len(test_df)),
        "test_good": int((test_df["defect_type"] == "good").sum()),
        "test_defective": int((test_df["label"] == 1).sum()),
        "train_good": int((train_df["label"] == 0).sum()),
        "defect_types": sorted([x for x in test_df["defect_type"].unique().tolist() if x != "good"]),
    }
    return summary


def verify_masks(config: Config) -> List[str]:
    """
    Check whether all defective test images have masks.
    Returns a list of missing mask paths for debugging.
    """
    records = build_records(config, "test")
    missing: List[str] = []
    for r in records:
        if r.label == 1 and r.mask_path is None:
            missing.append(str(r.image_path))
    return missing
