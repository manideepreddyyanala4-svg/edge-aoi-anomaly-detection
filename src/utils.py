import time
import random
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Timer:
    def __init__(self):
        self.start_time = None

    def start(self):
        self.start_time = time.time()

    def stop(self) -> float:
        if self.start_time is None:
            raise RuntimeError("Timer was not started.")
        return (time.time() - self.start_time) * 1000.0


def save_json(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_numpy(array: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)


def load_numpy(path: Path) -> np.ndarray:
    return np.load(path)


def check_dir_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Directory not found: {path}")


def list_images(folder: Path):
    exts = [".png", ".jpg", ".jpeg", ".bmp", ".PNG", ".JPG", ".JPEG", ".BMP"]
    files = []
    for ext in exts:
        files.extend(folder.glob(f"*{ext}"))
    return sorted(files)


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def normalize_features(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(x, dim=1)


def log(message: str) -> None:
    print(f"[INFO] {message}")
