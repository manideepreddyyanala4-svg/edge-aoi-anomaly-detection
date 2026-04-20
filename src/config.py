from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


@dataclass(frozen=True)
class Config:
    seed: int = 42

    data_root: Path = Path("data/mvtec")
    category: str = "bottle"
    test_defect_types: Tuple[str, ...] = ("broken_large", "broken_small", "contamination")

    img_size: int = 224
    resize_size: int = 256
    imagenet_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    imagenet_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)

    backbone: str = "wide_resnet50_2"
    use_pretrained: bool = True

    faiss_k: int = 1
    coreset_ratio: float = 0.10
    use_coreset: bool = True

    layer2_weight: float = 1.0

    heatmap_sigma: float = 0.3

    output_root: Path = Path("outputs")
    memory_bank_dir: Path = Path("outputs/memory_banks")
    results_dir: Path = Path("outputs/results")
    figures_dir: Path = Path("outputs/results/figures")
    evaluation_dir: Path = Path("outputs/evaluation")
    benchmarks_dir: Path = Path("outputs/benchmarks")

    @property
    def category_root(self) -> Path:
        return self.data_root / self.category

    @property
    def baseline_bank_path(self) -> Path:
        return self.memory_bank_dir / f"{self.category}_full_bank.pkl"

    @property
    def coreset_bank_path(self) -> Path:
        return self.memory_bank_dir / f"{self.category}_coreset_bank.pkl"

    @property
    def threshold_path(self) -> Path:
        return self.results_dir / f"{self.category}_threshold.txt"

    @property
    def metrics_path(self) -> Path:
        return self.results_dir / f"{self.category}_metrics.json"

    def ensure_dirs(self):
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.memory_bank_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.evaluation_dir.mkdir(parents=True, exist_ok=True)
        self.benchmarks_dir.mkdir(parents=True, exist_ok=True)
