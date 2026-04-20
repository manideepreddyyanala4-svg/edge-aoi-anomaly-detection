import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.config import Config
from src.data_loader import MVTecBottleDataset
from src.preprocess import preprocess_batch
from src.models import FeatureExtractor, fuse_patch_embeddings
from src.utils import set_seed, log, save_json


@dataclass
class MemoryBankInfo:
    category: str
    bank_type: str
    vector_dim: int
    num_vectors: int
    grid_h: int
    grid_w: int
    memory_mb: float
    memory_path: str


def _resolve_device(device: Optional[torch.device] = None) -> torch.device:
    """
    Prefer CUDA if it is actually usable.
    """
    if device is not None:
        return device

    if torch.cuda.is_available():
        try:
            _ = torch.zeros(1, device="cuda")
            torch.cuda.synchronize()
            log("CUDA test succeeded, using GPU.")
            return torch.device("cuda")
        except Exception as e:
            log(f"CUDA detected but unusable, falling back to CPU. Reason: {e}")

    log("Using CPU.")
    return torch.device("cpu")


def _make_loader(config: Config, split: str, batch_size: int = 8) -> DataLoader:
    dataset = MVTecBottleDataset(config, split=split)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: x)


def _extract_vectors_from_batch(
    batch: List[Dict[str, Any]],
    extractor: FeatureExtractor,
    config: Config,
    device: torch.device,
) -> Tuple[torch.Tensor, Tuple[int, int], List[str]]:
    """
    Preprocess a batch of PIL images and return patch vectors.
    The forward pass runs on GPU if CUDA is available.
    """
    images = tuple(sample["image"] for sample in batch)
    image_paths = [sample["image_path"] for sample in batch]

    x = preprocess_batch(images, config).to(device, non_blocking=True)

    with torch.inference_mode():
        fused_map = extractor.extract_fused_feature_map(x)

    if fused_map.device != x.device:
        raise RuntimeError(f"Device mismatch: input on {x.device}, fused_map on {fused_map.device}")

    grid_h, grid_w = fused_map.shape[-2], fused_map.shape[-1]
    patch_vectors = fuse_patch_embeddings(fused_map)

    patch_vectors = patch_vectors.detach().float().cpu()

    repeated_paths: List[str] = []
    for p in image_paths:
        repeated_paths.extend([p] * (grid_h * grid_w))

    return patch_vectors, (grid_h, grid_w), repeated_paths


def collect_training_patch_vectors(
    config: Config,
    batch_size: int = 8,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, List[str], Tuple[int, int]]:
    """
    Collect all patch vectors from train/good only.
    Returns:
        vectors: [N, D] CPU tensor
        source_paths: list length N
        grid_size: (H, W)
    """
    device = _resolve_device(device)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    extractor = FeatureExtractor(config).to(device)
    extractor.eval()

    loader = _make_loader(config, split="train", batch_size=batch_size)

    all_vectors: List[torch.Tensor] = []
    all_sources: List[str] = []
    grid_size: Optional[Tuple[int, int]] = None

    for batch in loader:
        vectors, gsize, sources = _extract_vectors_from_batch(batch, extractor, config, device)
        all_vectors.append(vectors)
        all_sources.extend(sources)
        if grid_size is None:
            grid_size = gsize

    if grid_size is None:
        raise RuntimeError("No training vectors were collected.")

    vectors = torch.cat(all_vectors, dim=0).contiguous().float()
    return vectors, all_sources, grid_size


def _memory_size_mb_torch(vectors: torch.Tensor) -> float:
    return float(vectors.numel() * vectors.element_size() / (1024.0 * 1024.0))


def greedy_furthest_point_sampling_torch(
    vectors: torch.Tensor,
    target_size: int,
    seed: int = 42,
) -> torch.Tensor:
    """
    Greedy farthest-point sampling in PyTorch.
    Runs on the same device as `vectors`.
    """
    if target_size <= 0:
        raise ValueError("target_size must be positive.")
    if vectors.ndim != 2:
        raise ValueError("vectors must be 2D.")

    n = vectors.shape[0]
    if target_size >= n:
        return torch.arange(n, device=vectors.device, dtype=torch.long)

    g = torch.Generator(device=vectors.device)
    g.manual_seed(seed)

    first_idx = int(torch.randint(0, n, (1,), generator=g, device=vectors.device).item())
    selected_mask = torch.zeros(n, dtype=torch.bool, device=vectors.device)
    selected_mask[first_idx] = True

    selected = [first_idx]

    ref = vectors[first_idx]
    min_dists = torch.sum((vectors - ref) ** 2, dim=1)
    min_dists[selected_mask] = -1.0

    for _ in range(1, target_size):
        next_idx = int(torch.argmax(min_dists).item())
        selected.append(next_idx)
        selected_mask[next_idx] = True

        ref = vectors[next_idx]
        dists = torch.sum((vectors - ref) ** 2, dim=1)
        min_dists = torch.minimum(min_dists, dists)
        min_dists[selected_mask] = -1.0

    return torch.tensor(selected, device=vectors.device, dtype=torch.long)


def build_and_save_memory_banks(
    config: Config,
    batch_size: int = 8,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Build both baseline and coreset memory banks and save them to disk.
    Everything stays in PyTorch; no FAISS is used.
    """
    config.ensure_dirs()
    set_seed(config.seed)

    device = _resolve_device(device)
    log(f"Using device: {device}")

    log(f"Collecting training patch vectors for category='{config.category}'...")
    vectors_cpu, sources, grid_size = collect_training_patch_vectors(
        config=config,
        batch_size=batch_size,
        device=device,
    )

    num_vectors, vector_dim = vectors_cpu.shape
    grid_h, grid_w = grid_size

    baseline_bank_path = config.memory_bank_dir / f"{config.category}_full_bank.pt"
    coreset_bank_path = config.memory_bank_dir / f"{config.category}_coreset_bank.pt"

    log(f"Saving baseline memory bank with {num_vectors} vectors...")
    baseline_meta = MemoryBankInfo(
        category=config.category,
        bank_type="baseline",
        vector_dim=vector_dim,
        num_vectors=num_vectors,
        grid_h=grid_h,
        grid_w=grid_w,
        memory_mb=_memory_size_mb_torch(vectors_cpu),
        memory_path=str(baseline_bank_path),
    )

    baseline_payload = {
        "info": asdict(baseline_meta),
        "vectors": vectors_cpu.contiguous(),
        "source_paths": sources,
    }
    torch.save(baseline_payload, baseline_bank_path)

    target_coreset_size = max(1, int(math.ceil(num_vectors * config.coreset_ratio)))
    log(f"Selecting coreset with target size={target_coreset_size} from {num_vectors} vectors...")

    vectors_device = vectors_cpu.to(device, non_blocking=True)

    selected_indices = greedy_furthest_point_sampling_torch(
        vectors=vectors_device,
        target_size=target_coreset_size,
        seed=config.seed,
    )

    coreset_vectors = vectors_cpu[selected_indices.detach().cpu()]
    coreset_sources = [sources[i] for i in selected_indices.detach().cpu().tolist()]

    coreset_meta = MemoryBankInfo(
        category=config.category,
        bank_type="coreset",
        vector_dim=vector_dim,
        num_vectors=int(coreset_vectors.shape[0]),
        grid_h=grid_h,
        grid_w=grid_w,
        memory_mb=_memory_size_mb_torch(coreset_vectors),
        memory_path=str(coreset_bank_path),
    )

    coreset_payload = {
        "info": asdict(coreset_meta),
        "vectors": coreset_vectors.contiguous(),
        "source_paths": coreset_sources,
    }
    torch.save(coreset_payload, coreset_bank_path)

    summary = {
        "category": config.category,
        "vector_dim": vector_dim,
        "grid_h": grid_h,
        "grid_w": grid_w,
        "device": str(device),
        "baseline": baseline_meta.__dict__,
        "coreset": coreset_meta.__dict__,
        "coreset_ratio_requested": config.coreset_ratio,
        "coreset_ratio_actual": float(coreset_meta.num_vectors / max(1, baseline_meta.num_vectors)),
    }
    save_json(summary, config.results_dir / f"{config.category}_memory_summary.json")

    log(
        f"Baseline mem: {baseline_meta.memory_mb:.2f} MB | "
        f"Coreset mem: {coreset_meta.memory_mb:.2f} MB"
    )

    return summary


def load_memory_bank_metadata(path: Path) -> Dict[str, Any]:
    return torch.load(path, map_location="cpu")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--category", type=str, default="bottle")
    args = parser.parse_args()

    cfg = Config(category=args.category)
    build_and_save_memory_banks(cfg)
