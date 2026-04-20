import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.config import Config
from src.models import FeatureExtractor, fuse_patch_embeddings
from src.preprocess import preprocess_image
from src.utils import log


class AnomalyDetector:
    """
    PatchCore-style anomaly detector for MVTec AD.

    GPU-only runtime path:
        image -> preprocess -> feature extractor -> patch embeddings -> torch.cdist
              -> patch distances -> anomaly score -> heatmap -> status
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        bank_type: str = "coreset",
        threshold: Optional[float] = None,
        device: Optional[torch.device] = None,
    ):
        self.config = config or Config()
        self.config.ensure_dirs()

        self.bank_type = bank_type.lower()
        if self.bank_type not in {"baseline", "coreset"}:
            raise ValueError("bank_type must be either 'baseline' or 'coreset'.")

        self.device = self._resolve_device(device)
        self.threshold = threshold if threshold is not None else self._load_threshold()

        self.extractor = FeatureExtractor(self.config).to(self.device)
        self.extractor.eval()

        self.memory_vectors, self.bank_metadata = self._load_memory_bank()
        self.grid_h = int(self.bank_metadata["grid_h"])
        self.grid_w = int(self.bank_metadata["grid_w"])

    def _resolve_device(self, device: Optional[torch.device] = None) -> torch.device:
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

    def _bank_paths(self) -> Path:
        if self.bank_type == "baseline":
            return self.config.memory_bank_dir / f"{self.config.category}_full_bank.pt"
        return self.config.memory_bank_dir / f"{self.config.category}_coreset_bank.pt"

    def _load_threshold(self) -> float:
        threshold_path = self.config.threshold_path
        if threshold_path.exists():
            try:
                value = float(threshold_path.read_text().strip())
                return value
            except Exception:
                pass
        return 0.5

    def _load_memory_bank(self):
        bank_path = self._bank_paths()
        if not bank_path.exists():
            raise FileNotFoundError(
                f"Memory bank not found: {bank_path}. Run src/build_memory.py first."
            )

        payload = torch.load(bank_path, map_location=self.device)
        meta = payload.get("info", {})
        vectors = payload.get("vectors", None)

        if not meta:
            raise RuntimeError(f"Invalid memory bank payload: {bank_path}")
        if vectors is None:
            raise RuntimeError(f"Vectors missing in memory bank payload: {bank_path}")

        vectors = vectors.to(self.device).contiguous().float()
        log(f"Loaded {self.bank_type} bank: {meta['num_vectors']} vectors, dim={meta['vector_dim']}")
        return vectors, meta

    def _load_input_image(self, image_input: Union[str, Path, Image.Image]) -> Image.Image:
        if isinstance(image_input, Image.Image):
            return image_input.convert("RGB")
        return Image.open(str(image_input)).convert("RGB")

    def _preprocess(self, image: Image.Image) -> torch.Tensor:
        return preprocess_image(image, self.config).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def _extract_patch_embeddings(self, image_tensor: torch.Tensor) -> torch.Tensor:
        fused_map = self.extractor.extract_fused_feature_map(image_tensor)
        patch_embeddings = fuse_patch_embeddings(fused_map)
        return patch_embeddings.contiguous().float()

    def _query_memory(self, patch_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Query memory bank and return per-patch nearest-neighbor distances.
        Runs entirely on the selected device.
        """
        memory = self.memory_vectors
        patch_embeddings = patch_embeddings.to(self.device)

        chunk_size = 1024
        best_distances = torch.full(
            (patch_embeddings.shape[0],),
            float("inf"),
            device=self.device,
            dtype=torch.float32,
        )

        for start in range(0, memory.shape[0], chunk_size):
            chunk = memory[start : start + chunk_size]
            distances = torch.cdist(patch_embeddings, chunk, p=2)
            chunk_best = distances.min(dim=1).values
            best_distances = torch.minimum(best_distances, chunk_best)

        return best_distances

    def _reshape_distances_to_grid(self, patch_distances: torch.Tensor) -> torch.Tensor:
        if patch_distances.numel() != self.grid_h * self.grid_w:
            raise ValueError(
                f"Patch count mismatch. Expected {self.grid_h * self.grid_w}, got {patch_distances.numel()}."
            )
        return patch_distances.view(self.grid_h, self.grid_w)

    def _make_gaussian_kernel(self, sigma: float) -> torch.Tensor:
        ksize = max(3, int(2 * round(3 * sigma) + 1))
        coords = torch.arange(ksize, device=self.device, dtype=torch.float32)
        coords = coords - (ksize - 1) / 2.0
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        kernel = torch.exp(-(xx**2 + yy**2) / (2.0 * sigma * sigma))
        kernel = kernel / kernel.sum()
        return kernel.view(1, 1, ksize, ksize)

    def _upsample_and_smooth(self, anomaly_grid: torch.Tensor) -> np.ndarray:
        grid = anomaly_grid.to(self.device).float()

        grid = grid - grid.min()
        grid = grid / (grid.max() + 1e-8)

        grid = grid.unsqueeze(0).unsqueeze(0)

        upsampled = F.interpolate(
            grid,
            size=(self.config.img_size, self.config.img_size),
            mode="bilinear",
            align_corners=False
        )

        sigma = float(self.config.heatmap_sigma)
        if sigma > 0:
            kernel = self._make_gaussian_kernel(sigma)
            pad = kernel.shape[-1] // 2
            upsampled = F.conv2d(upsampled, kernel, padding=pad)

        upsampled = upsampled - upsampled.min()
        upsampled = upsampled / (upsampled.max() + 1e-8)

        return upsampled.squeeze().cpu().numpy().astype(np.float32)

    def predict(self, image_input: Union[str, Path, Image.Image]) -> Dict[str, Any]:
        start = time.time()

        image = self._load_input_image(image_input)
        x = self._preprocess(image)

        with torch.inference_mode():
            patch_embeddings = self._extract_patch_embeddings(x)

        patch_distances = self._query_memory(patch_embeddings)
        score = float(patch_distances.max().item())

        anomaly_grid = self._reshape_distances_to_grid(patch_distances)
        heatmap = self._upsample_and_smooth(anomaly_grid)

        status = "FAIL" if score >= self.threshold else "PASS"
        latency_ms = (time.time() - start) * 1000.0

        return {
            "score": score,
            "threshold": float(self.threshold),
            "status": status,
            "heatmap": heatmap,
            "latency_ms": float(latency_ms),
            "grid_shape": (self.grid_h, self.grid_w),
        }

    def predict_path(self, image_path: Union[str, Path]) -> Dict[str, Any]:
        return self.predict(image_path)

    def predict_image(self, image: Image.Image) -> Dict[str, Any]:
        return self.predict(image)

    def save_threshold(self, threshold: float) -> None:
        self.config.threshold_path.parent.mkdir(parents=True, exist_ok=True)
        self.config.threshold_path.write_text(f"{float(threshold)}\n", encoding="utf-8")

    @property
    def bank_info(self) -> Dict[str, Any]:
        return dict(self.bank_metadata)
