import json
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def normalize_map(anomaly_map: np.ndarray) -> np.ndarray:
    """
    Normalize a heatmap to [0, 1].
    """
    anomaly_map = np.asarray(anomaly_map, dtype=np.float32)
    amin = float(np.min(anomaly_map))
    amax = float(np.max(anomaly_map))
    if amax > amin:
        anomaly_map = (anomaly_map - amin) / (amax - amin)
    else:
        anomaly_map = np.zeros_like(anomaly_map, dtype=np.float32)
    return np.clip(anomaly_map, 0.0, 1.0)


def apply_colormap(anomaly_map: np.ndarray, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """
    Convert a normalized anomaly map [0,1] into a color heatmap (BGR -> RGB).
    """
    norm_map = normalize_map(anomaly_map)
    heatmap = (norm_map * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return heatmap


def blend_heatmap_with_image(
    image: np.ndarray,
    anomaly_map: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    """
    Blend RGB image with an anomaly heatmap.
    image should be uint8 RGB in [0,255].
    """
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    heatmap = apply_colormap(anomaly_map)
    if heatmap.shape[:2] != image.shape[:2]:
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)

    blended = (1.0 - alpha) * image.astype(np.float32) + alpha * heatmap.astype(np.float32)
    return np.clip(blended, 0, 255).astype(np.uint8)


def tensor_to_uint8_image(image_tensor: np.ndarray) -> np.ndarray:
    """
    Convert a normalized/float image array to uint8 for plotting.
    Expects image in shape [H, W, C] or [C, H, W].
    """
    arr = np.asarray(image_tensor)
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))

    arr = np.asarray(arr, dtype=np.float32)
    amin, amax = float(arr.min()), float(arr.max())
    if amax > amin:
        arr = (arr - amin) / (amax - amin)
    else:
        arr = np.zeros_like(arr, dtype=np.float32)

    return (arr * 255).astype(np.uint8)


def plot_anomaly_result(
    image: np.ndarray,
    anomaly_map: np.ndarray,
    title: str = "",
    alpha: float = 0.45,
    save_path: Optional[Path] = None,
    show: bool = False,
) -> None:
    """
    Plot original image, heatmap, and overlay.
    image should be uint8 RGB.
    """
    image = np.asarray(image)
    anomaly_map = np.asarray(anomaly_map)

    heatmap = apply_colormap(anomaly_map)
    overlay = blend_heatmap_with_image(image, anomaly_map, alpha=alpha)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    if title:
        fig.suptitle(title, fontsize=14)

    axes[0].imshow(image)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(heatmap)
    axes[1].set_title("Heatmap")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=200)

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    auc_score: float,
    title: str = "ROC Curve",
    save_path: Optional[Path] = None,
    show: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, label=f"AUC = {auc_score:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=200)

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_pr_curve(
    recall: np.ndarray,
    precision: np.ndarray,
    ap_score: float,
    title: str = "Precision-Recall Curve",
    save_path: Optional[Path] = None,
    show: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(recall, precision, label=f"AP = {ap_score:.4f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=200)

    if show:
        plt.show()
    else:
        plt.close(fig)


def save_metrics_json(metrics: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)


def overlay_mask_on_image(
    image: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int] = (255, 0, 0),
    alpha: float = 0.35,
) -> np.ndarray:
    """
    Overlay a binary mask on an RGB image.
    """
    image = np.asarray(image)
    mask = np.asarray(mask)

    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    if mask.ndim == 3:
        mask = mask.squeeze()

    if mask.max() > 1:
        mask = (mask > 0).astype(np.uint8)

    overlay = image.copy().astype(np.float32)
    colored = np.zeros_like(image, dtype=np.float32)
    colored[:, :] = np.array(color, dtype=np.float32)

    mask_3d = np.repeat(mask[:, :, None], 3, axis=2).astype(np.float32)
    overlay = overlay * (1.0 - alpha * mask_3d) + colored * (alpha * mask_3d)
    return np.clip(overlay, 0, 255).astype(np.uint8)


def save_grid_images(
    images: List[np.ndarray],
    titles: List[str],
    save_path: Optional[Path] = None,
    ncols: int = 2,
    figsize: Tuple[int, int] = (12, 8),
    show: bool = False,
) -> None:
    """
    Save a grid of images with titles.
    """
    n = len(images)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.array(axes).reshape(-1)

    for i, ax in enumerate(axes):
        ax.axis("off")
        if i < n:
            ax.imshow(images[i])
            ax.set_title(titles[i])

    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=200)

    if show:
        plt.show()
    else:
        plt.close(fig)
