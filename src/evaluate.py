import csv
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
from PIL import Image
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score

from src.config import Config
from src.data_loader import MVTecBottleDataset
from src.inference import AnomalyDetector
from src.metrics import (
    image_level_auroc,
    pixel_level_auroc,
    image_level_ap,
    find_best_threshold,
    classification_metrics,
    confusion_metrics,
)
from src.preprocess import preprocess_mask
from src.visualization import (
    plot_roc_curve,
    plot_pr_curve,
    plot_anomaly_result,
    save_metrics_json,
)
from src.utils import set_seed, log


def _pil_to_rgb_uint8(image: Image.Image) -> np.ndarray:
    return np.array(image.convert("RGB"), dtype=np.uint8)


def _mask_to_binary_np(mask: Image.Image, config: Config) -> np.ndarray:
    """
    Convert PIL mask to a resized binary numpy array [H, W] with values 0/1.
    Uses the same preprocessing pipeline as the model input so masks and
    heatmaps are aligned.
    """
    mask_tensor = preprocess_mask(mask, config)
    mask_np = mask_tensor.squeeze(0).cpu().numpy().astype(np.uint8)
    return mask_np


def _load_detector(config: Config, bank_type: str = "coreset") -> AnomalyDetector:
    return AnomalyDetector(config=config, bank_type=bank_type)


def _evaluate_split(
    config: Config,
    detector: AnomalyDetector,
    split: str = "test",
) -> Dict[str, Any]:
    """
    Run inference on the test split and collect image-level and pixel-level outputs.
    """
    dataset = MVTecBottleDataset(config, split=split)

    image_paths: List[str] = []
    defect_types: List[str] = []
    labels: List[int] = []
    scores: List[float] = []
    heatmaps: List[np.ndarray] = []
    masks: List[np.ndarray] = []

    for idx in range(len(dataset)):
        sample = dataset[idx]
        image_path = sample["image_path"]
        defect_type = sample["defect_type"]
        label = int(sample["label"])
        image = sample["image"]
        mask = sample["mask"]

        result = detector.predict(image)
        score = float(result["score"])
        heatmap = np.asarray(result["heatmap"], dtype=np.float32)

        image_paths.append(image_path)
        defect_types.append(defect_type)
        labels.append(label)
        scores.append(score)
        heatmaps.append(heatmap)

        if label == 1:
            if mask is None:
                raise FileNotFoundError(
                    f"Missing ground truth mask for defective image: {image_path}"
                )
            mask_np = _mask_to_binary_np(mask, config)
        else:
            mask_np = np.zeros((config.img_size, config.img_size), dtype=np.uint8)

        masks.append(mask_np)

    labels_np = np.asarray(labels, dtype=np.int32)
    scores_np = np.asarray(scores, dtype=np.float32)
    heatmaps_np = np.stack(heatmaps, axis=0).astype(np.float32)
    masks_np = np.stack(masks, axis=0).astype(np.uint8)

    img_auc = image_level_auroc(labels_np, scores_np)
    pixel_auc = pixel_level_auroc(masks_np, heatmaps_np)
    best = find_best_threshold(labels_np, scores_np)
    cls_metrics = classification_metrics(labels_np, scores_np, best.threshold)
    conf = confusion_metrics(labels_np, scores_np, best.threshold)

    fpr, tpr, _ = roc_curve(labels_np, scores_np)
    precision, recall, _ = precision_recall_curve(labels_np, scores_np)
    ap = image_level_ap(labels_np, scores_np)

    output = {
        "image_paths": image_paths,
        "defect_types": defect_types,
        "labels": labels_np.tolist(),
        "scores": scores_np.tolist(),
        "heatmaps": heatmaps_np,
        "masks": masks_np,
        "image_auc": float(img_auc),
        "pixel_auc": float(pixel_auc) if pixel_auc == pixel_auc else None,
        "best_threshold": float(best.threshold),
        "best_f1": float(best.best_f1),
        "best_precision": float(best.precision),
        "best_recall": float(best.recall),
        "ap": float(ap),
        "cls_metrics": cls_metrics,
        "confusion": conf,
        "roc_curve": (fpr, tpr),
        "pr_curve": (precision, recall),
    }
    return output


def _save_false_examples(
    config: Config,
    split_result: Dict[str, Any],
) -> Dict[str, Optional[str]]:
    """
    Save one false positive and one false negative example if available.
    """
    labels = np.asarray(split_result["labels"], dtype=np.int32)
    scores = np.asarray(split_result["scores"], dtype=np.float32)
    image_paths = split_result["image_paths"]
    heatmaps = np.asarray(split_result["heatmaps"], dtype=np.float32)

    threshold = float(split_result["best_threshold"])
    y_pred = (scores >= threshold).astype(np.int32)

    fp_dir = config.results_dir / "false_positives"
    fn_dir = config.results_dir / "false_negatives"
    fp_dir.mkdir(parents=True, exist_ok=True)
    fn_dir.mkdir(parents=True, exist_ok=True)

    fp_path_saved: Optional[str] = None
    fn_path_saved: Optional[str] = None

    for i, (y_t, y_p) in enumerate(zip(labels, y_pred)):
        if y_t == 0 and y_p == 1 and fp_path_saved is None:
            img = Image.open(image_paths[i]).convert("RGB")
            img_np = _pil_to_rgb_uint8(img)

            save_path = fp_dir / f"{i:03d}_fp.png"
            plot_anomaly_result(
                image=img_np,
                anomaly_map=heatmaps[i],
                title=f"False Positive | score={scores[i]:.4f}",
                save_path=save_path,
                show=False,
            )
            fp_path_saved = str(save_path)

        if y_t == 1 and y_p == 0 and fn_path_saved is None:
            img = Image.open(image_paths[i]).convert("RGB")
            img_np = _pil_to_rgb_uint8(img)

            save_path = fn_dir / f"{i:03d}_fn.png"
            plot_anomaly_result(
                image=img_np,
                anomaly_map=heatmaps[i],
                title=f"False Negative | score={scores[i]:.4f}",
                save_path=save_path,
                show=False,
            )
            fn_path_saved = str(save_path)

        if fp_path_saved is not None and fn_path_saved is not None:
            break

    return {
        "false_positive": fp_path_saved,
        "false_negative": fn_path_saved,
    }


def _write_pixel_auroc_report(
    config: Config,
    split_result: Dict[str, Any],
) -> Path:
    """
    Write a CSV report of pixel AUROC per defective image.
    """
    report_path = config.evaluation_dir / f"{config.category}_pixel_auroc_report.csv"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    image_paths = split_result["image_paths"]
    defect_types = split_result["defect_types"]
    labels = np.asarray(split_result["labels"], dtype=np.int32)
    heatmaps = np.asarray(split_result["heatmaps"], dtype=np.float32)
    masks = np.asarray(split_result["masks"], dtype=np.uint8)

    rows: List[Dict[str, Any]] = []

    for i, defect_type in enumerate(defect_types):
        if labels[i] == 1:
            try:
                auc = pixel_level_auroc(masks[i], heatmaps[i])
            except Exception:
                auc = float("nan")

            rows.append(
                {
                    "image_path": image_paths[i],
                    "defect_type": defect_type,
                    "label": int(labels[i]),
                    "pixel_auroc": float(auc) if auc == auc else "nan",
                }
            )

    with open(report_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["image_path", "defect_type", "label", "pixel_auroc"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    return report_path


def evaluate_model(
    config: Config,
    bank_type: str = "coreset",
) -> Dict[str, Any]:
    """
    Full evaluation pipeline:
        - image AUROC
        - pixel AUROC
        - threshold tuning
        - ROC / PR plots
        - false positive / negative export
        - metric JSON summary
    """
    config.ensure_dirs()
    set_seed(config.seed)

    detector = _load_detector(config, bank_type=bank_type)

    log(f"Evaluating category='{config.category}' with bank_type='{bank_type}'...")
    split_result = _evaluate_split(config, detector, split="test")

    best_threshold = float(split_result["best_threshold"])
    detector.save_threshold(best_threshold)

    failure_paths = _save_false_examples(config, split_result)

    fpr, tpr = split_result["roc_curve"]
    precision, recall = split_result["pr_curve"]

    roc_path = config.figures_dir / f"{config.category}_{bank_type}_roc_curve.png"
    pr_path = config.figures_dir / f"{config.category}_{bank_type}_pr_curve.png"

    plot_roc_curve(
        fpr=np.asarray(fpr),
        tpr=np.asarray(tpr),
        auc_score=float(split_result["image_auc"]),
        save_path=roc_path,
        show=False,
    )
    plot_pr_curve(
        recall=np.asarray(recall),
        precision=np.asarray(precision),
        ap_score=float(split_result["ap"]),
        save_path=pr_path,
        show=False,
    )

    pixel_report_path = _write_pixel_auroc_report(config, split_result)

    metrics_summary = {
        "category": config.category,
        "bank_type": bank_type,
        "image_auroc": float(split_result["image_auc"]),
        "pixel_auroc": split_result["pixel_auc"],
        "average_precision": float(split_result["ap"]),
        "best_threshold": best_threshold,
        "best_f1": float(split_result["best_f1"]),
        "best_precision": float(split_result["best_precision"]),
        "best_recall": float(split_result["best_recall"]),
        "classification": split_result["cls_metrics"],
        "confusion": split_result["confusion"],
        "roc_curve_path": str(roc_path),
        "pr_curve_path": str(pr_path),
        "false_positive_path": failure_paths["false_positive"],
        "false_negative_path": failure_paths["false_negative"],
        "pixel_report_path": str(pixel_report_path),
    }

    save_metrics_json(metrics_summary, config.metrics_path)

    config.threshold_path.write_text(f"{best_threshold:.8f}\n", encoding="utf-8")

    log(
        f"[{config.category}] Image AUROC: {metrics_summary['image_auroc']:.4f} | "
        f"Pixel AUROC: {metrics_summary['pixel_auroc']} | "
        f"Best threshold: {metrics_summary['best_threshold']:.6f} | "
        f"Best F1: {metrics_summary['best_f1']:.4f}"
    )

    return metrics_summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--category", type=str, default="bottle")
    args = parser.parse_args()

    cfg = Config(category=args.category)
    result = evaluate_model(cfg, bank_type="coreset")
    print(json.dumps(result, indent=2))
