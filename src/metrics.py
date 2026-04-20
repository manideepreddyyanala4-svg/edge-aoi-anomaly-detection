from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    confusion_matrix,
)


@dataclass
class ThresholdResult:
    threshold: float
    best_f1: float
    precision: float
    recall: float


def safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(np.int32)
    y_score = np.asarray(y_score).astype(np.float32)

    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def safe_average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(np.int32)
    y_score = np.asarray(y_score).astype(np.float32)

    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def find_best_threshold(y_true: np.ndarray, y_score: np.ndarray) -> ThresholdResult:
    y_true = np.asarray(y_true).astype(np.int32)
    y_score = np.asarray(y_score).astype(np.float32)

    if len(np.unique(y_true)) < 2:
        threshold = 0.5
        preds = (y_score >= threshold).astype(np.int32)
        return ThresholdResult(
            threshold=float(threshold),
            best_f1=float(f1_score(y_true, preds, zero_division=0)) if len(y_true) > 0 else 0.0,
            precision=float(precision_score(y_true, preds, zero_division=0)) if len(y_true) > 0 else 0.0,
            recall=float(recall_score(y_true, preds, zero_division=0)) if len(y_true) > 0 else 0.0,
        )

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_score)

    f1s = (2 * precisions[:-1] * recalls[:-1]) / np.clip(precisions[:-1] + recalls[:-1], 1e-12, None)

    if len(f1s) == 0:
        threshold = 0.5
        preds = (y_score >= threshold).astype(np.int32)
        return ThresholdResult(
            threshold=float(threshold),
            best_f1=float(f1_score(y_true, preds, zero_division=0)),
            precision=float(precision_score(y_true, preds, zero_division=0)),
            recall=float(recall_score(y_true, preds, zero_division=0)),
        )

    best_idx = int(np.argmax(f1s))
    best_threshold = float(thresholds[best_idx])
    best_f1 = float(f1s[best_idx])
    best_precision = float(precisions[best_idx])
    best_recall = float(recalls[best_idx])

    return ThresholdResult(
        threshold=best_threshold,
        best_f1=best_f1,
        precision=best_precision,
        recall=best_recall,
    )


def classification_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(np.int32)
    y_score = np.asarray(y_score).astype(np.float32)
    y_pred = (y_score >= threshold).astype(np.int32)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_true)) > 1 else float("nan"),
    }


def confusion_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
) -> Dict[str, int]:
    y_true = np.asarray(y_true).astype(np.int32)
    y_score = np.asarray(y_score).astype(np.float32)
    y_pred = (y_score >= threshold).astype(np.int32)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def pixel_level_auroc(mask_true: np.ndarray, anomaly_map: np.ndarray) -> float:
    mask_true = np.asarray(mask_true).astype(np.int32).reshape(-1)
    anomaly_map = np.asarray(anomaly_map).astype(np.float32).reshape(-1)
    return safe_roc_auc(mask_true, anomaly_map)


def image_level_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    return safe_roc_auc(labels, scores)


def image_level_ap(labels: np.ndarray, scores: np.ndarray) -> float:
    return safe_average_precision(labels, scores)


def summarize_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    summary = {}
    summary["auroc"] = safe_roc_auc(y_true, y_score)
    summary["ap"] = safe_average_precision(y_true, y_score)

    cls = classification_metrics(y_true, y_score, threshold)
    summary.update(cls)

    return summary


def per_image_pixel_auroc(
    masks: np.ndarray,
    anomaly_maps: np.ndarray,
) -> float:
    masks = np.asarray(masks).astype(np.int32).reshape(-1)
    anomaly_maps = np.asarray(anomaly_maps).astype(np.float32).reshape(-1)
    return safe_roc_auc(masks, anomaly_maps)


def per_image_best_threshold(
    labels: np.ndarray,
    scores: np.ndarray,
) -> ThresholdResult:
    return find_best_threshold(labels, scores)
