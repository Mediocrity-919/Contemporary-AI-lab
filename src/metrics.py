"""评估指标：准确率、每类指标、混淆矩阵、NLL、ECE、Brier、Wilson 区间。输入均为 numpy 数组。"""
from __future__ import annotations

import numpy as np
from scipy.special import log_softmax, softmax
from scipy.stats import norm


def confusion_matrix(labels: np.ndarray, preds: np.ndarray, num_classes: int) -> np.ndarray:
    """cm[i, j] = 真实类别 i 被预测为 j 的数量。"""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(cm, (labels, preds), 1)
    return cm


def per_class_report(cm: np.ndarray) -> dict:
    tp = np.diag(cm).astype(float)
    recall = tp / np.maximum(cm.sum(1), 1)          # 即“每类准确率”
    precision = tp / np.maximum(cm.sum(0), 1)
    f1 = np.where(precision + recall > 0, 2 * precision * recall / np.maximum(precision + recall, 1e-12), 0)
    return {"accuracy": recall.tolist(), "precision": precision.tolist(), "recall": recall.tolist(),
            "f1": f1.tolist(), "support": cm.sum(1).tolist()}


def nll(logits: np.ndarray, labels: np.ndarray) -> float:
    return float(-log_softmax(logits, axis=1)[np.arange(len(labels)), labels].mean())


def ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """期望校准误差：按最大置信度等宽分箱，|准确率 − 平均置信度| 的加权平均。"""
    conf, pred = probs.max(1), probs.argmax(1)
    correct = (pred == labels).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.any():
            total += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(total)


def reliability_bins(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15):
    conf, pred = probs.max(1), probs.argmax(1)
    correct = (pred == labels).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi)
        rows.append((lo, hi, int(mask.sum()), correct[mask].mean() if mask.any() else np.nan,
                     conf[mask].mean() if mask.any() else np.nan))
    return rows


def brier(probs: np.ndarray, labels: np.ndarray) -> float:
    onehot = np.eye(probs.shape[1])[labels]
    return float(((probs - onehot) ** 2).sum(1).mean())


def wilson_ci(correct: int, n: int, level: float = 0.95) -> tuple[float, float]:
    """二项比例的 Wilson 区间（只反映评估集抽样不确定性）。"""
    z = norm.ppf(0.5 + level / 2)
    p = correct / n
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return float(center - half), float(center + half)


def full_report(logits: np.ndarray, labels: np.ndarray, class_names: list[str]) -> dict:
    probs = softmax(logits, axis=1)
    preds = probs.argmax(1)
    cm = confusion_matrix(labels, preds, len(class_names))
    correct = int((preds == labels).sum())
    return {
        "n": int(len(labels)),
        "accuracy": correct / len(labels),
        "accuracy_wilson95": wilson_ci(correct, len(labels)),
        "nll": nll(logits, labels),
        "ece15": ece(probs, labels),
        "brier": brier(probs, labels),
        "class_names": class_names,
        "per_class": per_class_report(cm),
        "confusion_matrix": cm.tolist(),
    }
