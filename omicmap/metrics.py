from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np


def pearson_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """计算 Pearson 相关系数；常量向量时返回 0.0。"""
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    if y_true.size == 0:
        return 0.0
    t_std = float(np.std(y_true))
    p_std = float(np.std(y_pred))
    if t_std < 1e-12 or p_std < 1e-12:
        return 0.0
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """计算 RMSE。"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """计算 R2；当分母为0时返回 0.0。"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot < 1e-12:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def regression_metrics_per_trait(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    trait_names: Sequence[str],
) -> List[Dict[str, float | str]]:
    """
    逐 trait 计算 Pearson / RMSE / R2。
    输入 shape: [N, T]
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    assert y_true.shape == y_pred.shape
    assert y_true.shape[1] == len(trait_names)

    rows: List[Dict[str, float | str]] = []
    for i, trait in enumerate(trait_names):
        yt = y_true[:, i]
        yp = y_pred[:, i]
        pearson = pearson_corr(yt, yp)
        rows.append(
            {
                "trait": str(trait),
                "pearson": pearson,
                "pcc": pearson,
                "rmse": rmse(yt, yp),
                "r2": r2_score(yt, yp),
            }
        )
    return rows
