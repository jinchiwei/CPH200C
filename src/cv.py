"""Patient-grouped CV splits + AUC confidence intervals.

Why grouped: same patient can have multiple encounters; row-level CV leaks.
We use first-encounter-only dedupe upstream, but the helpers here also work
for grouped CV if you keep all encounters.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, train_test_split


def patient_grouped_kfold(
    X: pd.DataFrame, y: pd.Series, groups: pd.Series, n_splits: int = 5, seed: int = 42,
):
    """Yield (train_idx, test_idx) per fold, grouped by `groups` and stratified by y."""
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, te in sgkf.split(X, y, groups=groups):
        yield tr, te


def patient_grouped_train_val_test(
    X: pd.DataFrame, y: pd.Series, groups: pd.Series,
    val_size: float = 0.15, test_size: float = 0.15, seed: int = 42,
):
    """Single 70/15/15 split with group-aware partitioning."""
    rng = np.random.default_rng(seed)
    unique_pids = np.array(sorted(groups.unique()))
    rng.shuffle(unique_pids)
    n = len(unique_pids)
    n_test = int(n * test_size)
    n_val = int(n * val_size)
    test_pids = set(unique_pids[:n_test])
    val_pids = set(unique_pids[n_test : n_test + n_val])
    train_idx = ~groups.isin(test_pids | val_pids)
    val_idx = groups.isin(val_pids)
    test_idx = groups.isin(test_pids)
    return (
        X[train_idx], X[val_idx], X[test_idx],
        y[train_idx], y[val_idx], y[test_idx],
    )


# ----- AUC + CIs ------------------------------------------------------------

def delong_auc_ci(y_true: np.ndarray, y_score: np.ndarray, alpha: float = 0.95) -> tuple[float, float, float]:
    """Compute AUC + DeLong 95% CI (via the `confidenceinterval` package)."""
    from confidenceinterval import roc_auc_score as ci_auc
    auc, (lo, hi) = ci_auc(y_true, y_score, confidence_level=alpha, method="delong")
    return float(auc), float(lo), float(hi)


def bootstrap_auc_ci(
    y_true: np.ndarray, y_score: np.ndarray, n_boot: int = 1000, alpha: float = 0.95, seed: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap AUC CI -- robust fallback if DeLong errors (e.g., constant scores)."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    aucs = []
    auc_point = roc_auc_score(y_true, y_score)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_score[idx]))
    a = (1 - alpha) / 2
    lo = float(np.quantile(aucs, a))
    hi = float(np.quantile(aucs, 1 - a))
    return float(auc_point), lo, hi
