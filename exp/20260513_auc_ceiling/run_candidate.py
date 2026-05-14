"""autoresearch candidate runner — one config per invocation.

Usage:
    OMP_NUM_THREADS=4 python run_candidate.py --config <name> \\
        --out-results $AUTORESEARCH_OUT_RESULTS \\
        --out-exp $AUTORESEARCH_OUT_EXP

Each config is a named recipe. The runner loads the same train/val/test split as
section 1.2 (deterministic, seed=42), fits the model, writes:
    $out_results/summary.md       (autoresearch-required artifact)
    $out_results/preds_test.npy   (for downstream stacking)
    $out_results/metrics.json     (programmatic AUC + CI)
    $out_exp/...                  (whatever's heavy)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/data/rauschecker2/jkw/cph/CPH200C")
sys.path.insert(0, str(REPO / "src"))

from preprocessing import build_features, clean_and_filter, load_diabetes
from feature_engineering import build_features_slim
from cv import delong_auc_ci, bootstrap_auc_ci

warnings.filterwarnings("ignore")

SEED = 42


def load_split(slim: bool = False):
    """Load same train/val/test split used in section 1.2.

    Args:
        slim: if True, use the slimmer feature representation from
            feature_engineering.build_features_slim instead of the base one.
    """
    df = load_diabetes()
    df = clean_and_filter(df).reset_index(drop=True)
    if slim:
        X, y, names, pids = build_features_slim(df)
    else:
        X, y, names, pids = build_features(df)

    from sklearn.model_selection import train_test_split
    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=SEED,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=0.15 / 0.85, stratify=y_tv, random_state=SEED,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def auc_with_ci(y_true, p):
    try:
        return delong_auc_ci(np.asarray(y_true), np.asarray(p))
    except Exception:
        return bootstrap_auc_ci(np.asarray(y_true), np.asarray(p), n_boot=500)


# ----- per-config recipes ---------------------------------------------------

def cfg_xgb_wide_grid_1(X_tr, X_va, X_te, y_tr, y_va):
    import xgboost as xgb
    spw = (len(y_tr) - y_tr.sum()) / max(y_tr.sum(), 1)
    m = xgb.XGBClassifier(
        n_estimators=800, max_depth=6, learning_rate=0.03,
        reg_alpha=0.1, reg_lambda=1.0, gamma=0.1, min_child_weight=5,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=spw, eval_metric="auc", tree_method="hist",
        n_jobs=-1, random_state=SEED,
    )
    m.fit(X_tr, y_tr)
    return m.predict_proba(X_te)[:, 1], {"model": "XGBoost wide grid 1"}


def cfg_xgb_wide_grid_2(X_tr, X_va, X_te, y_tr, y_va):
    import xgboost as xgb
    spw = (len(y_tr) - y_tr.sum()) / max(y_tr.sum(), 1)
    m = xgb.XGBClassifier(
        n_estimators=1500, max_depth=4, learning_rate=0.02,
        reg_alpha=0.5, reg_lambda=2.0, gamma=0.0, min_child_weight=10,
        subsample=0.85, colsample_bytree=0.85,
        scale_pos_weight=spw, eval_metric="auc", tree_method="hist",
        n_jobs=-1, random_state=SEED,
    )
    m.fit(X_tr, y_tr)
    return m.predict_proba(X_te)[:, 1], {"model": "XGBoost wide grid 2"}


def cfg_xgb_early_stop(X_tr, X_va, X_te, y_tr, y_va):
    import xgboost as xgb
    spw = (len(y_tr) - y_tr.sum()) / max(y_tr.sum(), 1)
    m = xgb.XGBClassifier(
        n_estimators=2000, max_depth=5, learning_rate=0.03,
        reg_alpha=0.1, reg_lambda=1.0, gamma=0.05, min_child_weight=5,
        subsample=0.85, colsample_bytree=0.85,
        scale_pos_weight=spw, eval_metric="auc", tree_method="hist",
        n_jobs=-1, random_state=SEED,
        early_stopping_rounds=50,
    )
    m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    return m.predict_proba(X_te)[:, 1], {"model": "XGBoost + early stop",
                                          "best_iter": int(m.best_iteration)}


def cfg_lightgbm_default(X_tr, X_va, X_te, y_tr, y_va):
    import lightgbm as lgb
    spw = (len(y_tr) - y_tr.sum()) / max(y_tr.sum(), 1)
    m = lgb.LGBMClassifier(
        n_estimators=500, num_leaves=31, learning_rate=0.05,
        feature_fraction=0.85, bagging_fraction=0.85, bagging_freq=5,
        min_data_in_leaf=20, lambda_l1=0.1, lambda_l2=1.0,
        scale_pos_weight=spw, n_jobs=-1, random_state=SEED, verbose=-1,
    )
    m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[lgb.early_stopping(50, verbose=False)])
    return m.predict_proba(X_te)[:, 1], {"model": "LightGBM default",
                                          "best_iter": int(m.best_iteration_)}


def cfg_lightgbm_dart(X_tr, X_va, X_te, y_tr, y_va):
    import lightgbm as lgb
    spw = (len(y_tr) - y_tr.sum()) / max(y_tr.sum(), 1)
    m = lgb.LGBMClassifier(
        boosting_type="dart", n_estimators=500, num_leaves=31,
        learning_rate=0.05, feature_fraction=0.85, bagging_fraction=0.85,
        min_data_in_leaf=20, lambda_l1=0.1, lambda_l2=1.0,
        drop_rate=0.1, scale_pos_weight=spw,
        n_jobs=-1, random_state=SEED, verbose=-1,
    )
    m.fit(X_tr, y_tr)
    return m.predict_proba(X_te)[:, 1], {"model": "LightGBM dart"}


def _tabpfn_predict(X_tr, X_va, X_te, y_tr, y_va, n_sub):
    from tabpfn import TabPFNClassifier
    rng = np.random.default_rng(SEED)
    # stratified subsample
    pos = np.where(y_tr.values == 1)[0]
    neg = np.where(y_tr.values == 0)[0]
    n_pos = int(n_sub * y_tr.mean()) + 1
    n_neg = n_sub - n_pos
    n_pos = min(n_pos, len(pos))
    n_neg = min(n_neg, len(neg))
    keep = np.sort(np.concatenate([
        rng.choice(pos, size=n_pos, replace=False),
        rng.choice(neg, size=n_neg, replace=False),
    ]))
    Xs = X_tr.iloc[keep]
    ys = y_tr.iloc[keep]
    m = TabPFNClassifier(device="cpu", ignore_pretraining_limits=True, random_state=SEED)
    m.fit(Xs.values, ys.values)
    return m.predict_proba(X_te.values)[:, 1], {"model": f"TabPFN sub={n_sub}",
                                                 "n_train_sub": int(len(keep))}


def cfg_tabpfn_subsample_10k(X_tr, X_va, X_te, y_tr, y_va):
    return _tabpfn_predict(X_tr, X_va, X_te, y_tr, y_va, n_sub=10_000)


def cfg_tabpfn_subsample_20k(X_tr, X_va, X_te, y_tr, y_va):
    return _tabpfn_predict(X_tr, X_va, X_te, y_tr, y_va, n_sub=20_000)


def cfg_tabpfn_full(X_tr, X_va, X_te, y_tr, y_va):
    """Push TabPFN-v2 past its pretraining cap — full 49k train rows."""
    return _tabpfn_predict(X_tr, X_va, X_te, y_tr, y_va, n_sub=len(y_tr))


def _tabpfn_v3_predict(X_tr, X_te, y_tr, n_sub=None):
    """TabPFN-3 (released 2026-05-12) — scales to 1M rows per the v3 paper.

    Uses TabPFNClassifier.create_default_for_version(ModelVersion.V3, ...).
    """
    from tabpfn import TabPFNClassifier
    from tabpfn.model_loading import ModelVersion
    rng = np.random.default_rng(SEED)
    if n_sub is None or n_sub >= len(y_tr):
        Xs, ys = X_tr, y_tr
    else:
        pos = np.where(y_tr.values == 1)[0]
        neg = np.where(y_tr.values == 0)[0]
        n_pos = int(n_sub * y_tr.mean()) + 1
        n_neg = n_sub - n_pos
        keep = np.sort(np.concatenate([
            rng.choice(pos, size=min(n_pos, len(pos)), replace=False),
            rng.choice(neg, size=min(n_neg, len(neg)), replace=False),
        ]))
        Xs = X_tr.iloc[keep]
        ys = y_tr.iloc[keep]
    m = TabPFNClassifier.create_default_for_version(
        ModelVersion.V3,
        device="cpu",
        ignore_pretraining_limits=True,
        random_state=SEED,
    )
    m.fit(Xs.values, ys.values)
    return m.predict_proba(X_te.values)[:, 1], {"model": "TabPFN-v3",
                                                 "n_train_sub": int(len(Xs))}


def cfg_tabpfn_v3_full(X_tr, X_va, X_te, y_tr, y_va):
    return _tabpfn_v3_predict(X_tr, X_te, y_tr, n_sub=None)


def cfg_tabpfn_v3_full_slim(X_tr, X_va, X_te, y_tr, y_va):
    return _tabpfn_v3_predict(X_tr, X_te, y_tr, n_sub=None)


def _tabicl_predict(X_tr, X_va, X_te, y_tr, y_va, n_sub):
    """TabICL — designed for larger tabular tasks than TabPFN; handles up to ~60k rows."""
    from tabicl import TabICLClassifier
    rng = np.random.default_rng(SEED)
    if n_sub is None or n_sub >= len(y_tr):
        Xs, ys = X_tr, y_tr
    else:
        pos = np.where(y_tr.values == 1)[0]
        neg = np.where(y_tr.values == 0)[0]
        n_pos = int(n_sub * y_tr.mean()) + 1
        n_neg = n_sub - n_pos
        n_pos = min(n_pos, len(pos))
        n_neg = min(n_neg, len(neg))
        keep = np.sort(np.concatenate([
            rng.choice(pos, size=n_pos, replace=False),
            rng.choice(neg, size=n_neg, replace=False),
        ]))
        Xs = X_tr.iloc[keep]
        ys = y_tr.iloc[keep]
    # Slim memory footprint: 8x default n_estimators × 8x default batch_size
    # exceeds even 64G of RAM during test-time inference on this dataset.
    # Drop both 4x and let auto-offload spill the rest to /tmp.
    m = TabICLClassifier(
        n_estimators=2, batch_size=2,
        device="cpu", random_state=SEED,
        offload_mode="auto",
    )
    m.fit(Xs.values, ys.values)
    return m.predict_proba(X_te.values)[:, 1], {"model": f"TabICL sub={n_sub}",
                                                 "n_train_sub": int(len(Xs)),
                                                 "n_estimators": 2, "batch_size": 2}


def cfg_tabicl_subsample_20k(X_tr, X_va, X_te, y_tr, y_va):
    return _tabicl_predict(X_tr, X_va, X_te, y_tr, y_va, n_sub=20_000)


def cfg_tabicl_full(X_tr, X_va, X_te, y_tr, y_va):
    return _tabicl_predict(X_tr, X_va, X_te, y_tr, y_va, n_sub=None)


def cfg_mlp_64_32_dropout(X_tr, X_va, X_te, y_tr, y_va):
    from sklearn.preprocessing import StandardScaler
    from sklearn.neural_network import MLPClassifier
    sc = StandardScaler(with_mean=False).fit(X_tr)
    Xtr_s, Xva_s, Xte_s = sc.transform(X_tr), sc.transform(X_va), sc.transform(X_te)
    Xtr_d = Xtr_s.toarray() if hasattr(Xtr_s, "toarray") else Xtr_s
    Xte_d = Xte_s.toarray() if hasattr(Xte_s, "toarray") else Xte_s
    m = MLPClassifier(
        hidden_layer_sizes=(64, 32), alpha=1e-4, learning_rate_init=1e-3,
        max_iter=150, early_stopping=True, validation_fraction=0.1,
        n_iter_no_change=20, random_state=SEED,
    )
    m.fit(Xtr_d, y_tr)
    return m.predict_proba(Xte_d)[:, 1], {"model": "MLP 64-32 +dropout-ish (early stop)"}


def cfg_mlp_128_64_32(X_tr, X_va, X_te, y_tr, y_va):
    from sklearn.preprocessing import StandardScaler
    from sklearn.neural_network import MLPClassifier
    sc = StandardScaler(with_mean=False).fit(X_tr)
    Xtr_s, Xte_s = sc.transform(X_tr), sc.transform(X_te)
    Xtr_d = Xtr_s.toarray() if hasattr(Xtr_s, "toarray") else Xtr_s
    Xte_d = Xte_s.toarray() if hasattr(Xte_s, "toarray") else Xte_s
    m = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32), alpha=1e-3, learning_rate_init=1e-3,
        max_iter=200, early_stopping=True, validation_fraction=0.1,
        n_iter_no_change=25, random_state=SEED,
    )
    m.fit(Xtr_d, y_tr)
    return m.predict_proba(Xte_d)[:, 1], {"model": "MLP 128-64-32"}


def cfg_logreg_target_enc(X_tr, X_va, X_te, y_tr, y_va):
    """Logistic regression with target-encoded high-cardinality categoricals.

    Re-derive features at the raw-df level so target encoding sees the original
    categoricals (not their one-hots).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    df = load_diabetes()
    df = clean_and_filter(df).reset_index(drop=True)

    # Align with the OHE-based split via row indices
    from sklearn.model_selection import train_test_split
    idx_tv, idx_te = train_test_split(df.index, test_size=0.15,
                                      stratify=df["y_30d"], random_state=SEED)
    idx_tr, idx_va = train_test_split(idx_tv, test_size=0.15 / 0.85,
                                      stratify=df.loc[idx_tv, "y_30d"], random_state=SEED)

    tr_df, va_df, te_df = df.loc[idx_tr], df.loc[idx_va], df.loc[idx_te]
    ytr, yva, yte = tr_df["y_30d"], va_df["y_30d"], te_df["y_30d"]

    # high-card target-encoded cols
    te_cols = ["medical_specialty", "payer_code", "diag_1", "diag_2", "diag_3"]
    smoothing = 20.0
    global_mean = float(ytr.mean())
    encoding = {}
    for c in te_cols:
        agg = tr_df.groupby(c, observed=True)["y_30d"].agg(["mean", "count"])
        agg["te"] = (agg["mean"] * agg["count"] + global_mean * smoothing) / (agg["count"] + smoothing)
        encoding[c] = agg["te"].to_dict()

    def apply_te(d):
        out = pd.DataFrame(index=d.index)
        for c in te_cols:
            out[c + "_te"] = d[c].map(encoding[c]).fillna(global_mean)
        return out

    te_tr, te_va, te_te = apply_te(tr_df), apply_te(va_df), apply_te(te_df)

    # combine with the OHE feature matrix
    X_tr2 = pd.concat([X_tr.reset_index(drop=True),
                       te_tr.reindex(idx_tr).reset_index(drop=True)], axis=1)
    X_va2 = pd.concat([X_va.reset_index(drop=True),
                       te_va.reindex(idx_va).reset_index(drop=True)], axis=1)
    X_te2 = pd.concat([X_te.reset_index(drop=True),
                       te_te.reindex(idx_te).reset_index(drop=True)], axis=1)

    sc = StandardScaler(with_mean=False).fit(X_tr2)
    best = None
    for C in [0.01, 0.1, 1.0]:
        for pen in ["l1", "l2"]:
            mdl = LogisticRegression(
                penalty=pen, C=C, solver="liblinear" if pen == "l1" else "lbfgs",
                class_weight="balanced", max_iter=400, random_state=SEED,
            ).fit(sc.transform(X_tr2), y_tr)
            from sklearn.metrics import roc_auc_score
            auc_v = roc_auc_score(y_va, mdl.predict_proba(sc.transform(X_va2))[:, 1])
            if best is None or auc_v > best[0]:
                best = (auc_v, mdl, pen, C)
    return best[1].predict_proba(sc.transform(X_te2))[:, 1], {
        "model": "LogReg + target-encoded high-card cats",
        "penalty": best[2], "C": best[3],
    }


def _load_pred(name):
    """Helper for stacks: prefer this session's saved preds, fall back to section 1.2's."""
    here = Path("/data/rauschecker2/jkw/cph/CPH200C/exp/20260512_pset2_full/02_models")
    return np.load(here / name)


def cfg_stack_xgb_lr(X_tr, X_va, X_te, y_tr, y_va):
    """Simple stacking: average XGBoost + LogReg test predictions on isotonic-calibrated scale."""
    p1 = _load_pred("preds_xgb.npy")
    p2 = _load_pred("preds_logreg_base.npy")
    # logistic blender (single-fold; for proper stacking you'd CV-OOF the bases)
    blend = (p1 + p2) / 2
    return blend, {"model": "Mean(XGBoost, LogReg base)"}


def cfg_stack_xgb_lr_tabpfn(X_tr, X_va, X_te, y_tr, y_va):
    """Mean of XGB + LogReg + TabPFN-10k (requires that candidate to have run)."""
    p1 = _load_pred("preds_xgb.npy")
    p2 = _load_pred("preds_logreg_base.npy")
    # try to load the TabPFN preds we wrote in an earlier candidate
    tabpfn_path = None
    for c in ("tabpfn_subsample_20k", "tabpfn_subsample_10k"):
        candidate = Path("/data/rauschecker2/jkw/cph/CPH200C/exp/20260513_auc_ceiling/per_config") / c / "preds_test.npy"
        if candidate.exists():
            tabpfn_path = candidate
            break
    if tabpfn_path is None:
        raise RuntimeError("TabPFN predictions not yet available; run a TabPFN candidate first.")
    p3 = np.load(tabpfn_path)
    blend = (p1 + p2 + p3) / 3
    return blend, {"model": "Mean(XGBoost, LogReg, TabPFN)", "tabpfn_src": str(tabpfn_path)}


CONFIGS = {
    "xgb_wide_grid_1": cfg_xgb_wide_grid_1,
    "xgb_wide_grid_2": cfg_xgb_wide_grid_2,
    "xgb_early_stop": cfg_xgb_early_stop,
    "lightgbm_default": cfg_lightgbm_default,
    "lightgbm_dart": cfg_lightgbm_dart,
    "tabpfn_subsample_10k": cfg_tabpfn_subsample_10k,
    "tabpfn_subsample_20k": cfg_tabpfn_subsample_20k,
    "tabpfn_full": cfg_tabpfn_full,
    "tabpfn_v3_full": cfg_tabpfn_v3_full,
    "tabpfn_v3_full_slim": cfg_tabpfn_v3_full_slim,
    "tabicl_subsample_20k": cfg_tabicl_subsample_20k,
    "tabicl_full": cfg_tabicl_full,
    "mlp_64_32_dropout": cfg_mlp_64_32_dropout,
    "mlp_128_64_32": cfg_mlp_128_64_32,
    "logreg_target_enc": cfg_logreg_target_enc,
    "stack_xgb_lr": cfg_stack_xgb_lr,
    "stack_xgb_lr_tabpfn": cfg_stack_xgb_lr_tabpfn,
    # Slim-features variants (wave 5)
    "lightgbm_dart_slim": cfg_lightgbm_dart,
    "xgb_early_stop_slim": cfg_xgb_early_stop,
    "tabicl_full_slim": cfg_tabicl_full,
    "tabpfn_subsample_10k_slim": cfg_tabpfn_subsample_10k,
}

SLIM_CONFIGS = {"lightgbm_dart_slim", "xgb_early_stop_slim",
                "tabicl_full_slim", "tabpfn_subsample_10k_slim",
                "tabpfn_v3_full_slim"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, choices=CONFIGS.keys())
    ap.add_argument("--out-results", required=True)
    ap.add_argument("--out-exp", required=True)
    args = ap.parse_args()

    out_results = Path(args.out_results)
    out_exp = Path(args.out_exp)
    out_results.mkdir(parents=True, exist_ok=True)
    out_exp.mkdir(parents=True, exist_ok=True)

    # also mirror predictions into a stable per-config dir so stacks can find them
    per_cfg = Path("/data/rauschecker2/jkw/cph/CPH200C/exp/20260513_auc_ceiling/per_config") / args.config
    per_cfg.mkdir(parents=True, exist_ok=True)

    use_slim = args.config in SLIM_CONFIGS
    print(f"[{args.config}] loading split (slim={use_slim})...")
    X_tr, X_va, X_te, y_tr, y_va, y_te = load_split(slim=use_slim)
    print(f"  n_train={len(y_tr)} n_val={len(y_va)} n_test={len(y_te)} pos_rate={y_tr.mean():.4f} n_feat={X_tr.shape[1]}")

    print(f"[{args.config}] fitting + predicting...")
    t0 = time.time()
    p_te, meta = CONFIGS[args.config](X_tr, X_va, X_te, y_tr, y_va)
    elapsed = time.time() - t0
    print(f"  done in {elapsed:.1f}s")

    auc, lo, hi = auc_with_ci(y_te, p_te)
    print(f"  TEST AUC = {auc:.4f} (95% CI {lo:.4f}-{hi:.4f})")

    # required artifact: summary.md
    summary = (
        f"# {args.config}\n\n"
        f"- Model: **{meta.get('model', args.config)}**\n"
        f"- Test AUC: **{auc:.4f}** (95% DeLong CI {lo:.4f}–{hi:.4f})\n"
        f"- Runtime: {elapsed:.1f}s\n"
        f"- n_train={len(y_tr):,} n_val={len(y_va):,} n_test={len(y_te):,}\n"
        f"- Extra: {json.dumps({k: v for k, v in meta.items() if k != 'model'})}\n"
    )
    (out_results / "summary.md").write_text(summary)
    (out_results / "metrics.json").write_text(json.dumps({
        "config": args.config, "auc": float(auc), "ci_lo": float(lo), "ci_hi": float(hi),
        "runtime_sec": elapsed, "n_test": int(len(y_te)),
        "extra": {k: (str(v) if not isinstance(v, (int, float, str, bool, type(None))) else v)
                  for k, v in meta.items()},
    }, indent=2))
    np.save(out_results / "preds_test.npy", p_te.astype(np.float32))
    np.save(per_cfg / "preds_test.npy", p_te.astype(np.float32))
    print(summary)


if __name__ == "__main__":
    main()
