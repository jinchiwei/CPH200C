"""Section 1.2 -- train linear / tree / NN on diabetes readmission, AUC + CIs.

Three model families: LogReg (with and without A1c x diag_1 interactions),
XGBoost, MLP. Patient-grouped 5-fold CV + a held-out test set; DeLong 95% CIs.
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

import xgboost as xgb

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

REPO = Path("/data/rauschecker2/jkw/cph/CPH200C")
sys.path.insert(0, str(REPO / "src"))
from cv import delong_auc_ci  # noqa: E402
from preprocessing import build_features, clean_and_filter, load_diabetes  # noqa: E402
from styling import (  # noqa: E402
    BLUEVIOLET, DEEPPINK, GOLD, TURQUOISE, apply_style, title,
)

OUT = REPO / "exp/20260512_pset2_full/02_models"
FIGS = REPO / "exp/20260512_pset2_full/figs"

SEED = 42
N_FOLDS = 5


# ----- feature interactions -------------------------------------------------

def add_a1c_diag1_interactions(X: pd.DataFrame) -> pd.DataFrame:
    """Add pairwise A1Cresult × diag_1_bucket interaction terms.

    These columns exist as dummies in X (e.g. 'A1Cresult_>8', 'diag_1_bucket_Circulatory');
    we multiply matching pairs to capture SDG+14's interaction story.
    """
    a1c_cols = [c for c in X.columns if c.startswith("A1Cresult_")]
    dx_cols = [c for c in X.columns if c.startswith("diag_1_bucket_")]
    extra = {}
    for a in a1c_cols:
        for d in dx_cols:
            extra[f"{a}__x__{d}"] = X[a].values * X[d].values
    return pd.concat([X, pd.DataFrame(extra, index=X.index)], axis=1)


# ----- model factories ------------------------------------------------------

def make_logreg(penalty: str = "l2", C: float = 1.0):
    return LogisticRegression(
        penalty=penalty, C=C, solver="liblinear" if penalty == "l1" else "lbfgs",
        max_iter=400, class_weight="balanced", random_state=SEED,
    )


def make_xgb(n_estimators: int = 400, max_depth: int = 5, lr: float = 0.05,
             spw: float = 1.0):
    return xgb.XGBClassifier(
        n_estimators=n_estimators, max_depth=max_depth, learning_rate=lr,
        subsample=0.85, colsample_bytree=0.85, scale_pos_weight=spw,
        eval_metric="auc", tree_method="hist", n_jobs=-1, random_state=SEED,
        use_label_encoder=False,
    )


def make_mlp(hidden=(64, 32), alpha: float = 1e-3, lr: float = 1e-3):
    return MLPClassifier(
        hidden_layer_sizes=hidden, alpha=alpha, learning_rate_init=lr,
        max_iter=60, early_stopping=True, validation_fraction=0.1,
        random_state=SEED,
    )


# ----- per-model fit + predict --------------------------------------------

def fit_predict(model_name: str, model_factory, X_train, y_train, X_test, needs_scaling=False):
    Xtr, Xte = X_train, X_test
    if needs_scaling:
        scaler = StandardScaler(with_mean=False)  # sparse-friendly
        Xtr = scaler.fit_transform(X_train)
        Xte = scaler.transform(X_test)
    model = model_factory()
    t0 = time.time()
    model.fit(Xtr, y_train)
    t_fit = time.time() - t0
    proba = model.predict_proba(Xte)[:, 1]
    return proba, model, t_fit


# ----- hyperparameter pick (val-based) -------------------------------------

def pick_logreg(X_train, y_train, X_val, y_val):
    best = None
    for penalty in ["l1", "l2"]:
        for C in [0.01, 0.1, 1.0]:
            model = make_logreg(penalty=penalty, C=C).fit(X_train, y_train)
            p = model.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, p)
            if best is None or auc > best[0]:
                best = (auc, penalty, C)
    return {"penalty": best[1], "C": best[2], "val_auc": best[0]}


def pick_xgb(X_train, y_train, X_val, y_val, spw):
    best = None
    for max_depth in [4, 6]:
        for n_estimators in [300, 600]:
            for lr in [0.05, 0.1]:
                model = make_xgb(n_estimators=n_estimators, max_depth=max_depth,
                                 lr=lr, spw=spw).fit(X_train, y_train)
                p = model.predict_proba(X_val)[:, 1]
                auc = roc_auc_score(y_val, p)
                if best is None or auc > best[0]:
                    best = (auc, max_depth, n_estimators, lr)
    return {"max_depth": best[1], "n_estimators": best[2], "learning_rate": best[3], "val_auc": best[0]}


def pick_mlp(X_train, y_train, X_val, y_val):
    best = None
    for hidden in [(32,), (64, 32)]:
        for alpha in [1e-4, 1e-3]:
            model = make_mlp(hidden=hidden, alpha=alpha).fit(X_train, y_train)
            p = model.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, p)
            if best is None or auc > best[0]:
                best = (auc, hidden, alpha)
    return {"hidden": best[1], "alpha": best[2], "val_auc": best[0]}


# ----- CV evaluation -------------------------------------------------------

def cv_auc(model_factory, X, y, groups, needs_scaling=False) -> tuple[list, list]:
    """Return per-fold AUCs + their 95% DeLong CIs."""
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    aucs, cis = [], []
    for fold, (tr, te) in enumerate(sgkf.split(X, y, groups=groups)):
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        ytr, yte = y.iloc[tr], y.iloc[te]
        proba, _, _ = fit_predict("cv", model_factory, Xtr, ytr, Xte, needs_scaling=needs_scaling)
        try:
            auc, lo, hi = delong_auc_ci(yte.values, proba)
        except Exception:
            from cv import bootstrap_auc_ci
            auc, lo, hi = bootstrap_auc_ci(yte.values, proba)
        aucs.append(auc)
        cis.append((lo, hi))
        print(f"  fold {fold+1}/{N_FOLDS}: AUC={auc:.4f} (95% CI {lo:.4f}-{hi:.4f})")
    return aucs, cis


# ----- forest plot ---------------------------------------------------------

def forest_plot(results: dict, out_path: Path):
    """results: dict[model_name] -> dict with 'mean_auc', 'ci_lo', 'ci_hi'."""
    apply_style()
    names = list(results.keys())
    means = [results[n]["mean_auc"] for n in names]
    los = [results[n]["ci_lo"] for n in names]
    his = [results[n]["ci_hi"] for n in names]

    y = np.arange(len(names))
    # Brand priority: turquoise -> deeppink -> amber (use GOLD for solid fills) -> blueviolet
    palette = [TURQUOISE, DEEPPINK, GOLD, BLUEVIOLET][: len(names)]

    fig, ax = plt.subplots(figsize=(8.5, 0.55 * len(names) + 1.4))
    for i, (m, lo, hi, c) in enumerate(zip(means, los, his, palette)):
        ax.plot([lo, hi], [i, i], color=c, lw=3, solid_capstyle="round")
        ax.plot(m, i, marker="o", color=c, markersize=9, markeredgecolor="black",
                markeredgewidth=0.8, zorder=4)
        ax.text(hi + 0.003, i, f"{m:.3f} ({lo:.3f}–{hi:.3f})",
                va="center", fontsize=9, color="black",
                fontfamily=["Geist Mono", "DejaVu Sans Mono"])
    ax.axvline(0.5, color="gray", ls=":", alpha=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(names, color="black",
                       fontfamily=["Geist Mono", "DejaVu Sans Mono"])
    ax.invert_yaxis()
    ax.set_xlabel("AUC (held-out test set, 95% DeLong CI)")
    ax.set_xlim(0.45, max(his) + 0.05)
    title(ax, "Section 1.2 — Held-out test AUC by model")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    fig.savefig(out_path.with_suffix(".pdf"))
    print(f"Saved forest plot -> {out_path}")


# ----- main -----------------------------------------------------------------

def main():
    apply_style()
    print("Loading data...")
    df = load_diabetes()
    df = clean_and_filter(df)
    X, y, names, pids = build_features(df)
    X_rich = add_a1c_diag1_interactions(X)
    print(f"n={len(df):,}  d_base={X.shape[1]}  d_rich={X_rich.shape[1]}  pos_rate={y.mean():.4f}")

    # Single train/val/test split (group-aware) for the headline numbers
    # Since first-encounter-only -> 1 row/patient, simple stratified split is group-safe.
    X_tv, X_test, y_tv, y_test, pids_tv, _ = train_test_split(
        X, y, pids, test_size=0.15, stratify=y, random_state=SEED,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=0.15 / 0.85, stratify=y_tv, random_state=SEED,
    )
    X_rich_tv = X_rich.loc[X_tv.index]
    X_rich_train = X_rich.loc[X_train.index]
    X_rich_val = X_rich.loc[X_val.index]
    X_rich_test = X_rich.loc[X_test.index]

    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    spw = n_neg / max(n_pos, 1)
    print(f"train={len(X_train):,} val={len(X_val):,} test={len(X_test):,}  scale_pos_weight={spw:.2f}")

    results = {}

    # --- LogReg base (no interactions) ---
    print("\n[LogReg base] hyper search...")
    Xtr_lin = pd.get_dummies(X_train, dummy_na=False).astype(np.float32)
    Xva_lin = X_val[Xtr_lin.columns].astype(np.float32)
    Xte_lin = X_test[Xtr_lin.columns].astype(np.float32)
    Xall_lin = X[Xtr_lin.columns].astype(np.float32)
    # scale
    scaler = StandardScaler(with_mean=False)
    Xtr_lin_s = scaler.fit_transform(Xtr_lin)
    Xva_lin_s = scaler.transform(Xva_lin)
    Xte_lin_s = scaler.transform(Xte_lin)
    Xall_lin_s = scaler.transform(Xall_lin)
    hp_lr = pick_logreg(Xtr_lin_s, y_train, Xva_lin_s, y_val)
    print(f"  best: {hp_lr}")
    model_lr = make_logreg(penalty=hp_lr["penalty"], C=hp_lr["C"]).fit(Xtr_lin_s, y_train)
    p_te = model_lr.predict_proba(Xte_lin_s)[:, 1]
    auc, lo, hi = delong_auc_ci(y_test.values, p_te)
    print(f"  TEST AUC: {auc:.4f} (95% CI {lo:.4f}-{hi:.4f})")
    results["LogReg (base)"] = {"mean_auc": auc, "ci_lo": lo, "ci_hi": hi, "hparams": hp_lr}
    np.save(OUT / "preds_logreg_base.npy", p_te)

    # --- LogReg with interactions ---
    print("\n[LogReg + A1c×diag1] hyper search...")
    Xtr_rich = pd.get_dummies(X_rich_train, dummy_na=False).astype(np.float32)
    Xva_rich = X_rich_val[Xtr_rich.columns].astype(np.float32)
    Xte_rich = X_rich_test[Xtr_rich.columns].astype(np.float32)
    scaler_r = StandardScaler(with_mean=False)
    Xtr_rich_s = scaler_r.fit_transform(Xtr_rich)
    Xva_rich_s = scaler_r.transform(Xva_rich)
    Xte_rich_s = scaler_r.transform(Xte_rich)
    hp_lr2 = pick_logreg(Xtr_rich_s, y_train, Xva_rich_s, y_val)
    print(f"  best: {hp_lr2}")
    model_lr2 = make_logreg(penalty=hp_lr2["penalty"], C=hp_lr2["C"]).fit(Xtr_rich_s, y_train)
    p_te = model_lr2.predict_proba(Xte_rich_s)[:, 1]
    auc, lo, hi = delong_auc_ci(y_test.values, p_te)
    print(f"  TEST AUC: {auc:.4f} (95% CI {lo:.4f}-{hi:.4f})")
    results["LogReg (+A1c×Dx)"] = {"mean_auc": auc, "ci_lo": lo, "ci_hi": hi, "hparams": hp_lr2}
    np.save(OUT / "preds_logreg_interact.npy", p_te)
    # save the scaler + lin column set so 1.4 can re-fit cleanly
    joblib.dump({"scaler": scaler, "cols": list(Xtr_lin.columns), "model": model_lr,
                 "hp": hp_lr},
                OUT / "model_logreg_base.joblib")

    # --- XGBoost ---
    print("\n[XGBoost] hyper search...")
    hp_xgb = pick_xgb(X_train, y_train, X_val, y_val, spw=spw)
    print(f"  best: {hp_xgb}")
    model_xgb = make_xgb(
        n_estimators=hp_xgb["n_estimators"], max_depth=hp_xgb["max_depth"],
        lr=hp_xgb["learning_rate"], spw=spw,
    ).fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    p_te = model_xgb.predict_proba(X_test)[:, 1]
    auc, lo, hi = delong_auc_ci(y_test.values, p_te)
    print(f"  TEST AUC: {auc:.4f} (95% CI {lo:.4f}-{hi:.4f})")
    results["XGBoost"] = {"mean_auc": auc, "ci_lo": lo, "ci_hi": hi, "hparams": hp_xgb}
    np.save(OUT / "preds_xgb.npy", p_te)
    model_xgb.save_model(OUT / "model_xgb.json")

    # --- MLP ---
    print("\n[MLP] hyper search...")
    # MLP needs scaled input; reuse the L2-style scaler
    hp_mlp = pick_mlp(Xtr_lin_s.toarray() if hasattr(Xtr_lin_s, "toarray") else Xtr_lin_s,
                     y_train,
                     Xva_lin_s.toarray() if hasattr(Xva_lin_s, "toarray") else Xva_lin_s,
                     y_val)
    print(f"  best: {hp_mlp}")
    model_mlp = make_mlp(hidden=hp_mlp["hidden"], alpha=hp_mlp["alpha"]).fit(
        Xtr_lin_s.toarray() if hasattr(Xtr_lin_s, "toarray") else Xtr_lin_s, y_train,
    )
    p_te = model_mlp.predict_proba(
        Xte_lin_s.toarray() if hasattr(Xte_lin_s, "toarray") else Xte_lin_s,
    )[:, 1]
    auc, lo, hi = delong_auc_ci(y_test.values, p_te)
    print(f"  TEST AUC: {auc:.4f} (95% CI {lo:.4f}-{hi:.4f})")
    results["MLP"] = {"mean_auc": auc, "ci_lo": lo, "ci_hi": hi, "hparams": hp_mlp}
    np.save(OUT / "preds_mlp.npy", p_te)

    # Save test rows so 1.3 (LLM) can score the same patients
    test_index = X_test.index.tolist()
    pd.Series(test_index, name="row_idx").to_csv(OUT / "test_row_idx.csv", index=False)
    pd.Series(y_test.values, index=test_index, name="y_30d").to_csv(OUT / "test_labels.csv")

    # Persist results
    with open(OUT / "results.json", "w") as f:
        # ensure hparams are JSON-serializable
        ser = {}
        for k, v in results.items():
            v2 = dict(v)
            if "hparams" in v2 and "hidden" in v2["hparams"]:
                v2["hparams"]["hidden"] = list(v2["hparams"]["hidden"])
            ser[k] = v2
        json.dump(ser, f, indent=2)

    # Forest plot
    forest_plot(results, FIGS / "02_models_forest.png")

    # Pick best by test AUC
    best_name = max(results, key=lambda k: results[k]["mean_auc"])
    print(f"\n===== BEST MODEL: {best_name} | AUC = {results[best_name]['mean_auc']:.4f} =====")


if __name__ == "__main__":
    main()
