"""Section 1.4 sidebar: CatBoost with raw ICD-9 under the same <50 -> >=50 shift.

Mirrors run_shift.py's setup but uses the best-overall model (CatBoost + raw cats)
rather than the best-linear (LogReg L2). Produces in-distribution AND shifted AUCs
that can be reported alongside the linear-model results.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

REPO = Path("/data/rauschecker2/jkw/cph/CPH200C")
sys.path.insert(0, str(REPO / "src"))
from preprocessing import AGE_MIDPOINTS, DRUG_COLUMNS, clean_and_filter, load_diabetes
from cv import bootstrap_auc_ci, delong_auc_ci

SEED = 42
OUT = REPO / "exp/20260513_auc_ceiling/results/shift_catboost_under50"
OUT.mkdir(parents=True, exist_ok=True)

YOUNG_BINS = {"[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)"}


def build_catboost_features(df):
    df = df.copy()
    df["age_midpoint"] = df["age"].map(AGE_MIDPOINTS).astype(float)
    numeric_cols = [
        "age_midpoint", "time_in_hospital",
        "num_lab_procedures", "num_procedures", "num_medications",
        "number_outpatient", "number_emergency", "number_inpatient",
        "number_diagnoses",
    ]
    cat_cols = [
        "race", "gender", "age",
        "admission_type_id", "discharge_disposition_id", "admission_source_id",
        "payer_code", "medical_specialty",
        "max_glu_serum", "A1Cresult", "change", "diabetesMed",
        "diag_1", "diag_2", "diag_3",
    ] + DRUG_COLUMNS
    cat_cols = [c for c in cat_cols if c in df.columns]
    for c in cat_cols:
        df[c] = df[c].astype(str).fillna("?")
    feat = df[numeric_cols + cat_cols].copy()
    cat_idx = [feat.columns.get_loc(c) for c in cat_cols]
    return feat, df["y_30d"].astype(int), cat_idx


def main():
    df = clean_and_filter(load_diabetes()).reset_index(drop=True)
    is_young = df["age"].isin(YOUNG_BINS).values
    print(f"<50:  n={is_young.sum():,}  rate={df.loc[is_young, 'y_30d'].mean():.4f}")
    print(f">=50: n={(~is_young).sum():,}  rate={df.loc[~is_young, 'y_30d'].mean():.4f}")

    df_y = df[is_young].reset_index(drop=True)
    df_o = df[~is_young].reset_index(drop=True)

    X_y, y_y, cat_idx = build_catboost_features(df_y)
    X_o, y_o, _ = build_catboost_features(df_o)

    # <50 train/val/test
    X_ytr, X_yte, y_ytr, y_yte = train_test_split(
        X_y, y_y, test_size=0.20, stratify=y_y, random_state=SEED,
    )
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_ytr, y_ytr, test_size=0.15, stratify=y_ytr, random_state=SEED,
    )
    spw = (len(y_tr) - y_tr.sum()) / max(y_tr.sum(), 1)

    m = CatBoostClassifier(
        iterations=2000, depth=6, learning_rate=0.05,
        l2_leaf_reg=3.0, subsample=0.85,
        cat_features=cat_idx,
        scale_pos_weight=spw,
        eval_metric="AUC", od_type="Iter", od_wait=100,
        random_seed=SEED, verbose=0,
    )
    m.fit(X_tr, y_tr, eval_set=(X_va, y_va), use_best_model=True)
    print(f"best iter: {m.get_best_iteration()}")

    p_in = m.predict_proba(X_yte)[:, 1]
    p_out = m.predict_proba(X_o)[:, 1]

    auc_in, lo_in, hi_in = delong_auc_ci(y_yte.values, p_in)
    auc_out, lo_out, hi_out = delong_auc_ci(y_o.values, p_out)
    print(f"<50 test AUC  = {auc_in:.4f}  ({lo_in:.4f}-{hi_in:.4f})")
    print(f">=50 test AUC = {auc_out:.4f}  ({lo_out:.4f}-{hi_out:.4f})")
    print(f"DELTA AUC     = {auc_in - auc_out:+.4f}")

    summary = {
        "model": "CatBoost (raw ICD-9, native cats)",
        "n_young": int(is_young.sum()), "n_old": int((~is_young).sum()),
        "rate_young": float(y_y.mean()), "rate_old": float(y_o.mean()),
        "auc_young_test": auc_in, "ci_young_test": [lo_in, hi_in],
        "auc_old": auc_out, "ci_old": [lo_out, hi_out],
        "delta_auc": auc_in - auc_out,
        "best_iter": int(m.get_best_iteration()),
    }
    (OUT / "results.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"saved -> {OUT/'results.json'}")


if __name__ == "__main__":
    main()
