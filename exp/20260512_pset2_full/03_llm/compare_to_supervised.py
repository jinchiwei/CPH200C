"""Score the supervised XGBoost predictions on the SAME 300-row sub-sample the
LLM was evaluated on. This makes the LLM-vs-supervised comparison apples-to-apples
(supervised's headline 0.691 was on 10,499 rows; the LLM was on 300).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/data/rauschecker2/jkw/cph/CPH200C")
sys.path.insert(0, str(REPO / "src"))
from cv import delong_auc_ci  # noqa: E402

EXP = REPO / "exp/20260512_pset2_full"


def main():
    # Load LLM sub-sample IDs
    sample_ids = pd.read_csv(EXP / "03_llm/sample_row_idx.csv")["row_idx"].tolist()
    # Load full test labels + predictions
    test_idx = pd.read_csv(EXP / "02_models/test_row_idx.csv")["row_idx"].tolist()
    y_test = pd.read_csv(EXP / "02_models/test_labels.csv", index_col=0)["y_30d"]

    # Load XGB and LogReg test predictions, aligned with test_idx order
    p_xgb = np.load(EXP / "02_models/preds_xgb.npy")
    p_lr = np.load(EXP / "02_models/preds_logreg_base.npy")
    p_lr2 = np.load(EXP / "02_models/preds_logreg_interact.npy")
    p_mlp = np.load(EXP / "02_models/preds_mlp.npy")
    test_idx = np.array(test_idx)

    # Build a lookup of test_idx -> prediction position
    pos = {rid: i for i, rid in enumerate(test_idx)}
    keep = [pos[rid] for rid in sample_ids]
    y_sub = y_test.loc[sample_ids].values
    preds = {"XGBoost": p_xgb[keep], "LogReg (base)": p_lr[keep],
             "LogReg (+A1c×Dx)": p_lr2[keep], "MLP": p_mlp[keep]}

    out = {}
    for name, p in preds.items():
        try:
            auc, lo, hi = delong_auc_ci(y_sub, p)
        except Exception:
            from cv import bootstrap_auc_ci
            auc, lo, hi = bootstrap_auc_ci(y_sub, p)
        out[name] = {"auc": float(auc), "ci_lo": float(lo), "ci_hi": float(hi), "n": len(y_sub)}
        print(f"{name:20s} AUC={auc:.4f} (95% CI {lo:.4f}-{hi:.4f}) on n={len(y_sub)}")

    with open(EXP / "03_llm/supervised_on_llm_sample.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved:", EXP / "03_llm/supervised_on_llm_sample.json")


if __name__ == "__main__":
    main()
