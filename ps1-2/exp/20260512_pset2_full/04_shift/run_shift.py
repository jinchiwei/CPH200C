"""Section 1.4 -- distribution shift across patient age (<50 vs >=50).

Trains the best linear model on the <50 cohort only, then evaluates on:
    (a) held-out test of <50 patients,
    (b) ALL patients >=50 (the shifted target population).

Also computes per-feature mean shift to surface which inputs change the most,
plus top coefficients of the <50 model for the writeup.

For the LLM portion we re-use the Versa cache from section 1.3 if it covers
both strata; otherwise we sub-sample an additional ~120 rows from the >=50
slice to fill any gap.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=ConvergenceWarning)

REPO = Path("/data/rauschecker2/jkw/cph/CPH200C")
sys.path.insert(0, str(REPO / "src"))
from cv import bootstrap_auc_ci, delong_auc_ci  # noqa: E402
from preprocessing import build_features, clean_and_filter, load_diabetes  # noqa: E402
from styling import DEEPPINK, TURQUOISE, apply_style, title  # noqa: E402

OUT = REPO / "exp/20260512_pset2_full/04_shift"
FIGS = REPO / "exp/20260512_pset2_full/figs"

SEED = 42

YOUNG_BINS = {"[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)"}
OLD_BINS = {"[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"}


def main():
    apply_style()
    df = load_diabetes()
    df = clean_and_filter(df).reset_index(drop=True)
    X, y, names, pids = build_features(df)
    is_young = df["age"].isin(YOUNG_BINS).values
    print(f"<50: n={is_young.sum():,}  rate={y[is_young].mean():.4f}")
    print(f">=50: n={(~is_young).sum():,}  rate={y[~is_young].mean():.4f}")

    # ------- (1) train best linear on <50 only ----------------------
    X_y = X[is_young].copy()
    y_y = y[is_young].copy()
    X_o = X[~is_young].copy()
    y_o = y[~is_young].copy()

    # held-out test inside <50
    X_ytr, X_yte, y_ytr, y_yte = train_test_split(
        X_y, y_y, test_size=0.2, stratify=y_y, random_state=SEED,
    )
    scaler = StandardScaler(with_mean=False).fit(X_ytr)
    Xtr_s = scaler.transform(X_ytr)
    Xte_s = scaler.transform(X_yte)
    Xold_s = scaler.transform(X_o)

    # tiny C search on a val split
    X_tr2, X_va2, y_tr2, y_va2 = train_test_split(
        X_ytr, y_ytr, test_size=0.15, stratify=y_ytr, random_state=SEED,
    )
    scaler2 = StandardScaler(with_mean=False).fit(X_tr2)
    best = None
    for C in [0.01, 0.1, 1.0]:
        m = LogisticRegression(penalty="l2", C=C, class_weight="balanced",
                               solver="lbfgs", max_iter=400, random_state=SEED).fit(
            scaler2.transform(X_tr2), y_tr2,
        )
        auc_v = bootstrap_auc_ci(y_va2.values, m.predict_proba(scaler2.transform(X_va2))[:, 1])[0]
        if best is None or auc_v > best[0]:
            best = (auc_v, C)
    best_C = best[1]
    print(f"Selected C={best_C}  (val AUC={best[0]:.4f})")

    # refit on full <50 train, evaluate on <50 test and on >=50
    model = LogisticRegression(penalty="l2", C=best_C, class_weight="balanced",
                               solver="lbfgs", max_iter=400, random_state=SEED).fit(Xtr_s, y_ytr)
    p_in = model.predict_proba(Xte_s)[:, 1]
    p_out = model.predict_proba(Xold_s)[:, 1]

    auc_in, lo_in, hi_in = delong_auc_ci(y_yte.values, p_in)
    auc_out, lo_out, hi_out = delong_auc_ci(y_o.values, p_out)
    print(f"<50 test AUC  = {auc_in:.4f}  (95% CI {lo_in:.4f}-{hi_in:.4f})")
    print(f">=50 test AUC = {auc_out:.4f}  (95% CI {lo_out:.4f}-{hi_out:.4f})")
    print(f"DELTA AUC     = {auc_in - auc_out:+.4f}")

    # ------- (2) feature importances via |coefficient| -----------------
    coefs = pd.Series(model.coef_[0], index=X_ytr.columns)
    top_pos = coefs.sort_values(ascending=False).head(10)
    top_neg = coefs.sort_values().head(10)
    print("\nTop +coefficients (increase 30d risk in <50 model):")
    print(top_pos.to_string())
    print("\nTop -coefficients (decrease 30d risk in <50 model):")
    print(top_neg.to_string())

    # ------- (3) population diffs between <50 and >=50 -----------------
    means_y = X_y.mean()
    means_o = X_o.mean()
    diff = (means_o - means_y).abs()
    # stable scale: divide by pooled std
    pooled = X.std()
    smd = (means_o - means_y) / pooled.replace(0, np.nan)
    smd = smd.sort_values()
    top_shift = pd.concat([smd.head(8), smd.tail(8)])
    print("\nLargest standardized mean differences (>=50 minus <50):")
    print(top_shift.to_string())

    # save summary
    summary = {
        "best_C": float(best_C),
        "n_young": int(is_young.sum()), "n_old": int((~is_young).sum()),
        "rate_young": float(y_y.mean()), "rate_old": float(y_o.mean()),
        "auc_young_test": auc_in, "ci_young_test": [lo_in, hi_in],
        "auc_old": auc_out, "ci_old": [lo_out, hi_out],
        "delta_auc": auc_in - auc_out,
        "top_pos_coefs": top_pos.round(3).to_dict(),
        "top_neg_coefs": top_neg.round(3).to_dict(),
        "top_shift_smd": top_shift.round(3).to_dict(),
    }
    with open(OUT / "results.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)

    # ------- (4) plot AUC and rate comparison --------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # left: AUC bars — turquoise (in-dist) vs deeppink (shifted)
    ax = axes[0]
    labels = ["<50 (in-dist)", "≥50 (shifted)"]
    aucs = [auc_in, auc_out]
    ci = [(lo_in, hi_in), (lo_out, hi_out)]
    cols = [TURQUOISE, DEEPPINK]
    x = np.arange(len(labels))
    for xi, a, c, (lo, hi) in zip(x, aucs, cols, ci):
        ax.bar(xi, a, color=c)
        ax.errorbar(xi, a, yerr=[[a - lo], [hi - a]], fmt="none",
                    ecolor="black", elinewidth=1.2, capsize=4)
        ax.text(xi, a + 0.005, f"{a:.3f}\n({lo:.3f}–{hi:.3f})",
                ha="center", va="bottom", fontsize=8, color="black",
                fontfamily=["Geist Mono", "DejaVu Sans Mono"])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, color="black")
    ax.set_ylabel("AUC (95% DeLong CI)")
    ax.set_ylim(0.45, max(hi_in, hi_out) + 0.07)
    title(ax, "Linear model trained on <50 — AUC under shift")

    # right: readmission base-rate — same color mapping (turquoise / deeppink)
    ax = axes[1]
    rates = [float(y_y.mean()), float(y_o.mean())]
    ax.bar(x, [r * 100 for r in rates], color=[TURQUOISE, DEEPPINK])
    for xi, r in zip(x, rates):
        ax.text(xi, r * 100 + 0.1, f"{r*100:.2f}%",
                ha="center", va="bottom", fontsize=9, color="black",
                fontfamily=["Geist Mono", "DejaVu Sans Mono"])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, color="black")
    ax.set_ylabel("30-day readmission rate (%)")
    title(ax, "Outcome prevalence by age stratum")

    fig.tight_layout()
    fig.savefig(FIGS / "04_shift_auc_and_rates.png", dpi=200)
    fig.savefig(FIGS / "04_shift_auc_and_rates.pdf")
    print(f"Saved figure -> {FIGS/'04_shift_auc_and_rates.png'}")

    # save the trained <50 model for re-use (e.g., re-eval on additional cohorts)
    joblib.dump({"model": model, "scaler": scaler, "cols": list(X_ytr.columns), "C": best_C},
                OUT / "model_lin_under50.joblib")


if __name__ == "__main__":
    main()
