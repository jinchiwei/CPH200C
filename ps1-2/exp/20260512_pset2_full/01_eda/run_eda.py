"""Section 1.1 -- 30-day readmission rate by age / gender / race + Wilson CIs.

Run with: /data/rauschecker1/jkw/envs/cph/bin/python run_eda.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportion_confint

REPO = Path("/data/rauschecker2/jkw/cph/CPH200C")
sys.path.insert(0, str(REPO / "src"))
from preprocessing import clean_and_filter, load_diabetes  # noqa: E402
from styling import TURQUOISE, apply_style, title  # noqa: E402

OUT = REPO / "exp/20260512_pset2_full/01_eda"
FIGS = REPO / "exp/20260512_pset2_full/figs"


def rate_table(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Wilson-95 CI per category."""
    rows = []
    for cat, sub in df.groupby(col, observed=True):
        n = len(sub)
        k = int(sub["y_30d"].sum())
        rate = k / n
        lo, hi = proportion_confint(k, n, alpha=0.05, method="wilson")
        rows.append({col: cat, "n": n, "k": k, "rate": rate, "ci_lo": lo, "ci_hi": hi})
    out = pd.DataFrame(rows)
    # natural sort for age bins
    if col == "age":
        order = ["[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)",
                 "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"]
        out = out.set_index(col).reindex(order).reset_index()
    else:
        out = out.sort_values("rate", ascending=False).reset_index(drop=True)
    return out


def plot_rates(tab: pd.DataFrame, col: str, ax: plt.Axes, label: str) -> None:
    x = np.arange(len(tab))
    rates = tab["rate"].values
    err_lo = rates - tab["ci_lo"].values
    err_hi = tab["ci_hi"].values - rates
    # Single-variable bar -> single brand-priority color (turquoise).
    ax.bar(x, rates * 100, color=TURQUOISE, edgecolor="none", zorder=2)
    ax.errorbar(x, rates * 100,
                yerr=np.vstack([err_lo, err_hi]) * 100,
                fmt="none", ecolor="black", elinewidth=1.0, capsize=3, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in tab[col].values], rotation=30, ha="right")
    ax.set_ylabel("30-day readmission rate (%)")
    # annotate n above each bar (subtle gray, body font)
    for xi, r, n in zip(x, rates, tab["n"].values):
        ax.text(xi, r * 100 + 0.3, f"n={n:,}",
                ha="center", va="bottom", fontsize=7, color="dimgray")
    title(ax, label)


def main():
    apply_style()
    df = load_diabetes()
    df = clean_and_filter(df)
    print(f"n={len(df):,}  positive rate={df['y_30d'].mean():.4f}")

    summaries = {}
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, col, label in zip(
        axes,
        ["age", "gender", "race"],
        ["30-day readmission by age", "by gender", "by race"],
    ):
        tab = rate_table(df, col)
        summaries[col] = tab
        plot_rates(tab, col, ax, label)
        tab.to_csv(OUT / f"rates_by_{col}.csv", index=False)
        print(f"\n=== {col} ===")
        print(tab.to_string(index=False))

    fig.suptitle("Section 1.1 — 30-day readmission rates with 95% Wilson CIs",
                 fontfamily=["Geist Mono", "DejaVu Sans Mono"], weight="bold",
                 fontsize=12, color="black")
    fig.tight_layout()
    fig.savefig(FIGS / "01_eda_readmission_rates.png", dpi=200)
    fig.savefig(FIGS / "01_eda_readmission_rates.pdf")
    print(f"\nSaved -> {FIGS/'01_eda_readmission_rates.png'}")


if __name__ == "__main__":
    main()
