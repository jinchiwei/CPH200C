"""Re-render the three figures from the saved data, without retraining anything.

After fixing styling in src/styling.py, this regenerates the PNGs/PDFs only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path("/data/rauschecker2/jkw/cph/CPH200C")
sys.path.insert(0, str(REPO / "src"))
from styling import BLUEVIOLET, DEEPPINK, GOLD, TURQUOISE, apply_style, title  # noqa
# Per-panel colors for the §1.1 EDA. Priority order: turquoise → deeppink → gold (solid-fill swap for amber) → blueviolet.
EDA_PANEL_COLORS = {"age": TURQUOISE, "gender": DEEPPINK, "race": GOLD}

EXP = REPO / "exp/20260512_pset2_full"
FIGS = EXP / "figs"


def plot_eda():
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, col, label in zip(
        axes,
        ["age", "gender", "race"],
        ["30-day readmission by age", "by gender", "by race"],
    ):
        tab = pd.read_csv(EXP / f"01_eda/rates_by_{col}.csv")
        # Drop tiny-n cells that blow up the y-axis (Wilson CIs become huge with n<10).
        tab = tab[tab["n"] >= 30].reset_index(drop=True)
        x = np.arange(len(tab))
        rates = tab["rate"].values
        err_lo = rates - tab["ci_lo"].values
        err_hi = tab["ci_hi"].values - rates
        ax.bar(x, rates * 100, color=EDA_PANEL_COLORS[col], edgecolor="none", zorder=2)
        ax.errorbar(x, rates * 100,
                    yerr=np.vstack([err_lo, err_hi]) * 100,
                    fmt="none", ecolor="black", elinewidth=1.0, capsize=3, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels([str(v) for v in tab[col].values], rotation=30, ha="right")
        ax.set_ylabel("30-day readmission rate (%)")
        for xi, r, n in zip(x, rates, tab["n"].values):
            ax.text(xi, r * 100 + 0.3, f"n={n:,}",
                    ha="center", va="bottom", fontsize=7, color="dimgray")
        title(ax, label)
    fig.suptitle("Section 1.1 — 30-day readmission rates with 95% Wilson CIs",
                 fontfamily=["Geist Mono", "DejaVu Sans Mono"], weight="bold",
                 fontsize=12, color="black")
    fig.tight_layout()
    fig.savefig(FIGS / "01_eda_readmission_rates.png", dpi=200)
    fig.savefig(FIGS / "01_eda_readmission_rates.pdf")
    plt.close(fig)
    print(f"wrote {FIGS / '01_eda_readmission_rates.png'}")


def plot_forest():
    apply_style()
    with open(EXP / "02_models/results.json") as f:
        results = json.load(f)
    names = list(results.keys())
    means = [results[n]["mean_auc"] for n in names]
    los = [results[n]["ci_lo"] for n in names]
    his = [results[n]["ci_hi"] for n in names]
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
    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels(names, color="black",
                       fontfamily=["Geist Mono", "DejaVu Sans Mono"])
    ax.invert_yaxis()
    ax.set_xlabel("AUC (held-out test set, 95% DeLong CI)")
    ax.set_xlim(0.45, max(his) + 0.05)
    title(ax, "Section 1.2 — Held-out test AUC by model")
    fig.tight_layout()
    fig.savefig(FIGS / "02_models_forest.png", dpi=200)
    fig.savefig(FIGS / "02_models_forest.pdf")
    plt.close(fig)
    print(f"wrote {FIGS / '02_models_forest.png'}")


def plot_shift():
    apply_style()
    with open(EXP / "04_shift/results.json") as f:
        s = json.load(f)
    auc_in = s["auc_young_test"]; lo_in, hi_in = s["ci_young_test"]
    auc_out = s["auc_old"]; lo_out, hi_out = s["ci_old"]
    rate_y = s["rate_young"]; rate_o = s["rate_old"]
    labels = ["<50 (in-dist)", "≥50 (shifted)"]
    x = np.arange(2)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    for xi, a, c, (lo, hi) in zip(x, [auc_in, auc_out], [TURQUOISE, DEEPPINK],
                                  [(lo_in, hi_in), (lo_out, hi_out)]):
        ax.bar(xi, a, color=c)
        ax.errorbar(xi, a, yerr=[[a - lo], [hi - a]], fmt="none",
                    ecolor="black", elinewidth=1.2, capsize=4)
        ax.text(xi, a + 0.005, f"{a:.3f}\n({lo:.3f}–{hi:.3f})",
                ha="center", va="bottom", fontsize=8, color="black",
                fontfamily=["Geist Mono", "DejaVu Sans Mono"])
    ax.set_xticks(x); ax.set_xticklabels(labels, color="black")
    ax.set_ylabel("AUC (95% DeLong CI)")
    ax.set_ylim(0.45, max(hi_in, hi_out) + 0.07)
    title(ax, "Linear model trained on <50 — AUC under shift")

    ax = axes[1]
    ax.bar(x, [rate_y * 100, rate_o * 100], color=[TURQUOISE, DEEPPINK])
    for xi, r in zip(x, [rate_y, rate_o]):
        ax.text(xi, r * 100 + 0.1, f"{r*100:.2f}%",
                ha="center", va="bottom", fontsize=9, color="black",
                fontfamily=["Geist Mono", "DejaVu Sans Mono"])
    ax.set_xticks(x); ax.set_xticklabels(labels, color="black")
    ax.set_ylabel("30-day readmission rate (%)")
    title(ax, "Outcome prevalence by age stratum")

    fig.tight_layout()
    fig.savefig(FIGS / "04_shift_auc_and_rates.png", dpi=200)
    fig.savefig(FIGS / "04_shift_auc_and_rates.pdf")
    plt.close(fig)
    print(f"wrote {FIGS / '04_shift_auc_and_rates.png'}")


if __name__ == "__main__":
    plot_eda()
    plot_forest()
    plot_shift()
