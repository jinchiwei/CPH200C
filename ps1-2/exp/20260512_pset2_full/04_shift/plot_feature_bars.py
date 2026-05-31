"""Section 1.4 — horizontal bar plot of the <50 LogReg L2 model's coefficients.

Top-N features by absolute coefficient. Color by sign:
    turquoise = negative β  (lowers 30-day readmission risk)
    deeppink  = positive β  (raises 30-day readmission risk)

Output: figs/04_shift_feature_importances.png + .pdf
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path("/data/rauschecker2/jkw/cph/CPH200C")
sys.path.insert(0, "/home/jiwei/arcadia/superstack/skills/_shared")
from mpl_style import DEEPPINK, TURQUOISE, apply_style, title  # noqa: E402

OUT = REPO / "exp/20260512_pset2_full/04_shift"
FIGS = REPO / "exp/20260512_pset2_full/figs"
TOP_N = 12  # 12 per direction (24 bars total)


def _pretty(name: str) -> str:
    """Clean up sklearn-mangled feature names for the plot labels."""
    n = name.replace("_", " ")
    # specific tidies
    fix = {
        "diag 1 bucket": "diag_1 bucket",
        "diag 2 bucket": "diag_2 bucket",
        "diag 3 bucket": "diag_3 bucket",
        "max glu serum": "max_glu_serum",
        "A1Cresult": "A1Cresult",
    }
    for k, v in fix.items():
        n = n.replace(k, v)
    return n


def main():
    apply_style()
    d = joblib.load(OUT / "model_lin_under50.joblib")
    coefs = pd.Series(d["model"].coef_[0], index=d["cols"]).sort_values()

    top_neg = coefs.head(TOP_N)       # most negative (turquoise — risk DOWN)
    top_pos = coefs.tail(TOP_N)[::-1] # most positive (deeppink — risk UP), reversed so largest first
    sel = pd.concat([top_neg, top_pos[::-1]])  # bottom -> top: most negative … most positive

    fig, ax = plt.subplots(figsize=(8.8, 7.6))
    colors = [TURQUOISE if v < 0 else DEEPPINK for v in sel.values]
    y = np.arange(len(sel))
    ax.barh(y, sel.values, color=colors, edgecolor="none", zorder=2)
    ax.axvline(0, color="black", lw=0.8, zorder=3)

    ax.set_yticks(y)
    ax.set_yticklabels([_pretty(n) for n in sel.index],
                       fontsize=9, color="black")
    ax.set_xlabel("Logistic-regression coefficient β  (log-odds per unit feature change)",
                  color="black")

    # subtle reference lines at +/- 0.1 and +/- 0.2
    for v in (-0.2, -0.1, 0.1, 0.2):
        ax.axvline(v, color="gray", ls=":", lw=0.5, alpha=0.5, zorder=1)

    # annotate each bar with β value
    for yi, v in zip(y, sel.values):
        ha = "left" if v >= 0 else "right"
        x_off = 0.005 if v >= 0 else -0.005
        ax.text(v + x_off, yi, f"{v:+.3f}",
                ha=ha, va="center", fontsize=8, color="black",
                fontfamily=["Geist Mono", "DejaVu Sans Mono"])

    title(ax, "Section 1.4 — Top features in the <50 LogReg L2 model")
    # subtitle with legend-style hint
    ax.text(0.99, -0.13, "turquoise = lowers 30-day risk   |   deeppink = raises 30-day risk",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            color="black", fontstyle="italic")
    ax.set_xlim(min(sel.values) * 1.18, max(sel.values) * 1.18)
    fig.tight_layout()
    fig.savefig(FIGS / "04_shift_feature_importances.png", dpi=200)
    fig.savefig(FIGS / "04_shift_feature_importances.pdf")
    plt.close(fig)
    print(f"wrote {FIGS / '04_shift_feature_importances.png'}")
    print(f"\nTop-5 each direction:")
    print("UP (deeppink):", top_pos.head().to_dict())
    print("DOWN (turquoise):", top_neg.head().to_dict())


if __name__ == "__main__":
    main()
