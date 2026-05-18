"""ROC plots with bootstrap CI bands.

Builds:
  - figs/roc_champion_tabpfn_v3.png : champion ROC (TabPFN-v3) with shaded 95% CI
  - figs/roc_llm_opus46.png         : 1x3 small multiples for Opus 4.6 × {structured, natural, abbreviated}
  - figs/roc_llm_o1.png             : 1x3 small multiples for o1, IF results are available

Styling: canonical mpl_style. ROC line turquoise, lighter turquoise CI band,
gray dashed y=x midline. Title Geist Mono BLACK. Body Geist BLACK.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

REPO = Path("/data/rauschecker2/jkw/cph/CPH200C")
sys.path.insert(0, "/home/jiwei/arcadia/superstack/skills/_shared")
sys.path.insert(0, str(REPO / "src"))

from mpl_style import DEEPPINK, TURQUOISE, apply_style, title

EXP = REPO / "exp/20260513_auc_ceiling"
FIGS = EXP / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

# Light-turquoise CI band: alpha-blend turquoise onto white
# (matplotlib will alpha it automatically with alpha=0.2)
CI_BAND_ALPHA = 0.22


def bootstrap_roc_ci(y_true: np.ndarray, y_score: np.ndarray,
                     n_boot: int = 2000, seed: int = 42, fpr_grid=None):
    """Return (fpr_grid, mean_tpr, lo_tpr, hi_tpr, auc_point, auc_lo, auc_hi)."""
    if fpr_grid is None:
        fpr_grid = np.linspace(0, 1, 101)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    tprs = []
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        fpr_b, tpr_b, _ = roc_curve(y_true[idx], y_score[idx])
        tpr_interp = np.interp(fpr_grid, fpr_b, tpr_b)
        tpr_interp[0] = 0.0
        tprs.append(tpr_interp)
        aucs.append(roc_auc_score(y_true[idx], y_score[idx]))
    tprs = np.array(tprs)
    mean_tpr = tprs.mean(axis=0)
    lo_tpr = np.quantile(tprs, 0.025, axis=0)
    hi_tpr = np.quantile(tprs, 0.975, axis=0)
    auc_point = roc_auc_score(y_true, y_score)
    auc_lo = float(np.quantile(aucs, 0.025))
    auc_hi = float(np.quantile(aucs, 0.975))
    return fpr_grid, mean_tpr, lo_tpr, hi_tpr, float(auc_point), auc_lo, auc_hi


def plot_single_roc(ax, y_true, y_score, label_title: str, color=TURQUOISE):
    fpr_g, mean_tpr, lo_tpr, hi_tpr, auc, lo, hi = bootstrap_roc_ci(
        np.asarray(y_true), np.asarray(y_score),
    )
    # observed ROC (not the bootstrap mean) — clinically expected
    fpr_obs, tpr_obs, _ = roc_curve(y_true, y_score)
    # CI band first (under the line)
    ax.fill_between(fpr_g, lo_tpr, hi_tpr, color=color, alpha=CI_BAND_ALPHA, linewidth=0)
    # ROC curve on top
    ax.plot(fpr_obs, tpr_obs, color=color, lw=2.0, zorder=3)
    # diagonal chance line
    ax.plot([0, 1], [0, 1], color="gray", lw=1.0, ls="--", alpha=0.7, zorder=1)
    # annotation
    ax.text(0.98, 0.04, f"AUC = {auc:.3f}\n95% CI ({lo:.3f}–{hi:.3f})",
            ha="right", va="bottom", fontsize=9.5, color="black",
            fontfamily=["Geist Mono", "DejaVu Sans Mono"])
    ax.set_xlim(-0.005, 1.005)
    ax.set_ylim(-0.005, 1.005)
    ax.set_xlabel("False Positive Rate", color="black")
    ax.set_ylabel("True Positive Rate", color="black")
    ax.set_aspect("equal", adjustable="box")
    title(ax, label_title)


def champion_roc_tabpfn_v3():
    apply_style()
    # TabPFN-v3 runner-up. The autoresearch sweep produced its own test split
    # (slightly different prevalence than section 1.2's saved test_labels.csv) —
    # use the SAVED y_te_aligned.npy from the verify run so labels match the preds.
    pred_path = EXP / "results/tabpfn_v3_full_verified/preds_test.npy"
    y_path = EXP / "results/tabpfn_v3_full_verified/y_te_aligned.npy"
    if not pred_path.exists() or not y_path.exists():
        # fall back to the original run's preds; recover y_te by replaying load_split
        pred_path = EXP / "results/tabpfn_v3_full/preds_test.npy"
        p_te = np.load(pred_path)
        from run_candidate import load_split  # type: ignore
        _, _, _, _, _, y_te = load_split()
        y_test = y_te.values
    else:
        p_te = np.load(pred_path)
        y_test = np.load(y_path)
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    # TabPFN-v3 is the runner-up — color deeppink per palette priority (turquoise reserved for the CatBoost champion).
    plot_single_roc(ax, y_test, p_te, "TabPFN-v3 (full 49k)", color=DEEPPINK)
    fig.tight_layout()
    fig.savefig(FIGS / "roc_champion_tabpfn_v3.png", dpi=200)
    fig.savefig(FIGS / "roc_champion_tabpfn_v3.pdf")
    plt.close(fig)
    print(f"wrote {FIGS / 'roc_champion_tabpfn_v3.png'}")


def llm_roc_grid(cache_path: Path, model_label: str, out_stem: str):
    """1×3 small multiples for {structured, natural, abbreviated}."""
    if not cache_path.exists():
        print(f"missing cache: {cache_path}")
        return
    apply_style()
    cache = pd.read_csv(cache_path)
    # Cross-reference with label
    sample_idx = pd.read_csv(REPO / "exp/20260512_pset2_full/03_llm/sample_row_idx.csv")["row_idx"].tolist()
    y_test = pd.read_csv(REPO / "exp/20260512_pset2_full/02_models/test_labels.csv",
                        index_col=0)["y_30d"]
    fmts = ["structured", "natural", "abbreviated"]
    fig, axes = plt.subplots(1, 3, figsize=(15.6, 5.2))
    any_plotted = False
    for ax, fmt in zip(axes, fmts):
        sub = cache[(cache["prompt_format"] == fmt) & (cache["row_id"].isin(sample_idx))].copy()
        valid = sub["parsed_p"].notna()
        n_valid = int(valid.sum())
        if n_valid < 30:
            ax.text(0.5, 0.5, f"{fmt}\n(only {n_valid} parsed responses)",
                    ha="center", va="center", color="dimgray")
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            title(ax, fmt)
            continue
        y = y_test.loc[sub.loc[valid, "row_id"]].values
        p = sub.loc[valid, "parsed_p"].astype(float).values
        plot_single_roc(ax, y, p, f"{fmt}  (n={n_valid})")
        any_plotted = True
    fig.suptitle(f"{model_label} — ROC per prompt format (bootstrap 95% CI)",
                 fontfamily=["Geist Mono", "DejaVu Sans Mono"], weight="bold",
                 fontsize=12, color="black")
    fig.tight_layout()
    out = FIGS / f"roc_{out_stem}.png"
    fig.savefig(out, dpi=200)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    if any_plotted:
        print(f"wrote {out}")
    else:
        print(f"wrote {out} (no valid responses parsed yet)")


if __name__ == "__main__":
    champion_roc_tabpfn_v3()
    llm_roc_grid(
        cache_path=REPO / "exp/20260512_pset2_full/03_llm/versa_cache.csv",
        model_label="Claude Opus 4.6", out_stem="llm_opus46",
    )
    llm_roc_grid(
        cache_path=EXP / "llm_o1/versa_cache_o1.csv",
        model_label="GPT o1 (2024-12-17)", out_stem="llm_o1",
    )
