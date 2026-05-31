"""Regenerate the forest plot (all candidates) + Champion ROC for the verified best.

Champion = CatBoost (raw ICD-9), AUC 0.7037 (95% CI 0.685-0.722, disk-roundtrip verified).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score

REPO = Path("/data/rauschecker2/jkw/cph/CPH200C")
sys.path.insert(0, "/home/jiwei/arcadia/superstack/skills/_shared")
sys.path.insert(0, str(REPO / "src"))
from mpl_style import BLUEVIOLET, DEEPPINK, GOLD, TURQUOISE, apply_style, palette, title

EXP = REPO / "exp/20260513_auc_ceiling"
SEC1_FIGS = REPO / "exp/20260512_pset2_full/figs"
NEW_FIGS = EXP / "figs"
NEW_FIGS.mkdir(parents=True, exist_ok=True)


def load_all_metrics():
    """Walk both section directories and return a list of (config, auc, ci_lo, ci_hi)."""
    rows = []
    # autoresearch sweep
    for p in sorted((EXP / "results").glob("*/metrics.json")):
        m = json.loads(p.read_text())
        rows.append({"config": m["config"], "auc": m["auc"],
                     "ci_lo": m["ci_lo"], "ci_hi": m["ci_hi"]})
    # section 1.2 (original four)
    sec1_path = REPO / "exp/20260512_pset2_full/02_models/results.json"
    if sec1_path.exists():
        sec1 = json.loads(sec1_path.read_text())
        # already represented if duplicate names exist in autoresearch — skip dupes
        existing = {r["config"] for r in rows}
        for name, v in sec1.items():
            slug = name.replace(" ", "_").replace("(", "").replace(")", "").replace("+", "plus")
            if slug not in existing and name not in existing:
                rows.append({"config": name, "auc": v["mean_auc"],
                             "ci_lo": v["ci_lo"], "ci_hi": v["ci_hi"]})
    df = pd.DataFrame(rows).sort_values("auc", ascending=True).reset_index(drop=True)
    return df


def forest_plot_all():
    apply_style()
    df = load_all_metrics()
    print(f"plotting {len(df)} candidates")
    names = df["config"].tolist()
    means = df["auc"].values
    los = df["ci_lo"].values
    his = df["ci_hi"].values
    y = np.arange(len(df))

    # Color: highlight the top model in turquoise, runner-up deeppink, others muted gray.
    colors = ["lightgray"] * len(df)
    colors[-1] = TURQUOISE   # best
    if len(df) >= 2:
        colors[-2] = DEEPPINK  # runner-up
    if len(df) >= 3:
        colors[-3] = GOLD      # 3rd

    fig, ax = plt.subplots(figsize=(9.5, 0.32 * len(df) + 1.5))
    for i, (m_, lo, hi, c) in enumerate(zip(means, los, his, colors)):
        ax.plot([lo, hi], [i, i], color=c, lw=3, solid_capstyle="round", zorder=2)
        ax.plot(m_, i, marker="o", color=c, markersize=7, markeredgecolor="black",
                markeredgewidth=0.6, zorder=3)
        ax.text(hi + 0.002, i, f"{m_:.4f}", va="center", fontsize=8.5, color="black",
                fontfamily=["Geist Mono", "DejaVu Sans Mono"])
    ax.axvline(0.5, color="gray", ls=":", alpha=0.5)
    ax.axvline(0.700, color="dimgray", ls="--", alpha=0.6, lw=1)
    ax.text(0.700, len(df) - 0.5, "  target 0.70", color="dimgray",
            fontsize=8, ha="left", va="top",
            fontfamily=["Geist Mono", "DejaVu Sans Mono"])
    ax.set_yticks(y)
    ax.set_yticklabels(names, color="black",
                       fontfamily=["Geist Mono", "DejaVu Sans Mono"], fontsize=9)
    ax.set_xlabel("Held-out test AUC (95% DeLong CI)")
    ax.set_xlim(0.62, his.max() + 0.025)
    title(ax, "All candidates — test AUC with 95% CI")
    fig.tight_layout()
    fig.savefig(NEW_FIGS / "forest_all.png", dpi=200)
    fig.savefig(NEW_FIGS / "forest_all.pdf")
    plt.close(fig)
    # Also copy into the section-1 figs dir so report.md figures can reference it.
    fig.savefig(SEC1_FIGS / "02_models_forest_all.png", dpi=200)
    print(f"wrote forest plot ({len(df)} rows)")


def bootstrap_roc(y_true, y_score, n_boot=2000, seed=42, fpr_grid=None):
    if fpr_grid is None:
        fpr_grid = np.linspace(0, 1, 101)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    tprs, aucs = [], []
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
    return (fpr_grid, tprs.mean(axis=0),
            np.quantile(tprs, 0.025, axis=0), np.quantile(tprs, 0.975, axis=0),
            roc_auc_score(y_true, y_score),
            float(np.quantile(aucs, 0.025)), float(np.quantile(aucs, 0.975)))


def _plot_roc(preds_path: Path, y_te_path: Path, title_text: str, out_stem: str):
    """Shared ROC plotter. Uses the saved y_te_aligned.npy as the source of truth
    (never reconstruct from train_test_split — see ALIGNMENT note in run_candidate.py).
    """
    apply_style()
    p = np.load(preds_path)
    y_te = np.load(y_te_path)
    fpr_g, mean_tpr, lo_tpr, hi_tpr, auc, lo, hi = bootstrap_roc(y_te, p)
    fpr_obs, tpr_obs, _ = roc_curve(y_te, p)
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.fill_between(fpr_g, lo_tpr, hi_tpr, color=TURQUOISE, alpha=0.22, linewidth=0)
    ax.plot(fpr_obs, tpr_obs, color=TURQUOISE, lw=2.0, zorder=3)
    ax.plot([0, 1], [0, 1], color="gray", lw=1.0, ls="--", alpha=0.7, zorder=1)
    ax.text(0.98, 0.04, f"AUC = {auc:.3f}\n95% CI ({lo:.3f}–{hi:.3f})",
            ha="right", va="bottom", fontsize=10, color="black",
            fontfamily=["Geist Mono", "DejaVu Sans Mono"])
    ax.set_xlim(-0.005, 1.005); ax.set_ylim(-0.005, 1.005)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_aspect("equal", adjustable="box")
    title(ax, title_text)
    fig.tight_layout()
    fig.savefig(NEW_FIGS / f"{out_stem}.png", dpi=200)
    fig.savefig(NEW_FIGS / f"{out_stem}.pdf")
    fig.savefig(SEC1_FIGS / f"{out_stem}.png", dpi=200)
    plt.close(fig)
    print(f"wrote {out_stem}: AUC={auc:.4f}")


def champion_roc_catboost():
    cb_dir = EXP / "results/catboost_raw_icd9"
    _plot_roc(cb_dir / "preds_test.npy", cb_dir / "y_te_aligned.npy",
              "CatBoost + raw ICD-9", "roc_catboost_raw_icd9")
    return  # original body replaced; legacy stub below for compat
    apply_style()
    cb_dir = EXP / "results/catboost_raw_icd9"
    p = np.load(cb_dir / "preds_test.npy")
    y_te = np.load(cb_dir / "y_te_aligned.npy")

    fpr_g, mean_tpr, lo_tpr, hi_tpr, auc, lo, hi = bootstrap_roc(y_te, p)
    fpr_obs, tpr_obs, _ = roc_curve(y_te, p)

    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.fill_between(fpr_g, lo_tpr, hi_tpr, color=TURQUOISE, alpha=0.22, linewidth=0)
    ax.plot(fpr_obs, tpr_obs, color=TURQUOISE, lw=2.0, zorder=3)
    ax.plot([0, 1], [0, 1], color="gray", lw=1.0, ls="--", alpha=0.7, zorder=1)
    ax.text(0.98, 0.04, f"AUC = {auc:.3f}\n95% CI ({lo:.3f}–{hi:.3f})",
            ha="right", va="bottom", fontsize=10, color="black",
            fontfamily=["Geist Mono", "DejaVu Sans Mono"])
    ax.set_xlim(-0.005, 1.005); ax.set_ylim(-0.005, 1.005)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_aspect("equal", adjustable="box")
    title(ax, "Champion model — CatBoost + raw ICD-9")
    fig.tight_layout()
    fig.savefig(NEW_FIGS / "roc_champion_catboost.png", dpi=200)
    fig.savefig(NEW_FIGS / "roc_champion_catboost.pdf")
    fig.savefig(SEC1_FIGS / "02_roc_champion_catboost.png", dpi=200)
    plt.close(fig)
    print(f"wrote champion ROC: AUC={auc:.4f}")


def roc_tabpfn_v3_full():
    """ROC for TabPFN-v3 full 49k (verified rerun). Uses verified preds + y_te."""
    v_dir = EXP / "results/tabpfn_v3_full_verified"
    _plot_roc(v_dir / "preds_test.npy", v_dir / "y_te_aligned.npy",
              "TabPFN-v3 (full 49k)", "roc_tabpfn_v3_full")


if __name__ == "__main__":
    forest_plot_all()
    champion_roc_catboost()
    roc_tabpfn_v3_full()
