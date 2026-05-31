"""Four single-panel ROC plots in palette-priority colors.

Layout:
  - figs/roc_4panel_catboost.png    turquoise
  - figs/roc_4panel_tabpfn_v3.png   deeppink
  - figs/roc_4panel_opus46.png      amber (lines variant — same color story)
  - figs/roc_4panel_o1.png          blueviolet

Each ROC uses the model's best comparable evaluation set:
  - CatBoost / TabPFN-v3 : section 1.2 test set (n=10,499)
  - Opus 4.6             : the parallelized full-test-set cache if available, else n=300 sample
  - o1                   : n=300 sample (we're not re-running o1 per user instruction)
  - For LLMs, plot the BEST prompt format (abbreviated for both).
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
from cv import delong_auc_ci, bootstrap_auc_ci  # noqa: E402
from mpl_style import AMBER, BLUEVIOLET, DEEPPINK, TURQUOISE, apply_style, title  # noqa: E402

EXP = REPO / "exp/20260513_auc_ceiling"
FIGS = EXP / "figs"
CI_BAND_ALPHA = 0.22


def bootstrap_roc_ci(y_true, y_score, n_boot=2000, seed=42):
    fpr_grid = np.linspace(0, 1, 101)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    tprs, aucs = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        fpr_b, tpr_b, _ = roc_curve(y_true[idx], y_score[idx])
        tpr_interp = np.interp(fpr_grid, fpr_b, tpr_b); tpr_interp[0] = 0.0
        tprs.append(tpr_interp); aucs.append(roc_auc_score(y_true[idx], y_score[idx]))
    tprs = np.array(tprs)
    return (fpr_grid, np.quantile(tprs, 0.025, axis=0), np.quantile(tprs, 0.975, axis=0),
            float(roc_auc_score(y_true, y_score)),
            float(np.quantile(aucs, 0.025)), float(np.quantile(aucs, 0.975)))


def single_roc(out_path: Path, y_true, y_score, color: str, title_text: str, n_label: str | None = None):
    apply_style()
    fpr_g, lo_tpr, hi_tpr, auc, lo, hi = bootstrap_roc_ci(np.asarray(y_true), np.asarray(y_score))
    fpr_obs, tpr_obs, _ = roc_curve(y_true, y_score)
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.fill_between(fpr_g, lo_tpr, hi_tpr, color=color, alpha=CI_BAND_ALPHA, linewidth=0)
    ax.plot(fpr_obs, tpr_obs, color=color, lw=2.0, zorder=3)
    ax.plot([0, 1], [0, 1], color="gray", lw=1.0, ls="--", alpha=0.7, zorder=1)
    ann = f"AUC = {auc:.3f}\n95% CI ({lo:.3f}–{hi:.3f})"
    if n_label:
        ann += f"\n{n_label}"
    ax.text(0.98, 0.04, ann, ha="right", va="bottom", fontsize=9.5, color="black",
            fontfamily=["Geist Mono", "DejaVu Sans Mono"])
    ax.set_xlim(-0.005, 1.005); ax.set_ylim(-0.005, 1.005)
    ax.set_xlabel("False Positive Rate", color="black")
    ax.set_ylabel("True Positive Rate", color="black")
    ax.set_aspect("equal", adjustable="box")
    title(ax, title_text)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
    print(f"wrote {out_path}  AUC={auc:.4f}")


def load_y_test_sec12():
    return pd.read_csv(REPO / "exp/20260512_pset2_full/02_models/test_labels.csv",
                       index_col=0)["y_30d"]


def catboost_roc():
    # Use the section-1.2-matched CatBoost preds (refit). Falls back to verified if missing.
    pred_candidates = [
        EXP / "results/catboost_raw_icd9_sec12/preds_test.npy",   # if we refit on sec1.2 split
        EXP / "results/catboost_raw_icd9/preds_test.npy",         # current saved (autoresearch cohort)
    ]
    label_candidates = [
        EXP / "results/catboost_raw_icd9_sec12/y_te_aligned.npy",
        EXP / "results/catboost_raw_icd9/y_te_aligned.npy",
    ]
    for pred_p, lab_p in zip(pred_candidates, label_candidates):
        if pred_p.exists() and lab_p.exists():
            print(f"[catboost] using preds={pred_p.name}  labels={lab_p.name}")
            p = np.load(pred_p)
            y = np.load(lab_p)
            single_roc(FIGS / "roc_4panel_catboost.png", y, p, TURQUOISE,
                       "CatBoost — raw ICD-9 + native categoricals",
                       n_label=f"n = {len(y):,}")
            return
    print("[catboost] no preds available")


def tabpfn_roc():
    pred_candidates = [
        EXP / "results/tabpfn_v3_full_sec12/preds_test.npy",
        EXP / "results/tabpfn_v3_full_verified/preds_test.npy",
        EXP / "results/tabpfn_v3_full/preds_test.npy",
    ]
    label_candidates = [
        EXP / "results/tabpfn_v3_full_sec12/y_te_aligned.npy",
        EXP / "results/tabpfn_v3_full_verified/y_te_aligned.npy",
        None,  # original run didn't save y_te_aligned
    ]
    for pred_p, lab_p in zip(pred_candidates, label_candidates):
        if pred_p.exists() and (lab_p is None or lab_p.exists()):
            print(f"[tabpfn-v3] using preds={pred_p.parent.name}/{pred_p.name}")
            p = np.load(pred_p)
            y = np.load(lab_p) if lab_p and lab_p.exists() else None
            if y is None:
                # last resort: load via current load_split (only valid if split is deterministic and matches)
                import sys as _sys
                _sys.path.insert(0, str(EXP))
                from run_candidate import load_split
                _, _, _, _, _, y_te = load_split()
                y = y_te.values
            single_roc(FIGS / "roc_4panel_tabpfn_v3.png", y, p, DEEPPINK,
                       "TabPFN-v3 (full 49k context)",
                       n_label=f"n = {len(y):,}")
            return
    print("[tabpfn-v3] no preds available")


def llm_roc(cache_path: Path, color: str, title_text: str, out_name: str,
            fmt: str = "abbreviated"):
    if not cache_path.exists():
        print(f"[{out_name}] missing cache: {cache_path}")
        return
    cache = pd.read_csv(cache_path)
    sub = cache[cache["prompt_format"] == fmt].copy()
    valid = sub["parsed_p"].notna()
    n_valid = int(valid.sum())
    if n_valid < 30:
        print(f"[{out_name}] only {n_valid} parsed responses — skipping")
        return
    y_test = load_y_test_sec12()
    y = y_test.loc[sub.loc[valid, "row_id"]].values
    p = sub.loc[valid, "parsed_p"].astype(float).values
    single_roc(FIGS / out_name, y, p, color, title_text,
               n_label=f"n = {n_valid:,}  (prompt: {fmt})")


def opus_roc():
    # Prefer full-test-set cache if the parallel eval has completed
    full_cache = REPO / "exp/20260512_pset2_full/03_llm/versa_cache_full.csv"
    sample_cache = REPO / "exp/20260512_pset2_full/03_llm/versa_cache.csv"
    cache = full_cache if full_cache.exists() else sample_cache
    print(f"[opus] using cache={cache.name}")
    llm_roc(cache, AMBER, "Claude Opus 4.6 (abbreviated prompt)",
            "roc_4panel_opus46.png")


def o1_roc():
    # Prefer the full-test cache if the parallel o1 eval has completed.
    full_cache = EXP / "llm_o1/versa_cache_o1_full.csv"
    sample_cache = EXP / "llm_o1/versa_cache_o1.csv"
    cache = full_cache if full_cache.exists() else sample_cache
    # At full scale o1's best format is `natural` (0.655), not abbreviated.
    # Use its best for the headline ROC, parallel to Opus's `abbreviated`.
    fmt = "natural" if cache == full_cache else "abbreviated"
    print(f"[o1] using cache={cache.name}  format={fmt}")
    llm_roc(cache, BLUEVIOLET, f"GPT o1-2024-12-17 ({fmt} prompt)",
            "roc_4panel_o1.png", fmt=fmt)


if __name__ == "__main__":
    catboost_roc()
    tabpfn_roc()
    opus_roc()
    o1_roc()
