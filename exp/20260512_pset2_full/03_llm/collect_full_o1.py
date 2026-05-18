"""Merge the o1 chunk caches into one full-test result.

Reads exp/20260513_auc_ceiling/llm_o1/chunks/versa_cache_chunk_NNN.csv, computes
AUC + CI per format, writes results_full_o1.json + merged cache.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/data/rauschecker2/jkw/cph/CPH200C")
sys.path.insert(0, str(REPO / "src"))
from cv import bootstrap_auc_ci, delong_auc_ci  # noqa: E402

OUT = REPO / "exp/20260513_auc_ceiling/llm_o1"
CHUNK_DIR = OUT / "chunks"


def main():
    chunks = sorted(CHUNK_DIR.glob("versa_cache_chunk_*.csv"))
    if not chunks:
        print("no chunk caches found", file=sys.stderr)
        return 1
    cache = pd.concat([pd.read_csv(p) for p in chunks], ignore_index=True)
    cache = cache.drop_duplicates(subset=["row_id", "prompt_format"], keep="last")
    print(f"merged {len(chunks)} chunk caches -> {len(cache)} entries")

    out_csv = OUT / "versa_cache_o1_full.csv"
    cache.to_csv(out_csv, index=False)
    print(f"wrote {out_csv}")

    test_idx = pd.read_csv(REPO / "exp/20260512_pset2_full/02_models/test_row_idx.csv")["row_idx"].tolist()
    y_test = pd.read_csv(REPO / "exp/20260512_pset2_full/02_models/test_labels.csv", index_col=0)["y_30d"]
    test_idx = sorted(test_idx)

    results = {}
    for fmt in ["structured", "natural", "abbreviated"]:
        sub = cache[(cache["prompt_format"] == fmt) & (cache["row_id"].isin(test_idx))].copy()
        valid = sub["parsed_p"].notna()
        n_valid = int(valid.sum())
        coverage = n_valid / len(test_idx)
        if n_valid < 100:
            print(f"{fmt}: only {n_valid} parsed — skipping AUC")
            results[fmt] = {"auc": None, "n_valid": n_valid, "coverage": coverage}
            continue
        y = y_test.loc[sub.loc[valid, "row_id"]].values
        p = sub.loc[valid, "parsed_p"].astype(float).values
        try:
            auc, lo, hi = delong_auc_ci(y, p)
        except Exception as e:
            print(f"  DeLong failed ({e}); using bootstrap")
            auc, lo, hi = bootstrap_auc_ci(y, p, n_boot=1000)
        print(f"{fmt:12s}  AUC={auc:.4f}  (95% CI {lo:.4f}-{hi:.4f})  n_valid={n_valid}  coverage={coverage:.3f}")
        results[fmt] = {"auc": float(auc), "ci_lo": float(lo), "ci_hi": float(hi),
                        "n_valid": n_valid, "coverage": coverage}

    with open(OUT / "results_full_o1.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {OUT / 'results_full_o1.json'}")


if __name__ == "__main__":
    sys.exit(main() or 0)
