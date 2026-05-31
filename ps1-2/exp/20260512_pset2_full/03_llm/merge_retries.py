"""Merge retry chunk results into versa_cache_full.csv.

For each retry-row that now has a parsed_p (was an API error originally),
update the original cache. Recompute per-format AUC + DeLong CI on the
fixed full cache.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path("/data/rauschecker2/jkw/cph/CPH200C")
sys.path.insert(0, str(REPO / "src"))
from cv import bootstrap_auc_ci, delong_auc_ci  # noqa: E402

OUT = REPO / "exp/20260512_pset2_full/03_llm"
RETRY_DIR = OUT / "retry_chunks"


def main():
    chunks = sorted(RETRY_DIR.glob("retry_chunk_*.csv"))
    if not chunks:
        print("no retry chunks", file=sys.stderr)
        return 1
    retries = pd.concat([pd.read_csv(p) for p in chunks], ignore_index=True)
    print(f"merged {len(chunks)} retry chunks, {len(retries)} retry rows")
    fixed = retries["parsed_p"].notna().sum()
    print(f"  of which {fixed} now parse successfully ({fixed/len(retries)*100:.1f}%)")

    full = pd.read_csv(OUT / "versa_cache_full.csv")
    # Replace failed rows with retries where retry succeeded
    key_cols = ["row_id", "prompt_format"]
    full_keyed = full.set_index(key_cols)
    retries_keyed = retries.set_index(key_cols)
    # Only overwrite rows where retry has a parsed_p (don't replace good with bad)
    good_retries = retries_keyed[retries_keyed["parsed_p"].notna()]
    full_keyed.loc[good_retries.index, ["response", "parsed_p"]] = good_retries[["response", "parsed_p"]]
    fixed_full = full_keyed.reset_index()
    fixed_full.to_csv(OUT / "versa_cache_full.csv", index=False)
    print(f"updated versa_cache_full.csv: {fixed} new successful parses applied")

    # Recompute AUC per format on the updated cache
    test_idx = pd.read_csv(REPO / "exp/20260512_pset2_full/02_models/test_row_idx.csv")["row_idx"].tolist()
    y_test = pd.read_csv(REPO / "exp/20260512_pset2_full/02_models/test_labels.csv", index_col=0)["y_30d"]
    test_idx = sorted(test_idx)

    results = {}
    for fmt in ["structured", "natural", "abbreviated"]:
        sub = fixed_full[(fixed_full["prompt_format"] == fmt) & (fixed_full["row_id"].isin(test_idx))].copy()
        valid = sub["parsed_p"].notna()
        n_valid = int(valid.sum())
        coverage = n_valid / len(test_idx)
        y = y_test.loc[sub.loc[valid, "row_id"]].values
        p = sub.loc[valid, "parsed_p"].astype(float).values
        try:
            auc, lo, hi = delong_auc_ci(y, p)
        except Exception:
            auc, lo, hi = bootstrap_auc_ci(y, p, n_boot=1000)
        print(f"{fmt:12s}  AUC={auc:.4f}  (95% CI {lo:.4f}-{hi:.4f})  n_valid={n_valid}  coverage={coverage:.3f}")
        results[fmt] = {"auc": float(auc), "ci_lo": float(lo), "ci_hi": float(hi),
                        "n_valid": n_valid, "coverage": coverage}

    with open(OUT / "results_full.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    sys.exit(main() or 0)
