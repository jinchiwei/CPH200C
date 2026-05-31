"""GPT o1 LLM eval on the same 300-row sub-sample as the Opus 4.6 run.

Same 3 prompt formats (structured / natural / abbreviated). Independent
cache file so o1 + Opus responses don't collide.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/data/rauschecker2/jkw/cph/CPH200C")
sys.path.insert(0, str(REPO / "src"))
from cv import delong_auc_ci, bootstrap_auc_ci  # noqa: E402
from llm_eval import PROMPT_BUILDERS, run_versa_eval  # noqa: E402
from preprocessing import clean_and_filter, load_diabetes  # noqa: E402

OUT = REPO / "exp/20260513_auc_ceiling/llm_o1"
OUT.mkdir(parents=True, exist_ok=True)
CACHE = OUT / "versa_cache_o1.csv"

# GPT o1 canonical version available via Versa (see ARIA project's model zoo)
MODEL = "o1-2024-12-17"
N_LLM = 300
SEED = 42


def main():
    df = clean_and_filter(load_diabetes()).reset_index(drop=True)
    # Reuse the SAME sample as Opus 4.6 for apples-to-apples comparison
    sample_idx = pd.read_csv(
        REPO / "exp/20260512_pset2_full/03_llm/sample_row_idx.csv"
    )["row_idx"].tolist()
    y_test = pd.read_csv(
        REPO / "exp/20260512_pset2_full/02_models/test_labels.csv", index_col=0
    )["y_30d"]
    print(f"Sub-sample n={len(sample_idx)}, prevalence={y_test.loc[sample_idx].mean():.3f}")

    results = {}
    for fmt in ["structured", "natural", "abbreviated"]:
        print(f"\n=== Format: {fmt} ({MODEL}) ===", flush=True)
        t0 = time.time()
        resp_df = run_versa_eval(
            df=df, row_ids=sample_idx, prompt_format=fmt,
            cache_path=CACHE, model=MODEL, log_dir=OUT, verbose=False,
        )
        dt = time.time() - t0
        resp_df = resp_df.set_index("row_id").loc[sample_idx]
        valid = resp_df["parsed_p"].notna()
        n_valid = int(valid.sum())
        print(f"  parsed {n_valid}/{len(resp_df)}  runtime={dt/60:.1f} min")
        if n_valid < 30:
            results[fmt] = {"auc": None, "ci_lo": None, "ci_hi": None,
                            "n_valid": n_valid, "runtime_sec": dt}
            continue
        y_sub = y_test.loc[resp_df.index[valid]].values
        p_sub = resp_df.loc[valid, "parsed_p"].astype(float).values
        try:
            auc, lo, hi = delong_auc_ci(y_sub, p_sub)
        except Exception:
            auc, lo, hi = bootstrap_auc_ci(y_sub, p_sub)
        print(f"  AUC={auc:.4f} (95% CI {lo:.4f}-{hi:.4f}) on n_valid={n_valid}")
        results[fmt] = {"auc": float(auc), "ci_lo": float(lo), "ci_hi": float(hi),
                        "n_valid": n_valid, "runtime_sec": float(dt)}

    with open(OUT / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nFinal:", json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
