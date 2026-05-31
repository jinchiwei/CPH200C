"""Section 1.3 -- Versa Opus 4.6 on the same test set as section 1.2.

Three prompt formats: structured / natural / abbreviated. Sub-samples the test
set down to N_LLM rows (default 300) to keep token cost & runtime tractable.
Cached responses live in this folder so reruns resume.
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
from cv import delong_auc_ci  # noqa: E402
from llm_eval import OPUS_46, PROMPT_BUILDERS, run_versa_eval  # noqa: E402
from preprocessing import clean_and_filter, load_diabetes  # noqa: E402

OUT = REPO / "exp/20260512_pset2_full/03_llm"
LLM_CACHE = OUT / "versa_cache.csv"

N_LLM = 300  # sub-sample size for the LLM eval (token + time budget)
SEED = 42


def main():
    # Load full filtered df (same as section 1.2)
    df_raw = load_diabetes()
    df = clean_and_filter(df_raw).reset_index(drop=True)

    # Load the canonical test row indices from 1.2
    test_idx = pd.read_csv(REPO / "exp/20260512_pset2_full/02_models/test_row_idx.csv")["row_idx"].tolist()
    y_test = pd.read_csv(REPO / "exp/20260512_pset2_full/02_models/test_labels.csv", index_col=0)["y_30d"]

    # Sub-sample N_LLM rows from the test set, stratified by outcome
    rng = np.random.default_rng(SEED)
    test_idx_arr = np.array(test_idx)
    y_arr = y_test.loc[test_idx_arr].values
    pos = test_idx_arr[y_arr == 1]
    neg = test_idx_arr[y_arr == 0]
    n_pos = min(int(N_LLM * y_arr.mean()) + 1, len(pos))
    n_neg = N_LLM - n_pos
    samp_pos = rng.choice(pos, size=n_pos, replace=False)
    samp_neg = rng.choice(neg, size=n_neg, replace=False)
    sample_idx = np.sort(np.concatenate([samp_pos, samp_neg])).tolist()
    print(f"LLM eval sub-sample: n={len(sample_idx)}, pos={n_pos}, neg={n_neg} "
          f"(prevalence={y_test.loc[sample_idx].mean():.3f})")

    pd.Series(sample_idx, name="row_idx").to_csv(OUT / "sample_row_idx.csv", index=False)

    results = {}
    for fmt in ["structured", "natural", "abbreviated"]:
        print(f"\n=== Format: {fmt} ===")
        t0 = time.time()
        resp_df = run_versa_eval(
            df=df, row_ids=sample_idx, prompt_format=fmt,
            cache_path=LLM_CACHE, model=OPUS_46, log_dir=OUT, verbose=False,
        )
        dt = time.time() - t0
        # align with labels
        resp_df = resp_df.set_index("row_id").loc[sample_idx]
        valid = resp_df["parsed_p"].notna()
        n_valid = int(valid.sum())
        print(f"  parsed {n_valid}/{len(resp_df)} responses; runtime={dt/60:.1f} min")
        if n_valid < 30:
            results[fmt] = {"auc": None, "ci_lo": None, "ci_hi": None,
                            "n_valid": n_valid, "runtime_sec": dt}
            continue
        y_sub = y_test.loc[resp_df.index[valid]].values
        p_sub = resp_df.loc[valid, "parsed_p"].astype(float).values
        try:
            auc, lo, hi = delong_auc_ci(y_sub, p_sub)
        except Exception as e:
            print(f"  DeLong failed ({e}); using bootstrap")
            from cv import bootstrap_auc_ci
            auc, lo, hi = bootstrap_auc_ci(y_sub, p_sub)
        print(f"  AUC={auc:.4f} (95% CI {lo:.4f}-{hi:.4f}) on n_valid={n_valid}")
        results[fmt] = {"auc": auc, "ci_lo": lo, "ci_hi": hi,
                        "n_valid": n_valid, "runtime_sec": dt}

    with open(OUT / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    print("\nFinal:", json.dumps(results, indent=2, default=float))


if __name__ == "__main__":
    main()
