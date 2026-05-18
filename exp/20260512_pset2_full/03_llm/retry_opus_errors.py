"""Retry the failed Opus 4.6 API calls (Bedrock throttling errors).

Reads the existing versa_cache_full.csv, finds rows where parsed_p is NaN AND
response is the "Error in API response" sentinel, slices to this chunk, and
re-calls Versa. Writes to a per-chunk retry cache. Collector merges.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/data/rauschecker2/jkw/cph/CPH200C")
sys.path.insert(0, str(REPO / "src"))
from llm_eval import OPUS_46, PROMPT_BUILDERS  # noqa: E402
from preprocessing import clean_and_filter, load_diabetes  # noqa: E402

OUT = REPO / "exp/20260512_pset2_full/03_llm"
RETRY_DIR = OUT / "retry_chunks"
RETRY_DIR.mkdir(parents=True, exist_ok=True)


def get_versa():
    sys.path.append("/data/rauschecker2/jkw/aria/dev/src/utils")
    from versa_api import VersaAI
    return VersaAI(deployment=OPUS_46, usage_log_dir=str(RETRY_DIR), run_dir=str(RETRY_DIR))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk-id", type=int, required=True, help="1-indexed")
    ap.add_argument("--num-chunks", type=int, required=True)
    args = ap.parse_args()

    full = pd.read_csv(OUT / "versa_cache_full.csv")
    failed = full[full["parsed_p"].isna() & full["response"].astype(str).str.startswith("Error in API")].copy()
    failed = failed.sort_values(["prompt_format", "row_id"]).reset_index(drop=True)
    print(f"Total API failures to retry: {len(failed):,}", flush=True)

    n = len(failed)
    starts = np.linspace(0, n, args.num_chunks + 1, dtype=int)
    lo, hi = starts[args.chunk_id - 1], starts[args.chunk_id]
    my = failed.iloc[lo:hi]
    print(f"chunk {args.chunk_id}/{args.num_chunks}: [{lo}, {hi}) = {len(my)} retries", flush=True)
    if not len(my):
        return 0

    df = clean_and_filter(load_diabetes()).reset_index(drop=True)
    versa = get_versa()

    out_rows = []
    cache_path = RETRY_DIR / f"retry_chunk_{args.chunk_id:03d}.csv"
    from llm_eval import parse_prob

    t0 = time.time()
    for i, r in enumerate(my.itertuples(index=False)):
        builder = PROMPT_BUILDERS[r.prompt_format]
        prompt = builder(df.loc[int(r.row_id)].to_dict())
        try:
            response = versa.predict(prompt, verbose=False)
        except Exception as e:
            response = f"Error in API response: {e}"
        parsed = parse_prob(response)
        out_rows.append({
            "row_id": int(r.row_id),
            "prompt_format": r.prompt_format,
            "response": response,
            "parsed_p": parsed,
        })
        # flush every 25
        if (i + 1) % 25 == 0:
            pd.DataFrame(out_rows).to_csv(cache_path, index=False)
            print(f"  {i+1}/{len(my)}  elapsed={(time.time()-t0)/60:.1f} min", flush=True)

    pd.DataFrame(out_rows).to_csv(cache_path, index=False)
    parsed = sum(1 for r in out_rows if r["parsed_p"] is not None)
    print(f"chunk {args.chunk_id}: {len(out_rows)} retries done, {parsed} now parseable", flush=True)


if __name__ == "__main__":
    sys.exit(main() or 0)
