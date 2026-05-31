"""One chunk of the full-test-set Opus 4.6 eval (parallel SLURM array).

Usage:
    python run_llm_chunk.py --chunk-id <1..N> --num-chunks <N>

Reads section 1.2's test_row_idx.csv, slices to this chunk's rows, runs Versa
Opus 4.6 on all three prompt formats for those rows. Writes its own per-chunk
cache so concurrent jobs don't collide.

Output:
    versa_cache_chunk_{NN}.csv  -- (row_id, prompt_format, response, parsed_p)

Aggregation happens in a separate collector job after the array finishes.
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
from llm_eval import OPUS_46, run_versa_eval  # noqa: E402
from preprocessing import clean_and_filter, load_diabetes  # noqa: E402

# Per-model directories so Opus and o1 don't share cache state.
MODEL_DIRS = {
    OPUS_46: REPO / "exp/20260512_pset2_full/03_llm",
    "o1-2024-12-17": REPO / "exp/20260513_auc_ceiling/llm_o1",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk-id", type=int, required=True, help="1-indexed")
    ap.add_argument("--num-chunks", type=int, required=True)
    ap.add_argument("--model", default=OPUS_46, help="Versa model id (default Opus 4.6)")
    args = ap.parse_args()
    OUT = MODEL_DIRS.get(args.model, REPO / "exp/20260512_pset2_full/03_llm")
    CHUNK_DIR = OUT / "chunks"
    CHUNK_DIR.mkdir(parents=True, exist_ok=True)

    if not (1 <= args.chunk_id <= args.num_chunks):
        raise SystemExit(f"chunk-id {args.chunk_id} out of range 1..{args.num_chunks}")

    # Same df + test set as section 1.2 + LLM eval
    df = clean_and_filter(load_diabetes()).reset_index(drop=True)
    test_idx = pd.read_csv(REPO / "exp/20260512_pset2_full/02_models/test_row_idx.csv")["row_idx"].tolist()
    test_idx = sorted(test_idx)

    # Slice to this chunk
    n = len(test_idx)
    starts = np.linspace(0, n, args.num_chunks + 1, dtype=int)
    lo, hi = starts[args.chunk_id - 1], starts[args.chunk_id]
    my_rows = test_idx[lo:hi]
    print(f"chunk {args.chunk_id}/{args.num_chunks}: rows [{lo}, {hi}) = {len(my_rows)} patients", flush=True)

    # Per-chunk cache (no collision with concurrent jobs)
    cache_path = CHUNK_DIR / f"versa_cache_chunk_{args.chunk_id:03d}.csv"
    log_dir = CHUNK_DIR / "logs"
    log_dir.mkdir(exist_ok=True)

    # Also bootstrap from the original n=300 cache so we never re-call API on those rows
    # Cache file name depends on the model (Opus: versa_cache.csv, o1: versa_cache_o1.csv)
    seed_cache_candidates = [OUT / "versa_cache.csv", OUT / "versa_cache_o1.csv"]
    for seed_path in seed_cache_candidates:
        if seed_path.exists() and not cache_path.exists():
            orig = pd.read_csv(seed_path)
            orig_subset = orig[orig["row_id"].isin(my_rows)]
            if len(orig_subset):
                orig_subset.to_csv(cache_path, index=False)
                print(f"seeded chunk cache with {len(orig_subset)} pre-cached rows from {seed_path.name}", flush=True)
                break

    for fmt in ["structured", "natural", "abbreviated"]:
        print(f"\n=== chunk {args.chunk_id}, format: {fmt}  model: {args.model} ===", flush=True)
        t0 = time.time()
        resp = run_versa_eval(
            df=df, row_ids=my_rows, prompt_format=fmt,
            cache_path=cache_path, model=args.model, log_dir=log_dir, verbose=False,
        )
        dt = time.time() - t0
        parsed = resp["parsed_p"].notna().sum()
        print(f"  done {len(resp)} rows ({parsed} parsed) in {dt/60:.1f} min", flush=True)

    print(f"chunk {args.chunk_id} complete", flush=True)


if __name__ == "__main__":
    main()
