"""Re-run TabPFN-v3 with alignment checkpoints baked in.

Saves both predictions AND the y_te that produced them, plus a metrics-with-
verification file that records AUC at three points: in-memory immediately after
predict_proba, after np.save round-trip, and against the section-1.2 saved labels.

If those three numbers diverge, we know exactly where the bug is.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

REPO = Path("/data/rauschecker2/jkw/cph/CPH200C")
sys.path.insert(0, str(REPO / "src"))
from preprocessing import build_features, clean_and_filter, load_diabetes

SEED = 42
OUT = REPO / "exp/20260513_auc_ceiling/results/tabpfn_v3_full_verified"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    print("loading split...", flush=True)
    df = load_diabetes()
    df = clean_and_filter(df).reset_index(drop=True)
    X, y, names, pids = build_features(df)
    X_tv, X_te, y_tv, y_te = train_test_split(X, y, test_size=0.15, stratify=y, random_state=SEED)
    X_tr, X_va, y_tr, y_va = train_test_split(X_tv, y_tv, test_size=0.15/0.85, stratify=y_tv, random_state=SEED)
    print(f"  n_train={len(y_tr)} n_test={len(y_te)} pos_rate={y_tr.mean():.4f}", flush=True)

    from tabpfn import TabPFNClassifier
    from tabpfn.model_loading import ModelVersion
    print("fitting TabPFN-v3 on full 49k...", flush=True)
    t0 = time.time()
    m = TabPFNClassifier.create_default_for_version(
        ModelVersion.V3, device="cpu", ignore_pretraining_limits=True, random_state=SEED,
    )
    m.fit(X_tr.values, y_tr.values)
    print(f"  fit done in {time.time()-t0:.1f}s", flush=True)

    print("predict_proba...", flush=True)
    t1 = time.time()
    proba = m.predict_proba(X_te.values)
    print(f"  predict done in {time.time()-t1:.1f}s", flush=True)
    print(f"  proba shape: {proba.shape}, classes_: {m.classes_}", flush=True)

    p_col1 = proba[:, 1]
    p_col0 = proba[:, 0]

    # ---- Checkpoint A: in-memory, just-computed ----
    auc_A_col1 = roc_auc_score(y_te.values, p_col1)
    auc_A_col0 = roc_auc_score(y_te.values, p_col0)
    print(f"CHECKPOINT A (in-memory):")
    print(f"  AUC with col1 (P(y=1)): {auc_A_col1:.4f}")
    print(f"  AUC with col0 (P(y=0)): {auc_A_col0:.4f}")

    # ---- Save predictions + the y_te that goes with them ----
    np.save(OUT / "preds_test.npy", p_col1.astype(np.float32))
    np.save(OUT / "preds_test_col0.npy", p_col0.astype(np.float32))
    np.save(OUT / "y_te_aligned.npy", y_te.values.astype(np.int8))
    # Also save the test indices for downstream debugging
    pd.Series(y_te.index, name="test_idx").to_csv(OUT / "test_indices.csv", index=False)

    # ---- Checkpoint B: round-trip via disk ----
    p_loaded = np.load(OUT / "preds_test.npy")
    y_loaded = np.load(OUT / "y_te_aligned.npy")
    auc_B = roc_auc_score(y_loaded, p_loaded)
    print(f"CHECKPOINT B (round-trip via saved npy):")
    print(f"  AUC: {auc_B:.4f}")
    print(f"  preds match in-memory: {np.allclose(p_col1.astype(np.float32), p_loaded)}")
    print(f"  y_te matches: {np.array_equal(y_loaded, y_te.values.astype(np.int8))}")

    # ---- Checkpoint C: section-1.2's saved test_labels.csv ----
    sec_1_2_labels = pd.read_csv(REPO / "exp/20260512_pset2_full/02_models/test_labels.csv", index_col=0)["y_30d"]
    auc_C = roc_auc_score(sec_1_2_labels.values, p_col1)
    print(f"CHECKPOINT C (section 1.2's saved y_test):")
    print(f"  AUC: {auc_C:.4f}")
    print(f"  y_te orderings equal: {np.array_equal(sec_1_2_labels.values, y_te.values)}")
    print(f"  y_te indices equal: {list(sec_1_2_labels.index[:5])} vs {list(y_te.index[:5])}")

    summary = {
        "checkpoint_A_inmem_col1": float(auc_A_col1),
        "checkpoint_A_inmem_col0": float(auc_A_col0),
        "checkpoint_B_roundtrip": float(auc_B),
        "checkpoint_C_sec1_2_labels": float(auc_C),
        "n_train": int(len(y_tr)), "n_test": int(len(y_te)),
        "classes_": list(m.classes_.tolist()),
    }
    (OUT / "verification.json").write_text(json.dumps(summary, indent=2))
    print(f"\nVERIFICATION SAVED to {OUT / 'verification.json'}")


if __name__ == "__main__":
    main()
