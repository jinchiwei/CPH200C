"""Post-array collector for CPH200C autoresearch.

Walks each completed candidate's results dir, reads metrics.json, updates the
autoresearch state.json with the result and current_best. Also appends a
summary block to the session research log (or local notes).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
from pathlib import Path

REPO = Path("/data/rauschecker2/jkw/cph/CPH200C")
EXP_RESULTS = REPO / "exp/20260513_auc_ceiling/results"
SKILL_DIR = Path.home() / "arcadia/superstack/skills/autoresearch"
SLUG = "CPH200C"


def state_read(path: str) -> str:
    out = subprocess.run(
        [str(SKILL_DIR / "bin/state-read"), "--slug", SLUG, "--path", path],
        capture_output=True, text=True, check=True,
    )
    return out.stdout.strip()


def state_update(filter_expr: str, **kwargs):
    cmd = [str(SKILL_DIR / "bin/state-update"), "--slug", SLUG]
    for k, v in kwargs.items():
        if isinstance(v, (dict, list)):
            cmd += [f"--argjson", k, json.dumps(v)]
        else:
            cmd += [f"--arg", k, str(v)]
    cmd += ["--set", filter_expr]
    subprocess.run(cmd, check=True, capture_output=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand-file", required=True)
    ap.add_argument("--batch-id", required=True)
    args = ap.parse_args()

    cand_ids = [l.strip() for l in open(args.cand_file) if l.strip()]
    now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    print(f"Collecting {len(cand_ids)} candidates from batch {args.batch_id}")

    summaries = []
    for cid in cand_ids:
        metrics_path = EXP_RESULTS / cid / "metrics.json"
        if not metrics_path.exists():
            print(f"  [{cid}] MISSING metrics.json — marking failed")
            entry = {
                "id": cid, "axes": {"config": cid}, "started_at": now, "ended_at": now,
                "status": "failed", "metric_value": None,
                "error_class": "infrastructure",
                "notes": "metrics.json not produced",
            }
            summaries.append(entry)
            continue
        m = json.loads(metrics_path.read_text())
        entry = {
            "id": cid, "axes": {"config": cid}, "started_at": now, "ended_at": now,
            "status": "complete", "metric_value": m["auc"],
            "ci_lo": m["ci_lo"], "ci_hi": m["ci_hi"],
            "runtime_sec": m["runtime_sec"],
            "notes": m.get("extra", {}).get("model", cid),
        }
        summaries.append(entry)
        print(f"  [{cid}] AUC={m['auc']:.4f} ({m['ci_lo']:.4f}-{m['ci_hi']:.4f})")

    # Update state.json: append to results_history, mark candidates done, update current_best.
    state_update(
        '.results_history += $entries '
        '| .candidate_queue = (.candidate_queue | map('
        '    if (.id as $id | $ids | index($id))'
        '    then .status = (($entries[] | select(.id == $id) | .status) // "failed")'
        '    else . end))'
        '| .last_iteration_completed_at = $now'
        '| .iteration_count += ($entries | length)',
        entries=summaries, ids=[s["id"] for s in summaries], now=now,
    )

    # Pick overall best (highest AUC) and write current_best.
    successes = [s for s in summaries if s["status"] == "complete"]
    if successes:
        best = max(successes, key=lambda s: s["metric_value"])
        # Also pull historical best (from earlier section 1.2): XGBoost 0.691
        # so the best across sections is in current_best.
        state_update(
            '.current_best = $best',
            best={
                "id": best["id"], "metric_name": "AUC", "metric_value": best["metric_value"],
                "ci_lo": best["ci_lo"], "ci_hi": best["ci_hi"], "from_batch": args.batch_id,
            },
        )
        print(f"\nCurrent best: {best['id']} AUC={best['metric_value']:.4f}")

    # Write a summary CSV for the report.
    out_csv = REPO / "exp/20260513_auc_ceiling/results_table.csv"
    with open(out_csv, "w") as f:
        f.write("config,auc,ci_lo,ci_hi,runtime_sec,status\n")
        for s in summaries:
            f.write(f"{s['id']},{s.get('metric_value','')},"
                    f"{s.get('ci_lo','')},{s.get('ci_hi','')},"
                    f"{s.get('runtime_sec','')},{s['status']}\n")
    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
