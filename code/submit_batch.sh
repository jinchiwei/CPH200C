#!/usr/bin/env bash
# Submit pending CPH200C autoresearch candidates as a SLURM array.
# Mirrors /data/rauschecker2/jkw/dmg/code/submit_batch.sh.
#
# Usage:
#   bash code/submit_batch.sh [N_TASKS]
# N_TASKS = max number of pending candidates to submit (default: 20)
set -euo pipefail

N=${1:-20}
REPO="/data/rauschecker2/jkw/cph/CPH200C"
SKILL_DIR="$HOME/arcadia/superstack/skills/autoresearch"
slug="CPH200C"
state_dir="$HOME/.gstack/projects/$slug/autoresearch"
SBATCH_SCRIPT="${REPO}/code/cph_cell.sbatch"
COLLECT_SCRIPT="${REPO}/code/collect_after.sbatch"

CAND_DIR="${state_dir}/batches"
mkdir -p "$CAND_DIR"
BATCH_ID="$(date +%Y%m%d_%H%M%S)"
CAND_FILE="${CAND_DIR}/batch_${BATCH_ID}.txt"
LOG_DIR="${REPO}/logs/slurm/${BATCH_ID}"
mkdir -p "$LOG_DIR"

# Pick pending candidates, ordered by their position in the queue.
"$SKILL_DIR/bin/state-read" --slug "$slug" \
  --path "[.candidate_queue[] | select(.status == \"pending\")] | [.[0:${N}] | .[].id] | .[]" \
  | tr -d '"' > "$CAND_FILE"

actual=$(wc -l < "$CAND_FILE" | tr -d ' ')
if [[ "$actual" -eq 0 ]]; then
  echo "no pending candidates"
  exit 0
fi

echo "Submitting $actual SLURM array tasks (batch_id=$BATCH_ID)"
echo "candidates:"
nl "$CAND_FILE"

# Mark all as running so a second submission doesn't double-claim.
IDS_JSON=$(/data/rauschecker1/jkw/envs/cph/bin/python -c "
import json
print(json.dumps([l.strip() for l in open('$CAND_FILE') if l.strip()]))
")

"$SKILL_DIR/bin/state-update" --slug "$slug" \
  --argjson ids "$IDS_JSON" \
  --set '
    .candidate_queue = (.candidate_queue | map(
      if (.id as $id | $ids | index($id)) and .status == "pending"
      then .status = "running" else . end
    ))
  ' >/dev/null

# Submit the array.
JOB_OUTPUT=$(sbatch \
  --output="${LOG_DIR}/cell_%A_%a.log" \
  --array="1-${actual}" \
  --export="ALL,CAND_FILE=${CAND_FILE},BATCH_ID=${BATCH_ID}" \
  "${SBATCH_SCRIPT}")
echo "$JOB_OUTPUT"
ARRAY_JOB_ID=$(echo "$JOB_OUTPUT" | awk '{print $4}')
echo "$ARRAY_JOB_ID" > "${CAND_DIR}/batch_${BATCH_ID}.jobid"

# Submit collector with afterany dependency.
COLLECT_OUTPUT=$(sbatch \
  --output="${LOG_DIR}/collect_%A.log" \
  --dependency="afterany:${ARRAY_JOB_ID}" \
  --export="ALL,BATCH_ID=${BATCH_ID},CAND_FILE=${CAND_FILE}" \
  "${COLLECT_SCRIPT}")
echo "$COLLECT_OUTPUT"
COLLECT_JOB_ID=$(echo "$COLLECT_OUTPUT" | awk '{print $4}')
echo "$COLLECT_JOB_ID" > "${CAND_DIR}/batch_${BATCH_ID}.collect.jobid"

echo "Array job: $ARRAY_JOB_ID"
echo "Collect job: $COLLECT_JOB_ID"
echo "Log dir: $LOG_DIR"
echo "Cand file: $CAND_FILE"
