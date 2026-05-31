#!/bin/bash
#SBATCH --job-name=cph_collect
#SBATCH --time=00:15:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --partition=short

# Runs after the array job (afterany). Reads each candidate's metrics.json,
# updates state.json with results, picks the overall best.

set -euo pipefail
REPO="/data/rauschecker2/jkw/cph/CPH200C"
CONDA_ENV_DIR="/data/rauschecker1/jkw/envs/cph"
PYTHON_BIN="${CONDA_ENV_DIR}/bin/python"
SKILL_DIR="$HOME/arcadia/superstack/skills/autoresearch"
slug="CPH200C"
BATCH_ID="${BATCH_ID:?BATCH_ID required}"
CAND_FILE="${CAND_FILE:?CAND_FILE required}"

cd "$REPO"
"$PYTHON_BIN" -u code/collect_batch.py \
  --cand-file "$CAND_FILE" \
  --batch-id "$BATCH_ID"
