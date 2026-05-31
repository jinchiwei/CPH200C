#!/bin/bash
#SBATCH --job-name=cph_cell
#SBATCH --time=00:55:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=4
#SBATCH --partition=short

# One array-task = one candidate config from the autoresearch queue.
# CAND_FILE is exported by submit_batch.sh; one CAND_ID per line.

set -euo pipefail

REPO="/data/rauschecker2/jkw/cph/CPH200C"
CONDA_ENV_DIR="/data/rauschecker1/jkw/envs/cph"
PYTHON_BIN="${CONDA_ENV_DIR}/bin/python"
EXP_ROOT="${REPO}/exp/20260513_auc_ceiling"

CAND_FILE="${CAND_FILE:?CAND_FILE must be set}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: python missing at $PYTHON_BIN" >&2
  exit 1
fi

CAND_ID=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$CAND_FILE")
if [[ -z "$CAND_ID" ]]; then
  echo "ERROR: no CAND_ID for task ${SLURM_ARRAY_TASK_ID}" >&2
  exit 1
fi

OUT_RESULTS="${EXP_ROOT}/results/${CAND_ID}"
OUT_EXP="${EXP_ROOT}/raw/${CAND_ID}"
mkdir -p "$OUT_RESULTS" "$OUT_EXP"

echo "$(date '+%Y-%m-%d %H:%M:%S') CAND_ID=$CAND_ID  host=$(hostname)"
echo "  OUT_RESULTS=$OUT_RESULTS"
echo "  OUT_EXP=$OUT_EXP"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export TABPFN_MODEL_CACHE_DIR="${TABPFN_MODEL_CACHE_DIR:-/working/rauschecker1/jkw/projects/cnsl/data/tabpfn/model_cache}"
mkdir -p "$TABPFN_MODEL_CACHE_DIR"
# Pull TABPFN_TOKEN (and any other API keys) into the job env from the dotenv file.
if [[ -f "$HOME/versa_key.env" ]]; then
  set -a; source "$HOME/versa_key.env"; set +a
fi
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

cd "$REPO"
"$PYTHON_BIN" -u "$EXP_ROOT/run_candidate.py" \
  --config "$CAND_ID" \
  --out-results "$OUT_RESULTS" \
  --out-exp "$OUT_EXP"
status=$?

if [[ $status -ne 0 ]]; then
  echo "FAILED: $CAND_ID (exit=$status)"
  exit $status
fi
echo "OK: $CAND_ID"
