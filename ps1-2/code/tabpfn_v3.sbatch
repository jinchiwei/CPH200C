#!/bin/bash
#SBATCH --job-name=tabpfn_v3
#SBATCH --time=03:55:00
#SBATCH --mem=256G
#SBATCH --cpus-per-task=8
#SBATCH --partition=long
#SBATCH --output=/data/rauschecker2/jkw/cph/CPH200C/logs/slurm/tabpfn_v3_%A_%a.log
#SBATCH --array=1-2

# Two-element array: 1 = tabpfn_v3_full (base features), 2 = tabpfn_v3_full_slim (slim features).

set -euo pipefail
REPO="/data/rauschecker2/jkw/cph/CPH200C"
PY="/data/rauschecker1/jkw/envs/cph/bin/python"

if [[ -f "$HOME/versa_key.env" ]]; then
  set -a; source "$HOME/versa_key.env"; set +a
fi
export TABPFN_MODEL_CACHE_DIR="${TABPFN_MODEL_CACHE_DIR:-/working/rauschecker1/jkw/projects/cnsl/data/tabpfn/model_cache}"
mkdir -p "$TABPFN_MODEL_CACHE_DIR"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8

case "$SLURM_ARRAY_TASK_ID" in
  1) CONFIG="tabpfn_v3_full" ;;
  2) CONFIG="tabpfn_v3_full_slim" ;;
  *) echo "bad task id"; exit 1 ;;
esac

OUT_RESULTS="${REPO}/exp/20260513_auc_ceiling/results/${CONFIG}"
OUT_EXP="${REPO}/exp/20260513_auc_ceiling/raw/${CONFIG}"
mkdir -p "$OUT_RESULTS" "$OUT_EXP"

echo "$(date) host=$(hostname) config=$CONFIG"
cd "$REPO"
"$PY" -u exp/20260513_auc_ceiling/run_candidate.py \
  --config "$CONFIG" \
  --out-results "$OUT_RESULTS" \
  --out-exp "$OUT_EXP"
