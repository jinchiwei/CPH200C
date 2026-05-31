#!/bin/bash
#SBATCH --job-name=catboost_bucketed
#SBATCH --time=00:20:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=short
#SBATCH --output=/data/rauschecker2/jkw/cph/CPH200C/logs/slurm/catboost_bucketed_%j.log

set -euo pipefail
REPO="/data/rauschecker2/jkw/cph/CPH200C"
PY="/data/rauschecker1/jkw/envs/cph/bin/python"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
OUT="${REPO}/exp/20260513_auc_ceiling/results/catboost_bucketed"
mkdir -p "$OUT" "${REPO}/exp/20260513_auc_ceiling/raw/catboost_bucketed"
cd "$REPO"
"$PY" -u exp/20260513_auc_ceiling/run_candidate.py \
  --config catboost_bucketed \
  --out-results "$OUT" \
  --out-exp "${REPO}/exp/20260513_auc_ceiling/raw/catboost_bucketed"
