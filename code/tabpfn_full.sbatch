#!/bin/bash
#SBATCH --job-name=tabpfn_full
#SBATCH --time=01:55:00
#SBATCH --mem=256G
#SBATCH --cpus-per-task=8
#SBATCH --partition=long
#SBATCH --output=/data/rauschecker2/jkw/cph/CPH200C/logs/slurm/tabpfn_full_%j.log

# Single-task: TabPFN ignoring pretraining cap on full 49k train rows.
# Memory is the unknown; O(N^2) attention over 49k could OOM. 256G is the cap try.

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

OUT_RESULTS="${REPO}/exp/20260513_auc_ceiling/results/tabpfn_full"
OUT_EXP="${REPO}/exp/20260513_auc_ceiling/raw/tabpfn_full"
mkdir -p "$OUT_RESULTS" "$OUT_EXP"

cd "$REPO"
"$PY" -u exp/20260513_auc_ceiling/run_candidate.py \
  --config tabpfn_full \
  --out-results "$OUT_RESULTS" \
  --out-exp "$OUT_EXP"
