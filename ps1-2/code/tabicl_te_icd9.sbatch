#!/bin/bash
#SBATCH --job-name=tabicl_te
#SBATCH --time=01:55:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --partition=long
#SBATCH --output=/data/rauschecker2/jkw/cph/CPH200C/logs/slurm/tabicl_te_icd9_%j.log

set -euo pipefail
REPO="/data/rauschecker2/jkw/cph/CPH200C"
PY="/data/rauschecker1/jkw/envs/cph/bin/python"
if [[ -f "$HOME/versa_key.env" ]]; then set -a; source "$HOME/versa_key.env"; set +a; fi
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
OUT="${REPO}/exp/20260513_auc_ceiling/results/tabicl_full_te_icd9"
mkdir -p "$OUT" "${REPO}/exp/20260513_auc_ceiling/raw/tabicl_full_te_icd9"
cd "$REPO"
"$PY" -u exp/20260513_auc_ceiling/run_candidate.py \
  --config tabicl_full_te_icd9 \
  --out-results "$OUT" \
  --out-exp "${REPO}/exp/20260513_auc_ceiling/raw/tabicl_full_te_icd9"
