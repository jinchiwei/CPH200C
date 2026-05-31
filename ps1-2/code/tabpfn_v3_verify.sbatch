#!/bin/bash
#SBATCH --job-name=tabpfn_v3_verify
#SBATCH --time=01:55:00
#SBATCH --mem=256G
#SBATCH --cpus-per-task=8
#SBATCH --partition=long
#SBATCH --output=/data/rauschecker2/jkw/cph/CPH200C/logs/slurm/tabpfn_v3_verify_%j.log

set -euo pipefail
REPO="/data/rauschecker2/jkw/cph/CPH200C"
PY="/data/rauschecker1/jkw/envs/cph/bin/python"
if [[ -f "$HOME/versa_key.env" ]]; then set -a; source "$HOME/versa_key.env"; set +a; fi
export TABPFN_MODEL_CACHE_DIR="${TABPFN_MODEL_CACHE_DIR:-/working/rauschecker1/jkw/projects/cnsl/data/tabpfn/model_cache}"
mkdir -p "$TABPFN_MODEL_CACHE_DIR"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8

cd "$REPO"
"$PY" -u exp/20260513_auc_ceiling/rerun_tabpfn_v3_verified.py
