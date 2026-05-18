#!/bin/bash
#SBATCH --job-name=shift_catboost
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=short
#SBATCH --output=/data/rauschecker2/jkw/cph/CPH200C/logs/slurm/shift_catboost_%j.log

set -euo pipefail
REPO="/data/rauschecker2/jkw/cph/CPH200C"
PY="/data/rauschecker1/jkw/envs/cph/bin/python"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
cd "$REPO"
"$PY" -u exp/20260513_auc_ceiling/run_shift_catboost.py
