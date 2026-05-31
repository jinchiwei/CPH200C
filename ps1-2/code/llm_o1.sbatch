#!/bin/bash
#SBATCH --job-name=cph_o1
#SBATCH --time=04:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --partition=long
#SBATCH --output=/data/rauschecker2/jkw/cph/CPH200C/logs/slurm/llm_o1_%j.log

# Run GPT o1 LLM evaluation via Versa on the same 300-row sub-sample as Opus 4.6.

set -euo pipefail

REPO="/data/rauschecker2/jkw/cph/CPH200C"
CONDA_ENV_DIR="/data/rauschecker1/jkw/envs/cph"
PYTHON_BIN="${CONDA_ENV_DIR}/bin/python"

if [[ -f "$HOME/versa_key.env" ]]; then
  set -a; source "$HOME/versa_key.env"; set +a
fi
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

cd "$REPO"
"$PYTHON_BIN" -u exp/20260513_auc_ceiling/run_llm_o1.py
