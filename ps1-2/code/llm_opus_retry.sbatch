#!/bin/bash
#SBATCH --job-name=cph_opus_retry
#SBATCH --time=00:25:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2
#SBATCH --partition=short
#SBATCH --output=/data/rauschecker2/jkw/cph/CPH200C/logs/slurm/opus_retry_%A_%a.log
# Array bounds + NUM_CHUNKS via --array / --export

set -euo pipefail
REPO="/data/rauschecker2/jkw/cph/CPH200C"
PY="/data/rauschecker1/jkw/envs/cph/bin/python"

if [[ -f "$HOME/versa_key.env" ]]; then
  set -a; source "$HOME/versa_key.env"; set +a
fi
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

NUM_CHUNKS="${NUM_CHUNKS:?NUM_CHUNKS must be set}"

cd "$REPO"
"$PY" -u exp/20260512_pset2_full/03_llm/retry_opus_errors.py \
  --chunk-id "$SLURM_ARRAY_TASK_ID" --num-chunks "$NUM_CHUNKS"
