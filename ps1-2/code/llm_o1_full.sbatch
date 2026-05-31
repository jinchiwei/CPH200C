#!/bin/bash
#SBATCH --job-name=cph_o1_full
#SBATCH --time=03:55:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2
#SBATCH --partition=long
#SBATCH --output=/data/rauschecker2/jkw/cph/CPH200C/logs/slurm/o1_full_%A_%a.log
# o1 is much slower than Opus (~6-10 sec/call) so we use long partition + more chunks.
# Array bounds set via --array on submission; NUM_CHUNKS via --export.

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
"$PY" -u exp/20260512_pset2_full/03_llm/run_llm_chunk.py \
  --chunk-id "$SLURM_ARRAY_TASK_ID" --num-chunks "$NUM_CHUNKS" \
  --model "o1-2024-12-17"
