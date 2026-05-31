#!/bin/bash
#SBATCH --job-name=cph_opus_collect
#SBATCH --time=00:10:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --partition=short
#SBATCH --output=/data/rauschecker2/jkw/cph/CPH200C/logs/slurm/opus_collect_%j.log

set -euo pipefail
REPO="/data/rauschecker2/jkw/cph/CPH200C"
PY="/data/rauschecker1/jkw/envs/cph/bin/python"
cd "$REPO"
"$PY" -u exp/20260512_pset2_full/03_llm/collect_full.py
