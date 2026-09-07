#!/bin/bash
#SBATCH --job-name=dpo-coder
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.log
# Direct preference optimisation on pairs harvested from the Coder Agent's own
# fix loop. See scripts/dpo/train_dpo.py for what is and is not being claimed.
set -euo pipefail

S=/mnt/fastscratch/users/$USER
export HF_HOME=$S/hf_cache
export UV_CACHE_DIR=$S/uv_cache
export TOKENIZERS_PARALLELISM=false

RUN_DIR=$S/dpo/run-$SLURM_JOB_ID
mkdir -p "$RUN_DIR"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

"$S/dpo/.venv/bin/python" "$S/dpo/train_dpo.py" \
    --pairs "$S/dpo/preference_pairs.jsonl" \
    --output-dir "$RUN_DIR" \
    --max-length 4096 \
    --max-prompt-length 768 \
    --epochs 3

echo "=== result ==="
cat "$RUN_DIR/result.json"
