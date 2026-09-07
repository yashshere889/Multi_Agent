#!/bin/bash
#SBATCH --job-name=dpo-env
#SBATCH --partition=lowpriority
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.log
# Builds the training environment for the preference-optimisation experiment.
# Run as a job rather than on the login node: uv resolving torch's CUDA wheels
# is a long task, and long tasks on the login node are killed without warning.
set -euo pipefail

S=/mnt/fastscratch/users/$USER
# Both caches go on fastscratch: home is inode-capped at 150k and a torch
# install alone is tens of thousands of files.
export UV_CACHE_DIR=$S/uv_cache
export HF_HOME=$S/hf_cache
export VENV=$S/dpo/.venv

mkdir -p "$S/dpo"
# Idempotent: `uv venv` fails on an existing environment, and under `set -e`
# that aborts the whole job before anything is upgraded. Re-running this script
# to change a pinned version must work.
if [ -x "$VENV/bin/python" ]; then
    echo "=== reusing $VENV ==="
else
    echo "=== building $VENV ==="
    uv venv --python 3.11 "$VENV"
fi

# Default PyPI, not the cu121 index: trl 1.12 imports torch.distributed.fsdp
# .FSDPModule, which torch 2.5.1 (the newest the cu121 index serves) does not
# have. The nodes run driver 595.x, which is new enough for the CUDA runtime
# bundled with current torch wheels.
# >=2.6 for torch.distributed.fsdp.FSDPModule, which trl 1.12 imports.
uv pip install --python "$VENV/bin/python" --upgrade "torch>=2.6"

uv pip install --python "$VENV/bin/python" \
    "transformers>=4.44" "trl>=0.11" "peft>=0.13" "datasets>=2.20" \
    "accelerate>=0.34"

echo "=== verifying ==="
"$VENV/bin/python" - <<'PY'
import torch, transformers, trl, peft, datasets, accelerate
print("torch       ", torch.__version__)
print("transformers", transformers.__version__)
print("trl         ", trl.__version__)
print("peft        ", peft.__version__)
print("datasets    ", datasets.__version__)
print("accelerate  ", accelerate.__version__)
PY
echo "=== ENV READY ==="
