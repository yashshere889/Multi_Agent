#!/bin/bash
# Same as run_pipeline_hf.sbatch, but runs through Barkla's own
# nemotron/nano-12b-v2 Apptainer module instead of our own .venv.
#
# Use this if `mamba_ssm` won't build in your .venv: that module ships a
# container with mamba_ssm/torch/transformers already working for this model,
# so we install just the rest of this repo's dependencies into it (as --user,
# so they land in $HOME and persist) instead of building mamba_ssm ourselves.
#
# One-time setup, from the login node (no build/GPU needed, just pip):
#   module load nemotron/nano-12b-v2
#   module show nemotron/nano-12b-v2        # confirm the NEMOTRON_SIF path below
#   apptainer exec "$NEMOTRON_SIF" python3 -m pip install --user \
#       -e "/mnt/fastscratch/users/$USER/multi-agent-langraph[huggingface]"
#
#   cd /mnt/fastscratch/users/$USER/multi-agent-langraph
#   sbatch scripts/slurm/run_pipeline_hf_container.sbatch "your research question"

#SBATCH --job-name=research-pipeline-hf-container
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=pipeline_hf_container_%j.log

set -euo pipefail

QUESTION="${1:?usage: sbatch run_pipeline_hf_container.sbatch \"research question\"}"

module purge
module load nemotron/nano-12b-v2

PROJECT_DIR="$SLURM_SUBMIT_DIR"
WORK_DIR="/mnt/fastscratch/users/$USER/pipeline-runs/$SLURM_JOB_ID"
mkdir -p "$WORK_DIR"

# Reuse the weights already fetched into fastscratch — no re-download. Mirrors
# what the `nemotron` wrapper script (/opt/apps/.../nemotron/bin/nemotron)
# does: bind the cache dir to /opt/huggingface_cache and point HF_HOME there,
# since the container only sees paths that are explicitly bind-mounted in.
NEMOTRON_CACHE="/mnt/fastscratch/users/$USER/hf_cache"

cd "$PROJECT_DIR"

echo "Job $SLURM_JOB_ID on $SLURM_NODELIST, work dir $WORK_DIR"
echo "Container: $NEMOTRON_SIF"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

export LLM_BACKEND=huggingface
export LLM_MODEL=nvidia/NVIDIA-Nemotron-Nano-12B-v2
export LLM_HF_DEVICE_MAP=auto
export LLM_HF_DTYPE=auto

# --bind /mnt: the container's root is otherwise read-only, so without this
# nothing under /mnt/fastscratch (outputs, papers, the HF cache) is writable
# or even visible from inside it.
apptainer exec --nv \
    --bind /mnt \
    --bind "$NEMOTRON_CACHE:/opt/huggingface_cache" \
    --env HF_HOME=/opt/huggingface_cache \
    "$NEMOTRON_SIF" python3 -m research_pipeline.cli orchestrate "$QUESTION" \
    --download-dir "$WORK_DIR/papers" \
    --output-dir "$WORK_DIR/outputs"

echo "Done. Outputs in $WORK_DIR/outputs (paper PDFs: v1.pdf, v2.pdf, ...)."
