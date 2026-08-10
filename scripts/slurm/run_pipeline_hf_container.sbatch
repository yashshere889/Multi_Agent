#!/bin/bash
# Same as run_pipeline_hf.sbatch, but runs through Barkla's own
# nemotron/nano-12b-v2 Apptainer module instead of our own .venv, via that
# module's own `nemotron` wrapper (/opt/apps/pkg/applications/containers/
# nemotron/bin/nemotron) rather than a hand-rolled `apptainer exec`.
#
# Use this if `mamba_ssm` won't build in your .venv: that module ships a
# container with mamba_ssm/torch/transformers already working for this model,
# so we install just the rest of this repo's dependencies into it (as --user,
# so they land in $HOME and persist) instead of building mamba_ssm ourselves.
#
# No separate "download the model" step: the wrapper downloads weights into
# NEMOTRON_CACHE on first use (same as any HF_HOME cache) and every later run
# reuses them from there, since NEMOTRON_CACHE below is a fixed fastscratch
# path rather than the wrapper's own default of node-local /tmp.
#
# One-time setup, from the login node (no build/GPU needed, just pip):
#   module load nemotron/nano-12b-v2
#   apptainer exec "$NEMOTRON_SIF" python3 -m pip install --user \
#       -e "/mnt/fastscratch/users/$USER/multi-agent-langraph[huggingface]"
#
# The Coder Agent's per-experiment package installs (sandbox.ensure_experiment_env)
# work here with no extra setup: it falls back to `python -m venv` + `pip` when
# `uv` isn't on this container's PATH, which it isn't (this module ships plain pip).
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

# Point the wrapper's cache at persistent fastscratch instead of its own
# default of node-local /tmp/users/$USER/nemotron-cache — that default is
# wiped per node/job, so every run would re-download from scratch. This path
# is fixed across jobs, so the model downloads once on first use and every
# run after reuses it (the wrapper itself binds this to /opt/huggingface_cache
# and sets HF_HOME inside the container — see nemotron/bin/nemotron).
export NEMOTRON_CACHE="/mnt/fastscratch/users/$USER/hf_cache"

cd "$PROJECT_DIR"

echo "Job $SLURM_JOB_ID on $SLURM_NODELIST, work dir $WORK_DIR"
echo "Container: $NEMOTRON_SIF"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

export LLM_BACKEND=huggingface
export LLM_MODEL=nvidia/NVIDIA-Nemotron-Nano-12B-v2
export LLM_HF_DEVICE_MAP=auto
export LLM_HF_DTYPE=auto

# The wrapper (module-provided, not ours) already handles --nv, binding /mnt
# and $NEMOTRON_CACHE, and setting HF_HOME inside the container — see
# /opt/apps/pkg/applications/containers/nemotron/bin/nemotron.
nemotron -m research_pipeline.cli orchestrate "$QUESTION" \
    --download-dir "$WORK_DIR/papers" \
    --output-dir "$WORK_DIR/outputs"

echo "Done. Outputs in $WORK_DIR/outputs (paper PDFs: v1.pdf, v2.pdf, ...)."
