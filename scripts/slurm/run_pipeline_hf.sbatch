#!/bin/bash
# Run the whole pipeline on Barkla with the model loaded IN-PROCESS
# (LLM_BACKEND=huggingface) — no vLLM, no container, no server, no port.
#
#   cd /mnt/fastscratch/users/$USER/multi-agent-langraph
#   sbatch scripts/slurm/run_pipeline_hf.sbatch "your research question"
#
# Compare run_pipeline.sbatch, which serves the model over vLLM. Pick between
# them on model size:
#
#   this script          Nemotron Nano 12B v2 (~24GB BF16). Fits every GPU
#                        partition except gpu-v100. Barkla ships a tested
#                        nemotron/nano-12b-v2 module for exactly this stack.
#   run_pipeline.sbatch  Nemotron 3 Nano 30B A3B (~60GB BF16). Needs an 80GB
#                        card, and at that size vLLM's throughput advantage
#                        over plain transformers is worth the extra plumbing.
#
# One-time setup on a viz node (NOT the login node — long tasks are killed
# there), which installs torch/transformers and pre-fetches the weights:
#   ssh <user>@barklaviz1.liv.ac.uk
#   cd /mnt/fastscratch/users/$USER/multi-agent-langraph
#   export HF_HOME=/mnt/fastscratch/users/$USER/hf_cache
#   uv sync --extra huggingface
#   uv run hf download nvidia/NVIDIA-Nemotron-Nano-12B-v2

#SBATCH --job-name=research-pipeline-hf
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=pipeline_hf_%j.log

# 24h rather than run_pipeline.sbatch's 12h: transformers has no paged
# attention or fused kernels, so the same number of calls takes appreciably
# longer than under vLLM. gpu-l40s allows up to 3 days if that isn't enough.
#
# Partitions open to all users (see the cluster's GPU doc):
#   gpu-l40s  2x L40S 48GB, 3-day limit  <- default; 24GB model fits one card
#   gpu-h100  4x H100 80GB, 3-day limit
#   gpu-v100  4x V100 16GB               <- too small, and Volta has no bfloat16
# Low priority (1-day limit, pre-empted and requeued when the owning group
# submits — a half-finished run restarts from scratch):
#   gpu-a100-lowbig, gpu-a-lowsmall
#
# No --cpus-per-task / --mem: Barkla allocates both from the GPU count.

set -euo pipefail

QUESTION="${1:?usage: sbatch run_pipeline_hf.sbatch \"research question\"}"

module purge

PROJECT_DIR="$SLURM_SUBMIT_DIR"
WORK_DIR="/mnt/fastscratch/users/$USER/pipeline-runs/$SLURM_JOB_ID"
mkdir -p "$WORK_DIR"

# Weights live on fastscratch, never home: home is capped at 75GB/100k inodes
# and a single checkpoint plus its blobs would eat a large share of both.
export HF_HOME="/mnt/fastscratch/users/$USER/hf_cache"

cd "$PROJECT_DIR"

echo "Job $SLURM_JOB_ID on $SLURM_NODELIST, work dir $WORK_DIR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Real environment variables beat .env, which config.py reads via load_dotenv().
export LLM_BACKEND=huggingface
export LLM_MODEL=nvidia/NVIDIA-Nemotron-Nano-12B-v2
# device_map=auto lets accelerate place the model across whatever GPUs were
# allocated, so bumping --gres above works without touching this script.
export LLM_HF_DEVICE_MAP=auto
export LLM_HF_DTYPE=auto

# No health check to wait on, unlike the vLLM path — the first agent's own
# call blocks while transformers loads the checkpoint, and a failure to load
# surfaces as a normal Python traceback rather than a server that never
# answers. That is most of why this script is shorter.
uv run research-pipeline orchestrate "$QUESTION" \
    --download-dir "$WORK_DIR/papers" \
    --output-dir "$WORK_DIR/outputs"

echo "Done. Outputs in $WORK_DIR/outputs (paper PDFs: v1.pdf, v2.pdf, ...)."
