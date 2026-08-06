#!/bin/bash
# Serve NVIDIA Nemotron 3 Nano behind an OpenAI-compatible endpoint (vLLM) on a
# Barkla GPU node, WITHOUT running the pipeline in the same job.
#
#   sbatch scripts/slurm/run_llm_server.sbatch
#
# Use this when you want a long-lived model server to develop against — running
# individual agents by hand, iterating on prompts, driving it from your laptop.
# To run the whole pipeline unattended, prefer run_pipeline.sbatch, which puts
# the server and the pipeline in one job and skips all the plumbing below.
#
# Build the container first (once): scripts/slurm/build_vllm_sif.sh
#
# Once it's up, the log prints the node and port. From elsewhere on Barkla
# (e.g. a viz node) point the client straight at the node:
#   LLM_BASE_URL=http://<node>:<port>/v1
# From your laptop, tunnel through the login node first:
#   ssh -L 8000:<node>:<port> <user>@barklalogin1.liv.ac.uk
#   LLM_BASE_URL=http://localhost:8000/v1

#SBATCH --job-name=llm-server
#SBATCH --partition=gpu-h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=llm_server_%j.log

# See run_pipeline.sbatch for the full partition table and the VRAM reasoning.
# gpu-h100 (4x H100 80GB) and gpu-l40s (2x L40S 48GB) are open to all users;
# on gpu-l40s set --gres=gpu:2 and TP=2, since 48GB alone won't hold 60GB of
# BF16 weights.

set -euo pipefail

module purge
module load apptainer/1.3.6

export HF_HOME="/mnt/fastscratch/users/$USER/hf_cache"
VLLM_SIF="/mnt/fastscratch/users/$USER/containers/vllm.sif"
MODEL="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
TP=1
PORT=$(( 8000 + SLURM_JOB_ID % 1000 ))

# shellcheck source=_vllm_serve.sh
source "$SLURM_SUBMIT_DIR/scripts/slurm/_vllm_serve.sh"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cleanup() { [[ -n "${VLLM_PID:-}" ]] && kill "$VLLM_PID" 2>/dev/null || true; }
trap cleanup EXIT

vllm_serve_background
wait_for_vllm

cat <<EOF

================================================================
  Model server ready.
    node: $SLURM_NODELIST
    port: $PORT

  On Barkla:   LLM_BASE_URL=http://$SLURM_NODELIST:$PORT/v1
  Via tunnel:  ssh -L 8000:$SLURM_NODELIST:$PORT $USER@barklalogin1.liv.ac.uk
               LLM_BASE_URL=http://localhost:8000/v1

  LLM_MODEL=$MODEL
================================================================

EOF

# Hold the job open for the rest of its wall clock, serving requests.
wait "$VLLM_PID"
