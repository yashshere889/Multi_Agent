#!/bin/bash -l
# Run the WHOLE pipeline (vLLM server + all six agents) inside one Barkla job.
#
#   cd /mnt/fastscratch/users/$USER/multi-agent-langraph
#   sbatch scripts/slurm/run_pipeline.sbatch "your research question"
#
# Submit from scratch or fastscratch — Barkla's storage guide says not to submit
# from home, data or localscratch.
#
# Why one job instead of the README's server-job-plus-SSH-tunnel setup: the
# pipeline itself is pure CPU (langgraph/requests/reportlab) and only needs to
# reach the model over HTTP. Co-locating it with the server on the same node
# means localhost — no node-hostname discovery, no tunnel, and nothing
# long-running on the login node, which Barkla explicitly forbids. The GPU
# partitions used below have 96-168 CPU cores and 1.5-2TB RAM per node, so the
# pipeline's CPU work is free alongside the server.

#SBATCH --job-name=research-pipeline
#SBATCH --partition=gpu-h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --time=12:00:00
#SBATCH --output=pipeline_%j.log

# Partition choice (from Barkla's GPU doc §14.3/§14.6) — all users can use:
#   gpu-h100        4x H100 80GB   3-day limit   <- default here, 2 GPUs requested
#   gpu-l40s        2x L40S 48GB   3-day limit   <- needs --gres=gpu:2 and TP=2 (48GB < 60GB)
#   gpu-v100        4x V100 16GB   3-day limit   <- unusable here: no BF16 (SM 7.0), too little VRAM
#
# --gres=gpu:2 with TP=1: the server takes GPU 0 and the second card is left
# for generated experiments to use. Qwen3-Coder-30B-A3B is ~57GiB of BF16
# weights, which fits one H100 with a measured 150,272 tokens of KV cache
# (Barkla job 10274103) — so sharding it across both cards would spend the
# second GPU to buy context the pipeline does not need, and leave any
# GPU-needing experiment with nothing to run on. See _vllm_serve.sh for the
# measurements and for what to change if you do want TP=2.
#
# Low-priority (1-day limit, PRE-EMPTIBLE — job is killed and requeued when the
# owning group submits, so a half-finished run restarts from scratch):
#   gpu-a100-lowbig 4x A100 80GB
#   gpu-a-lowsmall  2x A100 80GB / 2x A40 48GB
# Don't request multiple partitions at once — the doc warns node specs differ.
#
# Do NOT request --cpus-per-task or --mem: Barkla's guide says cores and memory
# are allocated automatically from the GPU count, and asking manually fights it.

set -euo pipefail

QUESTION="${1:?usage: sbatch run_pipeline.sbatch \"research question\"}"

module purge
module load apptainer/1.3.6

PROJECT_DIR="$SLURM_SUBMIT_DIR"
WORK_DIR="/mnt/fastscratch/users/$USER/pipeline-runs/$SLURM_JOB_ID"
mkdir -p "$WORK_DIR"

export HF_HOME="/mnt/fastscratch/users/$USER/hf_cache"
VLLM_SIF="/mnt/fastscratch/users/$USER/containers/vllm.sif"
MODEL="Qwen/Qwen3-Coder-30B-A3B-Instruct"
# Overridable from the submitting environment, because which value is right
# depends on the card the job actually lands on rather than on this file: the
# ~57GiB of weights fit one 80GB H100 or A100 at TP=1, but a 48GB L40S needs
# both of its cards (TP=2). MAXLEN travels with it — see _vllm_serve.sh — and
# LLM_CONTEXT_WINDOW below is bound to the same number.
#
#   sbatch -p gpu-l40s-low --gres=gpu:2 --export=ALL,TP=2 scripts/slurm/run_pipeline.sbatch "..."
TP="${TP:-1}"

# gpu-l40s and gpu-h100 nodes are shared between users, so a fixed port can
# collide with someone else's server. Derive one from the job id instead.
PORT=$(( 8000 + SLURM_JOB_ID % 1000 ))

# shellcheck source=_vllm_serve.sh
source "$PROJECT_DIR/scripts/slurm/_vllm_serve.sh"

echo "Job $SLURM_JOB_ID on $SLURM_NODELIST, port $PORT, work dir $WORK_DIR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Always tear the server down: it holds an 80GB GPU, and a leaked process would
# keep it allocated for the rest of the job's wall clock.
cleanup() { [[ -n "${VLLM_PID:-}" ]] && kill "$VLLM_PID" 2>/dev/null || true; }
trap cleanup EXIT

vllm_serve_background
wait_for_vllm

cd "$PROJECT_DIR"

# Point the pipeline at the co-located server. These override .env, which is
# read by config.py via load_dotenv() — real environment variables win.
# Barkla §8: an OpenMP-linked library (numpy, scipy, sklearn, Stan) defaults to
# every core on the node — 96 on an H100 node — and overloads it. The pipeline
# itself is IO-bound; this ceiling is for the experiment code it runs.
# Each experiment's throwaway venv is thousands of small files, and Barkla's
# scratch/fastscratch carry inode quotas (300k / 500k). localscratch has none,
# is node-local NVMe, and is cleared automatically — and the venv is rebuilt per
# job anyway, so nothing worth keeping lives there (Barkla §5.1.7).
export CODER_VENV_ROOT="${CODER_VENV_ROOT:-/tmp/users/$USER/coder-venvs-${SLURM_JOB_ID}}"
mkdir -p "$CODER_VENV_ROOT"
# CODER_VENV_ROOT keeps the venvs off home; this keeps the *wheels* off it too.
# uv caches downloads under ~/.cache by default, and home carries an inode quota
# (Barkla: 150k soft / 200k hard) that a few hundred experiment venvs will
# exhaust — 2026-09-04 it stood at 198k files with ~/.cache/uv alone at 15 GB.
# Hitting it makes sandbox.ensure_experiment_env fail, and env-provisioning
# failures are deliberately never retried, so it ends a plan outright instead of
# costing a fix attempt (job 10426356: "Disk quota exceeded (os error 122)").
# Node-local scratch has no quota and is cleared automatically.
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/users/$USER/uv-cache-${SLURM_JOB_ID}}"
mkdir -p "$UV_CACHE_DIR"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-24}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"

export LLM_BASE_URL="http://localhost:${PORT}/v1"
export LLM_MODEL="$MODEL"
# Bound to the server's own --max-model-len rather than left to the code default
# or whatever .env happens to say. These two numbers must agree: every agent
# sizes its completion request against this client-side, and a value above what
# vLLM actually serves buys a 400 on the first long prompt instead of a
# completion. Keep in step with _vllm_serve.sh.
export LLM_CONTEXT_WINDOW="${MAXLEN:-131072}"

# The Coder Agent's fix-pattern store defaults to CODER_FIX_STORE_BACKEND=sqlite,
# whose dependency lives behind an optional extra — and a plain `uv sync` does
# not install it. Job 10279024 died on exactly this, after Literature,
# Hypothesis and the Planner had all completed: the run reached the Coder and
# exited on a missing optional dependency, throwing away twenty minutes of
# upstream work. Syncing it here costs nothing when it is already present.
uv sync --quiet --extra checkpoint-sqlite

uv run research-pipeline orchestrate "$QUESTION" \
    --download-dir "$WORK_DIR/papers" \
    --output-dir "$WORK_DIR/outputs"

echo "Done. Outputs in $WORK_DIR/outputs (paper PDFs: v1.pdf, v2.pdf, ...)."
