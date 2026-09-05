#!/bin/bash -l
# Run the Coder Agent's frozen benchmark corpus and print the score.
#
#   sbatch scripts/slurm/run_coder_benchmark.sbatch
#
# run_coder.sbatch replays *one* plan, which is how you debug a change.
# This replays all twelve (agents/coder/benchmark_plans/), which is how you find
# out whether the change helped: it produces one comparable number per run, so
# two of these runs can be diffed with
#
#   uv run research-pipeline coder-benchmark compare <before-dir> <after-dir>
#
# Read `interpretable`, not `completed` — see agents/coder/benchmark.py.
#
# Partition and TP are the two settings that actually matter here, and both are
# measured rather than guessed:
#
# - The model is ~57GiB of weights at TP=1, so it needs an 80GB card. Every idle
#   L40S partition on Barkla is 48GB and would OOM at TP=1; run_coder.sbatch
#   defaults to TP=2 for exactly that reason, splitting the model over both of
#   its cards. Here we take an 80GB A100 and serve at TP=1 instead, which leaves
#   the *second* GPU free for the generated experiments — the whole point when
#   what is being measured is whether they can train anything.
# - gpu-a-lowsmall mixes 2x A100 80GB nodes with 2x A40 48GB nodes under one
#   partition name, so `--gres=gpu:2` can land on an A40 and OOM. The `a100`
#   type is not optional.

#SBATCH --job-name=coder-benchmark
#SBATCH --partition=gpu-a-lowsmall
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:2
#SBATCH --time=1-00:00:00
#SBATCH --output=coder_benchmark_%j.log
# A benchmark that silently restarts from case one after a preemption would
# report a run that never happened. Better to stop, and re-run what is missing.
#SBATCH --no-requeue
#SBATCH -D ./
#SBATCH --export=ALL

set -uo pipefail

module purge
module load apptainer/1.3.6

FASTSCRATCH="/mnt/fastscratch/users/$USER"
export HF_HOME="${HF_HOME:-$FASTSCRATCH/hf_cache}"
VLLM_SIF="${VLLM_SIF:-$FASTSCRATCH/containers/vllm.sif}"
MODEL="${LLM_MODEL:-Qwen/Qwen3-Coder-30B-A3B-Instruct}"
PORT="${PORT:-$((8000 + RANDOM % 1000))}"
# TP=1 on an 80GB card: see the header. Overridable for a 48GB partition.
TP="${TP:-1}"
MAXLEN="${MAXLEN:-131072}"
export PATH="$HOME/.local/bin:$PATH"

RUN_DIR="${BENCH_RUN_DIR:-$FASTSCRATCH/coder-benchmark-runs/${SLURM_JOB_ID}}"
mkdir -p "$RUN_DIR"

export LLM_BASE_URL="http://localhost:${PORT}/v1"
export LLM_MODEL="$MODEL"
export LLM_CONTEXT_WINDOW="$MAXLEN"
# Each case gets its own experiments/outputs directory *under* RUN_DIR — the
# benchmark runner does that itself, so these are only the defaults it starts from.
export CODER_EXPERIMENTS_DIR="$RUN_DIR/experiments"
export CODER_OUTPUT_DIR="$RUN_DIR/outputs"
export CODER_VENV_ROOT="${CODER_VENV_ROOT:-/tmp/users/$USER/coder-venvs-${SLURM_JOB_ID}}"
mkdir -p "$CODER_VENV_ROOT"
# Same reasoning as run_coder.sbatch: CODER_VENV_ROOT keeps the venvs off home,
# this keeps uv's *wheels* off it too. Home carries a 150k-inode soft quota and
# was at 198k on 2026-09-04 with ~/.cache/uv alone at 15GB; exceeding it makes
# ensure_experiment_env fail, and env-provisioning failures are deliberately
# never retried, so it ends a plan outright instead of costing a fix attempt.
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/users/$USER/uv-cache-${SLURM_JOB_ID}}"
mkdir -p "$UV_CACHE_DIR"
# Barkla §8: otherwise an OpenMP-linked library in a generated experiment takes
# every core on the node.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
# TP=1 leaves GPU 1 idle, and this node is single-tenant compute already
# attached to this pipeline rather than a shared queue to jump — which is the
# exact condition this opt-in exists for (see CLAUDE.md). Without it every
# high-complexity case defers to run.sbatch and scores as "not interpretable",
# which measures the deferral policy rather than the agent.
export CODER_RUN_HIGH_COMPLEXITY_WHEN_GPU_AVAILABLE=true

echo "=== coder benchmark job $SLURM_JOB_ID on $(hostname) ==="
echo "run dir: $RUN_DIR"
echo "model:   $MODEL (TP=$TP, max-model-len=$MAXLEN)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

# shellcheck source=_vllm_serve.sh
source "$SLURM_SUBMIT_DIR/scripts/slurm/_vllm_serve.sh"
vllm_serve_background
trap 'kill $VLLM_PID 2>/dev/null || true' EXIT
wait_for_vllm || { echo "error: the model server never came up" >&2; exit 1; }

uv sync --quiet --extra checkpoint-sqlite
uv run research-pipeline coder-benchmark run "$RUN_DIR"
STATUS=$?

echo
echo "=== finished with status $STATUS ==="
# Scored again at the end even though `run` already printed it: if the job was
# cut short, this reports whatever cases did finish rather than nothing. Every
# case writes its own summary as it completes, so a partial run is still a
# readable result.
uv run research-pipeline coder-benchmark score "$RUN_DIR" || true
echo
echo "compare against another run with:"
echo "  uv run research-pipeline coder-benchmark compare <other-run-dir> $RUN_DIR"
exit $STATUS
