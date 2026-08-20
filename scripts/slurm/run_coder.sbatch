#!/bin/bash -l
# Run ONLY the Coder Agent, against an experiment plan that already exists.
#
#   sbatch scripts/slurm/run_coder.sbatch /path/to/experiment_plan_<ts>.json
#
# The full pipeline spends twenty-odd minutes on Literature, Hypothesis and the
# Planner before the Coder ever runs, which is a long way to travel to test a
# change to the Coder. Every plan the Planner writes is a complete, replayable
# input (see README's "Chaining agents individually"), so this replays one.
#
# Also the honest way to A/B a Coder change: the same plan through the old and
# new agent, comparing how many fix_history entries were spent on environment
# problems rather than on the code.

#SBATCH --job-name=coder-only
#SBATCH --partition=gpu-l40s-low
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --time=4:00:00
#SBATCH --output=coder_only_%j.log
#SBATCH --no-requeue
#SBATCH -D ./
#SBATCH --export=ALL

set -uo pipefail

PLAN="${1:?usage: sbatch run_coder.sbatch <experiment_plan.json>}"
[[ -f "$PLAN" ]] || { echo "error: plan file not found: $PLAN" >&2; exit 2; }
PLAN="$(cd "$(dirname "$PLAN")" && pwd)/$(basename "$PLAN")"

module purge
module load apptainer/1.3.6

FASTSCRATCH="/mnt/fastscratch/users/$USER"
export HF_HOME="${HF_HOME:-$FASTSCRATCH/hf_cache}"
VLLM_SIF="${VLLM_SIF:-$FASTSCRATCH/containers/vllm.sif}"
MODEL="${LLM_MODEL:-Qwen/Qwen3-Coder-30B-A3B-Instruct}"
PORT="${PORT:-$((8000 + RANDOM % 1000))}"
# Defaults suit a 48GB card (both needed); override TP=1 on an 80GB partition.
TP="${TP:-2}"
MAXLEN="${MAXLEN:-131072}"
export PATH="$HOME/.local/bin:$PATH"

RUN_DIR="$FASTSCRATCH/coder-only-runs/${SLURM_JOB_ID}"
mkdir -p "$RUN_DIR"

export LLM_BASE_URL="http://localhost:${PORT}/v1"
export LLM_MODEL="$MODEL"
export LLM_CONTEXT_WINDOW="$MAXLEN"
export CODER_EXPERIMENTS_DIR="$RUN_DIR/experiments"
export CODER_OUTPUT_DIR="$RUN_DIR/outputs"
export CODER_VENV_ROOT="${CODER_VENV_ROOT:-/tmp/users/$USER/coder-venvs-${SLURM_JOB_ID}}"
mkdir -p "$CODER_VENV_ROOT"
# Barkla §8: otherwise an OpenMP-linked library in a generated experiment takes
# every core on the node.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"

echo "=== coder-only job $SLURM_JOB_ID on $(hostname) ==="
echo "plan:    $PLAN"
echo "run dir: $RUN_DIR"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

# shellcheck source=_vllm_serve.sh
source "$SLURM_SUBMIT_DIR/scripts/slurm/_vllm_serve.sh"
vllm_serve_background
trap 'kill $VLLM_PID 2>/dev/null || true' EXIT
wait_for_vllm || { echo "error: the model server never came up" >&2; exit 1; }

uv sync --quiet --extra checkpoint-sqlite
uv run research-pipeline coder --from-file "$PLAN" --output-dir "$CODER_OUTPUT_DIR"
STATUS=$?

echo
echo "=== finished with status $STATUS ==="
echo "summary:  $CODER_OUTPUT_DIR/coder_agent_summary_*.json"
echo "per-plan: $CODER_EXPERIMENTS_DIR/*/"
exit $STATUS
