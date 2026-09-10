#!/bin/bash -l
# Run ONE question end to end against an already-running model server.
#
#   scripts/slurm/run_one_question.sh "your research question" [tag]
#
# The point is turnaround. run_pipeline_batch.sbatch starts its own vLLM server
# inside every job, so a one-question change costs a queue wait plus several
# minutes of model loading before the first token — a poor trade when what you
# are debugging is a Python decision the model never sees. This attaches to the
# long-lived server from run_llm_server.sbatch instead, so the same question
# costs only the pipeline.
#
# Start the server once:      sbatch scripts/slurm/run_llm_server.sbatch
# Then run as many as you like; each writes its own output directory.
#
# Runs under srun on a CPU node by default — generated experiments here are
# low/medium complexity and CPU-bound, and anything needing a GPU defers to
# run.sbatch as it always does. Set SRUN=0 to run in the current allocation.
set -euo pipefail

QUESTION="${1:?usage: run_one_question.sh \"question\" [tag]}"
TAG="${2:-$(date -u +%H%M%S)}"

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
S="/mnt/fastscratch/users/$USER"

# --- find the server ------------------------------------------------------
# From squeue rather than from a file the caller has to keep in step: the job
# id is the source of truth for both the node and the port, since the port is
# derived from the id.
JOB=$(squeue -u "$USER" -n llm-server -h -t RUNNING -o "%i" | head -1)
if [[ -z "$JOB" ]]; then
    echo "No running llm-server job. Start one with:" >&2
    echo "    sbatch scripts/slurm/run_llm_server.sbatch" >&2
    exit 1
fi
NODE=$(squeue -j "$JOB" -h -o "%N")
PORT=$(( 8000 + JOB % 1000 ))
BASE_URL="http://$NODE:$PORT/v1"

if ! curl -sf --max-time 10 "$BASE_URL/models" >/dev/null; then
    echo "llm-server $JOB is running on $NODE but not serving yet on $PORT." >&2
    echo "It loads ~57GiB of weights first; watch llm_server_$JOB.log." >&2
    exit 1
fi

# --- environment ----------------------------------------------------------
# Same choices run_pipeline_batch.sbatch makes, and for the same reasons: venvs
# and the uv cache go to node-local scratch because home carries an inode quota
# that a few hundred experiment venvs exhaust, and a provisioning failure ends a
# plan outright rather than costing a fix attempt.
export LLM_BASE_URL="$BASE_URL"
export LLM_MODEL="${LLM_MODEL:-Qwen/Qwen3-Coder-30B-A3B-Instruct}"
export LLM_CONTEXT_WINDOW="${LLM_CONTEXT_WINDOW:-131072}"
export CODER_VENV_ROOT="${CODER_VENV_ROOT:-/tmp/users/$USER/coder-venvs-debug}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/users/$USER/uv-cache-debug}"
mkdir -p "$CODER_VENV_ROOT" "$UV_CACHE_DIR"

WORK_DIR="$S/pipeline-runs/debug-$TAG"
export CODER_EXPERIMENTS_DIR="$WORK_DIR/experiments"
mkdir -p "$CODER_EXPERIMENTS_DIR"

echo "server:   $BASE_URL (job $JOB on $NODE)"
echo "question: $QUESTION"
echo "output:   $WORK_DIR"
echo

cd "$PROJECT_DIR"
RUN=(uv run research-pipeline orchestrate "$QUESTION"
     --output-dir "$WORK_DIR/outputs"
     --download-dir "$WORK_DIR/papers")

if [[ "${SRUN:-1}" == "1" ]]; then
    srun --partition="${DEBUG_PARTITION:-cpu-l40s-low}" \
         --time="${DEBUG_TIME:-02:00:00}" \
         --cpus-per-task="${DEBUG_CPUS:-8}" \
         --mem="${DEBUG_MEM:-32G}" \
         --job-name="debug-$TAG" \
         "${RUN[@]}"
else
    "${RUN[@]}"
fi

echo
echo "=== provenance ==="
python3 "$PROJECT_DIR/scripts/summarize_run.py" "$WORK_DIR" || true
