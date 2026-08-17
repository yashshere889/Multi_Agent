#!/bin/bash
# Run the pipeline over MANY research questions as a SLURM job array — the
# unattended-sweep counterpart to run_pipeline.sbatch's single question.
#
#   cd /mnt/fastscratch/users/$USER/multi-agent-langraph
#   # one question per line; blank lines and #-comments are skipped
#   vim questions.txt
#   sbatch --array=0-9%3 scripts/slurm/run_pipeline_batch.sbatch questions.txt 10
#
# Each array task takes a contiguous slice of the question list and works
# through it sequentially, writing into one shared batch_manifest.json. The
# manifest is rewritten after every question, so a task that hits its wall
# clock (or gets pre-empted) loses at most the question it was mid-way
# through — resubmit the same command and finished questions are skipped.
#
# SLICE_COUNT ($2) must equal the number of array tasks, i.e. --array=0-(N-1)
# means SLICE_COUNT=N. The %N suffix on --array caps how many run at once;
# keep it modest, these are whole GPUs on a shared cluster.
#
# Why an array rather than one long job: 100 questions x (literature search +
# codegen/fix attempts + writer/reviewer cycles) can exceed even gpu-h100's
# 3-day cap, and a single job failing loses the whole sweep. Why a vLLM server
# per task rather than one shared server job: a shared server is a single point
# of failure for every task at once, and the model load cost amortizes fine
# across a slice of ~10-20 questions.

#SBATCH --job-name=research-pipeline-batch
#SBATCH --partition=gpu-h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --time=3-00:00:00
#SBATCH --output=pipeline_batch_%A_%a.log

# See run_pipeline.sbatch for the full partition table and the VRAM reasoning.
# --gres=gpu:2/TP=2 here is for KV-cache headroom (_vllm_serve.sh targets
# 262144's --max-model-len, sized for TP=2's budget), not because the 30B
# model needs 2 GPUs to fit on gpu-h100 — but note this is PER ARRAY TASK: at
# --array=0-N%C, C concurrent tasks now hold 2*C GPUs instead of C, on a
# partition shared with everyone else on Barkla. Pick %C with that in mind, or
# drop back to --gres=gpu:1/TP=1 (and lower _vllm_serve.sh's --max-model-len
# to match — see its comment) if a sweep doesn't need the extra context.
# Do NOT request --cpus-per-task or --mem — Barkla allocates both from the GPU
# count.

set -euo pipefail

QUESTIONS_FILE="${1:?usage: sbatch --array=0-N%C run_pipeline_batch.sbatch questions.txt SLICE_COUNT}"
SLICE_COUNT="${2:?usage: sbatch --array=0-N%C run_pipeline_batch.sbatch questions.txt SLICE_COUNT}"

module purge
module load apptainer/1.3.6

PROJECT_DIR="$SLURM_SUBMIT_DIR"
# Keyed on the ARRAY job id, not the task id, so every task in the sweep shares
# one output root and therefore one manifest.
WORK_DIR="/mnt/fastscratch/users/$USER/pipeline-runs/batch-$SLURM_ARRAY_JOB_ID"
mkdir -p "$WORK_DIR"

export HF_HOME="/mnt/fastscratch/users/$USER/hf_cache"
VLLM_SIF="/mnt/fastscratch/users/$USER/containers/vllm.sif"
MODEL="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
TP=2

# Derived from the task's own job id: array tasks land on shared nodes and a
# fixed port would collide with a sibling task.
PORT=$(( 8000 + SLURM_JOB_ID % 1000 ))

# shellcheck source=_vllm_serve.sh
source "$PROJECT_DIR/scripts/slurm/_vllm_serve.sh"

echo "Array task $SLURM_ARRAY_TASK_ID/$SLICE_COUNT (job $SLURM_JOB_ID) on $SLURM_NODELIST, port $PORT"
echo "Questions: $QUESTIONS_FILE, shared output root: $WORK_DIR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cleanup() { [[ -n "${VLLM_PID:-}" ]] && kill "$VLLM_PID" 2>/dev/null || true; }
trap cleanup EXIT

vllm_serve_background
wait_for_vllm

cd "$PROJECT_DIR"

export LLM_BASE_URL="http://localhost:${PORT}/v1"
export LLM_MODEL="$MODEL"

# Auto-submission of generated GPU experiments stays OFF by default. To let the
# Coder Agent submit its own high-complexity/GPU experiment jobs during an
# unattended sweep, export CODER_AUTO_SUBMIT_SLURM=true here — it still refuses
# anything the static safety check flags, and both job caps still apply.
# export CODER_AUTO_SUBMIT_SLURM=true

uv run research-pipeline orchestrate-batch \
    --questions-file "$QUESTIONS_FILE" \
    --output-dir "$WORK_DIR/outputs" \
    --download-dir "$WORK_DIR/papers" \
    --slice-index "$SLURM_ARRAY_TASK_ID" \
    --slice-count "$SLICE_COUNT"

echo "Done. Manifest: $WORK_DIR/outputs/batch_manifest.json"
