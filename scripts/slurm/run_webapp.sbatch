#!/bin/bash -l
# Run the web UI (vLLM server + `research-pipeline serve`) inside one Barkla
# job, for interactively starting/watching runs from your laptop over an SSH
# tunnel — as opposed to run_pipeline.sbatch, which runs one question
# unattended via the `orchestrate` CLI and exits.
#
#   cd /mnt/fastscratch/users/$USER/multi-agent-langraph
#   sbatch scripts/slurm/run_webapp.sbatch
#
# Submit from scratch or fastscratch — Barkla's storage guide says not to submit
# from home, data or localscratch.
#
# One-time setup beyond build_vllm_sif.sh (see run_pipeline.sbatch):
#   uv sync --extra webapp --extra checkpoint-sqlite
#
# Co-located with vLLM on the same node for the same reason as
# run_pipeline.sbatch: the webapp needs localhost access to LLM_BASE_URL, and
# the app itself binds 127.0.0.1 only (it has no authentication and can start
# jobs / read files — see README's "Web UI" section) so a login-node server
# would be no use anyway. Reach it from your laptop with:
#
#   ssh -J <user>@barklalogin1.liv.ac.uk -L 8000:localhost:<port> <user>@<node>
#
# then open http://localhost:8000 in a browser. The compute node and port are
# only known once the job starts and change on every resubmission — watch the
# job log (or squeue -u $USER -n research-pipeline-webapp) for the exact
# command this job prints.
#
# The -J matters, and the forward target must be `localhost`, not the node.
# Because the app binds the *node's* loopback, the forward has to be set up by
# an ssh whose own endpoint is on that node — which is what jumping to it
# through the login node achieves. The obvious-looking
#
#   ssh -L 8000:<node>:<port> <user>@barklalogin1.liv.ac.uk     # WRONG
#
# asks the *login* node to open a socket to <node>:<port>, and nothing is
# listening on the node's external interface, so it fails with connection
# refused every time. If your ssh is too old for -J, chain it by hand instead:
#
#   ssh <user>@barklalogin1.liv.ac.uk 'ssh -f -N -L <port>:localhost:<port> <node>'
#   ssh -N -L 8000:localhost:<port> <user>@barklalogin1.liv.ac.uk
#
# run_llm_server.sbatch documents the plain `-L 8000:<node>:<port>` form and is
# right to: _vllm_serve.sh passes --host 0.0.0.0, so vLLM really is reachable on
# the node's external interface. The difference is deliberate, not an
# inconsistency to tidy away — vLLM is a model server, while this one can start
# jobs and read files with no authentication in front of it.

#SBATCH --job-name=research-pipeline-webapp
#SBATCH --partition=gpu-h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --time=08:00:00
#SBATCH --output=webapp_%j.log

# Same partition table as run_pipeline.sbatch (see there for the full VRAM
# reasoning). Stick to gpu-h100/gpu-l40s (3-day cap, not pre-emptible) rather
# than the low-priority partitions — a pre-empted job kills your browser
# session mid-review, not just a batch question.
#   gpu-h100   4x H100 80GB   <- default here, 2 GPUs (KV-cache headroom, not because 1 can't fit the model)
#   gpu-l40s   2x L40S 48GB   <- needs --gres=gpu:2 and TP=2 (48GB < 60GB of weights, a different reason)
#
# _vllm_serve.sh's --max-model-len targets 131072 tokens, which fits at TP=1
# with headroom — keep LLM_CONTEXT_WINDOW in step if you change
# that, or vLLM will refuse to start (see _vllm_serve.sh's comment).
#
# Do NOT request --cpus-per-task or --mem: Barkla's guide says cores and memory
# are allocated automatically from the GPU count, and asking manually fights it.

set -euo pipefail

module purge
module load apptainer/1.3.6

PROJECT_DIR="$SLURM_SUBMIT_DIR"
WORK_DIR="/mnt/fastscratch/users/$USER/pipeline-runs/webapp-$SLURM_JOB_ID"
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

# gpu-l40s and gpu-h100 nodes are shared between users, so fixed ports can
# collide with someone else's server. Derive both from the job id, offset so
# the webapp port never lands on the vLLM port.
PORT=$(( 8000 + SLURM_JOB_ID % 1000 ))
WEBAPP_PORT=$(( 9000 + SLURM_JOB_ID % 1000 ))

# shellcheck source=_vllm_serve.sh
source "$PROJECT_DIR/scripts/slurm/_vllm_serve.sh"

echo "Job $SLURM_JOB_ID on $SLURM_NODELIST, vLLM port $PORT, webapp port $WEBAPP_PORT, work dir $WORK_DIR"
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
# Each experiment's throwaway venv is thousands of small files, and Barkla's
# scratch/fastscratch carry inode quotas (300k / 500k). localscratch has none,
# is node-local NVMe, and is cleared automatically — and the venv is rebuilt per
# job anyway, so nothing worth keeping lives there (Barkla §5.1.7).
export CODER_VENV_ROOT="${CODER_VENV_ROOT:-/tmp/users/$USER/coder-venvs-${SLURM_JOB_ID}}"
mkdir -p "$CODER_VENV_ROOT"

export LLM_BASE_URL="http://localhost:${PORT}/v1"
export LLM_MODEL="$MODEL"

# Runs (PDFs, papers, generated code) go on fastscratch, not home — same
# 75GB/100k-inode reasoning as WORK_DIR above.
export WEBAPP_RUNS_DIR="$WORK_DIR/runs"
export WEBAPP_PORT="$WEBAPP_PORT"
# 127.0.0.1 is already the default; set explicitly so this job never binds
# wider even if .env was edited to something else.
export WEBAPP_HOST="127.0.0.1"

# A durable checkpointer so a pre-empted/killed job leaves something the
# webapp's own Resume button can pick back up, rather than starting the run
# over — worth the extra dependency for a long-lived interactive session in a
# way it isn't for a one-shot orchestrate call.
export CHECKPOINTER_BACKEND="sqlite"
export CHECKPOINTER_SQLITE_PATH="$WORK_DIR/checkpoints/pipeline.db"

cat <<EOF

================================================================
  Web UI ready.
    node: $SLURM_NODELIST
    port: $WEBAPP_PORT

  From your laptop:
    ssh -J $USER@barklalogin1.liv.ac.uk -L 8000:localhost:$WEBAPP_PORT $USER@$SLURM_NODELIST
    then open http://localhost:8000

  (The app binds this node's loopback, so the forward target is localhost and
   the -J jump is what puts the forward's endpoint on this node.)

  Runs land under: $WORK_DIR/runs
================================================================

EOF

uv run --extra webapp --extra checkpoint-sqlite research-pipeline serve --host 127.0.0.1 --port "$WEBAPP_PORT"
