# Running the research pipeline on Barkla

A practical, start-to-finish runbook for getting this pipeline onto Barkla and
running it — as a single question, or unattended across many. For the
pipeline's own architecture see [README.md](../README.md); this file only
covers the Barkla-specific mechanics.

## 0. Prerequisites

- A Barkla account and SSH access.
- A [Semantic Scholar API key](https://www.semanticscholar.org/product/api#api-key)
  (optional — without it that source is skipped with a warning, not a failure).
- Barkla's connection guide is explicit that long-running builds/downloads
  belong on a **viz node** (`barklaviz1.liv.ac.uk`), never the login node
  (`barklalogin1.liv.ac.uk`) — long tasks there get killed without warning.

## 1. Get the code onto Barkla

Storage guide: never build/run from `home` (75GB/100k-inode cap) — use
`fastscratch`.

```bash
ssh <user>@barklalogin1.liv.ac.uk
mkdir -p /mnt/fastscratch/users/$USER
cd /mnt/fastscratch/users/$USER
git clone <your-repo-url> multi-agent-langraph      # or rsync from your laptop
cd multi-agent-langraph
cp .env.example .env
```

`uv` isn't a Barkla-provided module (Barkla's own Python tooling is the system
interpreter, `miniforge3` modules, or `pixi` — none of which this project uses),
so install it once per account before `uv sync` will work:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc   # or open a new shell — picks up ~/.local/bin on PATH
```
`uv` manages its own Python versions independently of Barkla's system Python
(3.9.21, older than this project's `>=3.10` requirement), so nothing further is
needed there.

Fill in `.env` — at minimum `SEMANTIC_SCHOLAR_API_KEY`. Leave
`LLM_BACKEND`/`LLM_BASE_URL`/`LLM_MODEL` alone; every sbatch script below
overrides those with real environment variables at job time.

## 2. Pick a backend

Three ready-made job scripts, chosen by model size (and, for the 12B model,
whether you'd rather build your own `.venv` or use a module-provided container):

| | Model | GPU need | Extra setup |
|---|---|---|---|
| [`run_pipeline_hf.sbatch`](../scripts/slurm/run_pipeline_hf.sbatch) | Nemotron Nano **12B** v2, in-process transformers | fits `gpu-l40s` (48GB) and up | none — no container, no server |
| [`run_pipeline_hf_container.sbatch`](../scripts/slurm/run_pipeline_hf_container.sbatch) | Nemotron Nano **12B** v2, via Barkla's `nemotron` Apptainer module | fits `gpu-l40s` (48GB) and up | module-provided container, no build — use if `mamba_ssm` won't build in your own `.venv` |
| [`run_pipeline.sbatch`](../scripts/slurm/run_pipeline.sbatch) | Nemotron 3 Nano **30B A3B**, served by vLLM | needs an 80GB card (`gpu-h100`) | one-time Apptainer build |

Start with the 12B/HF path for a quick single run — it's strictly less
plumbing. For a large batch, throughput matters more; see §5's backend note.

### One-time setup — HF (12B) path

On a viz node:
```bash
ssh <user>@barklaviz1.liv.ac.uk
cd /mnt/fastscratch/users/$USER/multi-agent-langraph
export HF_HOME=/mnt/fastscratch/users/$USER/hf_cache
uv sync --extra huggingface
uv run hf download nvidia/NVIDIA-Nemotron-Nano-12B-v2
```

### One-time setup — HF via nemotron container (12B, no build)

On the login node (no build/GPU needed, just pip):
```bash
module load nemotron/nano-12b-v2
apptainer exec "$NEMOTRON_SIF" python3 -m pip install --user \
    -e "/mnt/fastscratch/users/$USER/multi-agent-langraph[huggingface]"
```
This uses Barkla's own container (mamba_ssm/torch/transformers already built
in) instead of your own `.venv` — pick this path if `mamba_ssm` won't build
for you directly. No separate "download the model" step: the wrapper
downloads weights on first use into a fastscratch cache and every later run
reuses them.

### One-time setup — vLLM (30B) path

Also on a viz node (no vLLM module on Barkla, so it runs under Apptainer):
```bash
ssh <user>@barklaviz1.liv.ac.uk
cd /mnt/fastscratch/users/$USER/multi-agent-langraph
bash scripts/slurm/build_vllm_sif.sh
export HF_HOME=/mnt/fastscratch/users/$USER/hf_cache
apptainer exec --bind /mnt/fastscratch \
    /mnt/fastscratch/users/$USER/containers/vllm.sif \
    hf download nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
```
Apptainer only auto-binds `$HOME` — without `--bind /mnt/fastscratch` the
container can't write to `$HF_HOME`, and `hf download` fails with `[Errno 30]
Read-only file system`.

## 3. Run one question

```bash
cd /mnt/fastscratch/users/$USER/multi-agent-langraph
sbatch scripts/slurm/run_pipeline_hf.sbatch "your research question"
# or, for the nemotron-container/12B path:
sbatch scripts/slurm/run_pipeline_hf_container.sbatch "your research question"
# or, for the vLLM/30B path:
sbatch scripts/slurm/run_pipeline.sbatch "your research question"
```

Monitor:
```bash
squeue -u $USER
tail -f pipeline_hf_<jobid>.log      # or pipeline_<jobid>.log
```

Results land under `/mnt/fastscratch/users/$USER/pipeline-runs/<jobid>/outputs/`
— `paper_v1.pdf`, `v2.pdf`, ... (Writer/Reviewer iterations), plus each
stage's JSON (`hypotheses_*.json`, `experiment_plan_*.json`,
`coder_agent_summary_*.json`).

## 4. What happens when the Coder Agent's generated code fails

Every experiment plan the Coder Agent turns into code goes through a bounded
**generate → execute → diagnose → fix → retry** loop before it gives up — up
to `CODER_MAX_FIX_ATTEMPTS` times (default 3, set in `.env`). A compile error,
a runtime exception, a bad `results.json`, or a static-safety-lint finding all
feed the concrete error back to the model and it regenerates. This is fully
automatic and needs nothing from you; each experiment's `coder_agent_summary_*.json`
records `fix_attempts`/`fix_history` per hypothesis so you can see what got
fixed and what didn't.

**High-complexity or GPU-needing plans stay manual by default.** These can't
be executed locally (no GPU on the node running the agent, or the plan is
flagged `high` complexity), so the Coder Agent writes a `run.sbatch` under
`experiments/<hypothesis_id>/` and stops — review it and `sbatch` it yourself.
This is a deliberate guardrail: nothing has ever run that code, and a bad job
spends GPU allocation on a cluster other people share.

To let the pipeline submit these itself during an unattended run, set in
`.env` (or export before a job):
```bash
CODER_AUTO_SUBMIT_SLURM=true
CODER_MAX_CONCURRENT_SLURM_JOBS=4   # checked against `squeue`, cluster-wide
CODER_MAX_SLURM_JOBS_PER_RUN=10     # per research question
```
Even with this on, submission is refused unless the generated code passes a
static safety check and an LLM pre-flight review — anything that trips either
gate falls back to the manual `run.sbatch`-for-review path, with a `reason`
in the summary saying which gate stopped it.

## 5. Run many questions unattended (batch mode)

This is what `orchestrate-batch` and `run_pipeline_batch.sbatch` are for —
reads a list of questions, runs the whole pipeline on each independently, and
keeps going if one question fails.

```bash
cd /mnt/fastscratch/users/$USER/multi-agent-langraph
cat > questions.txt <<'EOF'
# one question per line, blank lines and # comments are skipped
recent approaches to reducing hallucination in RAG systems
efficient fine-tuning methods for small language models
graph neural networks for molecular property prediction
EOF

# --array=0-N%C : N+1 array tasks, %C caps how many run concurrently
# SLICE_COUNT (2nd arg) must equal N+1
sbatch --array=0-1 scripts/slurm/run_pipeline_batch.sbatch questions.txt 2
```

Each array task takes a contiguous slice of the question list, runs its own
vLLM server, and works through its slice sequentially. All tasks share one
output root and one manifest:
```
/mnt/fastscratch/users/$USER/pipeline-runs/batch-<array-job-id>/outputs/batch_manifest.json
```

**Recommended backend for batch runs: vLLM (30B)**, not the in-process HF
path — throughput matters far more at this scale (dozens of LLM calls per
question × many questions), and vLLM's batching advantage compounds. The HF
path is fine for one-off runs, not a large sweep.

**Why a job array instead of one big job**: 100 questions × (literature
search + fix-loop attempts + Writer/Reviewer cycles) can exceed even
`gpu-h100`'s 3-day wall-clock cap, and one job failing would lose the whole
sweep. An array fails independently per slice.

**Resuming**: the manifest is rewritten after every question, not just at the
end. If a task is pre-empted or hits its wall clock, resubmit the exact same
`sbatch` command — questions already marked `"completed"` are skipped
automatically. Use `--no-resume` (passed through `orchestrate-batch`, not
exposed as an sbatch flag — edit the script if you need it) to force a
full re-run.

**Circuit breaker**: if `BATCH_MAX_CONSECUTIVE_FAILURES` (default 5)
questions fail in a row within one array task, that task stops rather than
burning through the rest of its slice — a sign something systemic broke (LLM
server down, expired key), not that the questions themselves are hard. Failed
questions stay `"pending"`, so a resubmit retries them once the underlying
issue is fixed.

### Checking results

```bash
cat /mnt/fastscratch/users/$USER/pipeline-runs/batch-<jobid>/outputs/batch_manifest.json
```
gives you, per question: status (`completed`/`failed`/`pending`), its own
output directory, and the final paper path or error. Each question's own
output directory underneath has the same per-stage JSON + paper PDFs as a
single run (§3).

**Worth checking specifically after a first run**: each experiment's
`fix_history` inside its `coder_agent_summary_*.json` — whether `resolved`
ever comes back `true` tells you whether the fix-loop is actually correcting
errors or just burning attempts on unfixable ones.

## 6. Recommended first run

Don't go straight to 100 questions. Submit a small slice first
(`--array=0-1` with 4-6 questions, as in §5's example), confirm the manifest
and outputs look right, and only then scale up the questions file and
`--array` range.

## Config reference

All of the following go in `.env` (see [`.env.example`](../.env.example) for
the full annotated list):

| Variable | Default | Purpose |
|---|---|---|
| `CODER_MAX_FIX_ATTEMPTS` | `3` | How many times the Coder Agent regenerates code after a failure before giving up |
| `CODER_AUTO_SUBMIT_SLURM` | `false` | Let the pipeline `sbatch` its own high-complexity/GPU experiment jobs |
| `CODER_MAX_CONCURRENT_SLURM_JOBS` | `4` | Cluster-wide cap (via `squeue`) on the Coder Agent's own auto-submitted jobs |
| `CODER_MAX_SLURM_JOBS_PER_RUN` | `10` | Per-question cap on auto-submitted jobs |
| `BATCH_OUTPUT_ROOT` | `outputs/batch` | Default root when `orchestrate-batch` is run without `--output-dir` |
| `BATCH_MAX_CONSECUTIVE_FAILURES` | `5` | Circuit breaker: consecutive question failures before a batch task stops |
