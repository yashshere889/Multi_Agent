# Barkla command cheat sheet

Quick reference for day-to-day Barkla use with this repo: connecting, getting
code on/off the cluster, submitting/checking/cancelling jobs, and downloading
results. For the full setup walkthrough (backends, `.env`, Coder Agent SLURM
gating) see [BARKLA.md](BARKLA.md). Sourced from the official Barkla docs
(`~/Desktop/Barkla/*.pdf`) plus this repo's `scripts/slurm/`.

## 1. Connecting

| Purpose | Node | Command |
|---|---|---|
| Submit jobs, light editing | login | `ssh <user>@barklalogin1.liv.ac.uk` |
| Builds, big data transfers, GPU debug (<8h, ≤8 cores) | viz | `ssh <user>@barklaviz1.liv.ac.uk` (or `barklaviz2`) |

Never run builds, downloads, or production work on `barklalogin1` — it's
monitored and long/heavy tasks are killed without warning. Off-campus: connect
to the university VPN first (eduroam needs UoL credentials).

### Tunneling into a service running on a compute node

Compute nodes aren't reachable directly — forward a local port through the
login node to the node your job landed on:
```bash
ssh -L <local-port>:<compute-node>:<remote-port> <user>@barklalogin1.liv.ac.uk
```
Used for the web UI (`run_webapp.sbatch`, §7 of [BARKLA.md](BARKLA.md) — the
job's log prints the exact command) and, for interactive/dev use, a long-lived
model server (`run_llm_server.sbatch`). Get the node from `squeue -u $USER -o
"%N %j"` if you don't have the log handy. Re-run the tunnel after every job
resubmission — the node it points at changes each time.

## 2. Git — getting code on/off Barkla

Clone/pull happens **on the cluster**, from `fastscratch` (never `home` — 75GB/100k-inode cap):

```bash
ssh <user>@barklalogin1.liv.ac.uk
mkdir -p /mnt/fastscratch/users/$USER
cd /mnt/fastscratch/users/$USER
git clone <your-repo-url> multi-agent-langraph
cd multi-agent-langraph
git pull                      # update an existing checkout
git log --oneline -5          # sanity-check what's actually deployed
git status                    # before/after any run — catch stray local edits
```

If the cluster can't reach your git host directly, `rsync` from your laptop instead:
```bash
rsync -avz --exclude .venv --exclude outputs \
    ./ <user>@barklaviz1.liv.ac.uk:/mnt/fastscratch/users/<user>/multi-agent-langraph/
```
Run `rsync`/`scp` against a **viz node**, not the login node (data transfer is a
"lengthy" task).

## 3. Submitting jobs (batch)

```bash
sbatch scripts/slurm/run_pipeline.sbatch "your research question"        # vLLM/30B, single question
sbatch scripts/slurm/run_pipeline_hf.sbatch "your research question"     # HF/12B, single question
sbatch --array=0-1 scripts/slurm/run_pipeline_batch.sbatch questions.txt 2   # batch sweep
sbatch scripts/slurm/run_webapp.sbatch                                   # web UI, tunnel in from your laptop
```
Must be submitted from `scratch`/`fastscratch` (compute nodes can't write
anywhere else). Useful `sbatch`/script directives:

| Flag | Meaning |
|---|---|
| `-p <partition>` / `#SBATCH -p gpu-h100` | Partition (this repo uses `gpu-h100`; see §5) |
| `-N 1` | Nodes |
| `--gres=gpu:1` / `--gres=gpu:l40s:2` | GPU count / specific GPU type |
| `-t 12:00:00` | Wall-clock limit (repo default 12h; cluster max 3 days unless dedicated) |
| `-J <name>` | Job name |
| `-o out.log -e err.log` | stdout/stderr paths (default `slurm-%j.out`) |
| `--qos=low` (with `-p lowpriority`) | Opportunistic/low-priority QOS |
| `--qos=dedicated` (with a dedicated partition, e.g. `cooper`, `phi`) | Dedicated-partition QOS |

Interactive GPU shell (debugging, not for real runs):
```bash
srun -p gpu-l40s -N 1 --gres=gpu:1 --pty /bin/bash
```

**Per-account limits**: max 1000 total jobs (incl. array sub-jobs) queued/pending
at once; max 400 CPU cores running at once under the default `normal` QOS.
Hitting either gets `sbatch: error: ... Job violates accounting/QOS policy`.

## 4. Checking jobs

```bash
squeue -u $USER              # your jobs: JOBID, PARTITION, NAME, ST, TIME, NODES
squeue                       # everyone's jobs
squeue -j <jobid>             # one job's status
squeue -l                     # long form — shows pending REASON (e.g. QOSMaxCpuPerUserLimit)
sinfo -Nl                     # per-node state across all partitions
scancel <jobid>               # kill a job
```
Job states: `PD` pending → `CF` configuring → `R` running → `CG` completing →
`CD` completed / `F` failed. `F` usually means a bad path/permission on the
script or executable, or an unreadable input / unwritable output file.

Live log while running:
```bash
tail -f pipeline_<jobid>.log          # vLLM/30B path
tail -f pipeline_hf_<jobid>.log       # HF/12B path
```

## 5. GPU partitions in this repo

| Partition | GPUs | Wall-clock cap | Used by |
|---|---|---|---|
| `gpu-h100` | 4× H100 80GB per node (shared, common) | 3 days | `run_pipeline.sbatch` (vLLM/30B, `--gres=gpu:2`, TP=2 — 2 GPUs for KV-cache headroom, not because the model needs 2 to fit; see `scripts/slurm/_vllm_serve.sh`) |
| `gpu-l40s` | 2× L40S 48GB per node (shared, common) | 3 days | 12B HF path fits here too (`--gres=gpu:2`, TP=2 — here 2 GPUs *are* needed, 48GB < 60GB of weights) |

Don't request multiple partitions in one `sbatch`/`srun` call — node specs
differ across them and it won't behave as expected.

## 6. Downloading results back to your machine

Run transfers from a **viz node** (`barklaviz1`/`barklaviz2`), not the login node.

```bash
# single run
scp -r <user>@barklaviz1.liv.ac.uk:/mnt/fastscratch/users/<user>/pipeline-runs/<jobid>/outputs ./outputs

# whole batch sweep (resumable, skips unchanged files on rerun)
rsync -avz <user>@barklaviz1.liv.ac.uk:/mnt/fastscratch/users/<user>/pipeline-runs/batch-<jobid>/outputs/ ./batch-outputs/

# just the manifest, to check status before pulling everything
scp <user>@barklaviz1.liv.ac.uk:/mnt/fastscratch/users/<user>/pipeline-runs/batch-<jobid>/outputs/batch_manifest.json .
```
`rsync -avz` is preferred over `scp -r` for anything you'll re-download later
(retries, only transfers changed files — useful while a batch array is still
running and you want incremental pulls).

## 7. Sharing data with another Barkla user (optional)

```bash
groups                                          # your groupship(s)
id <username>                                   # someone else's
setfacl -R -m u:<username>:rx /mnt/fastscratch/users/$USER/multi-agent-langraph
getfacl /mnt/fastscratch/users/$USER/multi-agent-langraph   # verify
```
Grant `rx` (read/execute) only, never `w` — write access on a shared directory
puts your data at risk of accidental deletion by the other user.
