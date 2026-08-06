#!/bin/bash
# Example SLURM job to launch an OpenAI-compatible LLM server (llama-server)
# on a Barkla GPU node. This is a starting template, not a verified script —
# adjust the partition/module/account lines below to match your Barkla
# account before submitting (`sbatch run_llm_server.sbatch`).
#
# Once running, find which node it landed on:
#   squeue -u $USER
# then tunnel to it from your laptop:
#   ssh -L 8080:<compute-node-hostname>:8080 <user>@barkla.liverpool.ac.uk
# and point LLM_BASE_URL=http://localhost:8080/v1 in your .env.

#SBATCH --job-name=llm-server
#SBATCH --partition=gpu            # TODO: set to Barkla's actual GPU partition name
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=llm_server_%j.log

set -euo pipefail

module purge
module load cuda                  # TODO: match an available `module avail cuda` entry

# TODO: point this at wherever llama-server is built/installed on Barkla
LLAMA_SERVER_BIN="$HOME/llama.cpp/build/bin/llama-server"

"$LLAMA_SERVER_BIN" \
    -hf unsloth/gemma-4-12b-it-GGUF:UD-Q4_K_XL \
    --jinja \
    -c 8192 \
    -ngl 999 \
    --host 0.0.0.0 \
    --port 8080
