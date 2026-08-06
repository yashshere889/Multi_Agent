#!/bin/bash
# Example SLURM job to serve NVIDIA Nemotron 3 Nano behind an OpenAI-compatible
# endpoint (vLLM) on a Barkla GPU node. This is a starting template, not a
# verified script — adjust the partition/module/account lines below to match
# your Barkla account before submitting (`sbatch run_llm_server.sbatch`).
#
# Model card (source of the flags below):
#   https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
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

# TODO: activate wherever vLLM is installed on Barkla (venv, conda env, or container)
# source "$HOME/venvs/vllm/bin/activate"

MODEL="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"

# Deliberately no --served-model-name: vLLM then serves the model under its
# full HF id, which is exactly what LLM_MODEL defaults to in .env.example. If
# you do alias it, change LLM_MODEL to match or every request 404s.
#
# --max-model-len is set well below Nemotron's 1M ceiling: the pipeline's
# longest prompts are batched paper abstracts, and a smaller KV cache is what
# makes this fit on one GPU. Raise it (and set VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
# past 256k) only if a stage actually starts truncating.
#
# The reasoning parser keeps <think> traces out of `content` when
# LLM_ENABLE_THINKING=true. Agents strip traces themselves too
# (llm_json.strip_reasoning), so a server started without it still works.
vllm serve "$MODEL" \
    --host 0.0.0.0 \
    --port 8080 \
    --tensor-parallel-size 1 \
    --max-model-len 32768 \
    --max-num-seqs 8 \
    --trust-remote-code \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --reasoning-parser-plugin nano_v3_reasoning_parser.py \
    --reasoning-parser nano_v3
