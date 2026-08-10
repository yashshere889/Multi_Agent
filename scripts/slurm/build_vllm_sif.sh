#!/bin/bash
# Build the vLLM Apptainer image. Run this ONCE, interactively, before
# submitting any job:
#
#   ssh <user>@barklaviz1.liv.ac.uk        # NOT barklalogin1 — see below
#   bash scripts/slurm/build_vllm_sif.sh
#
# Barkla's connection guide is explicit that lengthy builds and data transfers
# belong on the viz nodes (barklaviz1/barklaviz2), not the login node, where
# long-running tasks "will be terminated without warning". This build pulls and
# converts a multi-GB image, so it is exactly that kind of task.
#
# The .sif goes on fastscratch, not home: home is 75GB/100k files and the image
# is several GB. It also collapses what would otherwise be ~100k inodes of
# Python environment into a single file (Apptainer guide §17.1).

set -euo pipefail

module purge
module load apptainer/1.3.6

SIF_DIR="/mnt/fastscratch/users/$USER/containers"
mkdir -p "$SIF_DIR"

# Pin a release rather than :latest so a rebuild months from now doesn't
# silently change vLLM under the pipeline. Bump this deliberately.
#
# Nemotron 3 Nano is a hybrid Mamba-2 model; support for it is recent, so if
# the server errors on an unknown architecture, move to a newer tag here
# before assuming the model or the flags are wrong.
apptainer build "$SIF_DIR/vllm.sif" docker://vllm/vllm-openai:v0.11.0

echo
echo "Built: $SIF_DIR/vllm.sif"
echo "Now pre-download the model (also on this node, not the login node):"
echo "  export HF_HOME=/mnt/fastscratch/users/\$USER/hf_cache"
echo "  apptainer exec --bind /mnt/fastscratch $SIF_DIR/vllm.sif hf download <model-id>"
