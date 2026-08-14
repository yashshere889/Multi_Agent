#!/bin/bash
# Shared vLLM launch/health-check helpers, sourced by run_llm_server.sbatch and
# run_pipeline.sbatch so the serve flags live in exactly one place.
#
# Barkla has no vLLM module (`module avail` only ships nemotron/nano-12b-v2,
# which is a transformers container, not an OpenAI-compatible server), so we
# run the official vLLM image under Apptainer. Build it once with
# scripts/slurm/build_vllm_sif.sh before submitting anything.
#
# Expects these to be set by the caller:
#   VLLM_SIF   path to the built .sif
#   MODEL      HF repo id to serve (must match LLM_MODEL in .env)
#   PORT       port to bind on the compute node
#   TP         --tensor-parallel-size (must equal the GPU count requested)
#   HF_HOME    HuggingFace cache, on fastscratch (NOT home — 100k inode quota)

vllm_serve_background() {
    # --nv passes the host NVIDIA driver into the container (Apptainer guide
    # §17.1.3); no CUDA module needed, the container carries its own toolkit.
    #
    # Barkla's home/scratch/fastscratch are separate mounts, and Apptainer only
    # binds $HOME by default — fastscratch must be bound explicitly or the
    # model cache silently misses and re-downloads into the container's tmpfs.
    #
    # --max-model-len targets Nemotron's actual ~1M-token card ceiling: on one
    # 80GB H100 that's not deployable (the ~59GB BF16 weights alone leave too
    # little KV-cache headroom), which is why TP/--gres above are 2, not 1 —
    # sharding the weights across 2 H100s leaves each card mostly free for KV
    # cache instead of mostly consumed by weights. This has NOT been verified
    # against the real per-GPU KV budget on Barkla (this repo's TP=2 precedent
    # is only for the 12B model on gpu-l40s, not this 30B hybrid Mamba+attention
    # model on gpu-h100) — if 1048576 doesn't fit, vLLM refuses to start and
    # exits within seconds rather than hanging, which wait_for_vllm's `kill -0`
    # check below catches immediately (no 30-minute wait, no wasted job time).
    # Check the log for vLLM's own "GPU KV cache size: N tokens" line to see
    # what actually fit, and lower this to match if it errored. Test via
    # run_llm_server.sbatch (server only, no pipeline) before trusting this in
    # a full run_pipeline.sbatch job. LLM_CONTEXT_WINDOW in .env must match
    # whatever value ends up working here, since that's what
    # coder_agent._bounded_max_tokens sizes completions against client-side.
    #
    # Deliberately no --served-model-name: vLLM then advertises the model under
    # its full HF id, which is what LLM_MODEL is set to. Alias it and every
    # request 404s.
    #
    # No --reasoning-parser: LLM_ENABLE_THINKING defaults to false, and the
    # nano_v3 parser needs a plugin .py from the model card in the job's CWD.
    # llm_json.strip_reasoning strips any trace that arrives anyway, so a
    # server started without the parser is safe. Add it back only if you turn
    # thinking on.
    apptainer exec --nv \
        --bind /mnt/fastscratch,/mnt/scratch \
        --env HF_HOME="$HF_HOME" \
        "$VLLM_SIF" \
        vllm serve "$MODEL" \
            --host 0.0.0.0 \
            --port "$PORT" \
            --tensor-parallel-size "$TP" \
            --max-model-len 1048576 \
            --gpu-memory-utilization 0.95 \
            --max-num-seqs 8 \
            --trust-remote-code \
        &
    VLLM_PID=$!
}

# Model load is minutes, not seconds (60GB of BF16 weights off Lustre), and a
# failed load exits rather than ever listening — so poll the endpoint and also
# watch that the process is still alive, instead of a fixed sleep.
wait_for_vllm() {
    local url="http://localhost:${PORT}/v1/models"
    local deadline=$(( SECONDS + 1800 ))
    while (( SECONDS < deadline )); do
        if ! kill -0 "$VLLM_PID" 2>/dev/null; then
            echo "ERROR: vLLM exited during startup — see the log above." >&2
            return 1
        fi
        if curl -sf "$url" >/dev/null 2>&1; then
            echo "vLLM is serving on port ${PORT}:"
            curl -s "$url"
            return 0
        fi
        sleep 10
    done
    echo "ERROR: vLLM did not become ready within 30 minutes." >&2
    return 1
}
