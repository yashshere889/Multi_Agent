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
    # --max-model-len is set to 262144, not the ~1M-token figure once assumed
    # here: that number was wrong, not just optimistic — Nemotron 3 Nano's own
    # config.json caps max_position_embeddings at 262144, and a value above
    # that fails vLLM's own startup validation outright (a hard architectural
    # ceiling, not a KV-cache/memory budget question). Confirmed on Barkla:
    # 08-15 jobs 10231751/10231760/10231770 all died in seconds with
    # "User-specified max_model_len (1048576) is greater than the derived
    # max_model_len (max_position_embeddings=262144.0)" before vLLM ever
    # started listening.
    #
    # 262144 is still ~4x smaller than the 1048576 this repo used to request,
    # so it should sit comfortably within TP=2's KV-cache budget — the reason
    # TP/--gres are 2, not 1, is unchanged: sharding the ~59GB of BF16 weights
    # across 2 H100s leaves each card mostly free for KV cache instead of
    # mostly consumed by weights. That said, the exact per-GPU KV budget on
    # Barkla is still unverified for this model, so if 262144 somehow doesn't
    # fit, vLLM refuses to start and exits within seconds rather than hanging,
    # which wait_for_vllm's `kill -0` check below catches immediately (no
    # 30-minute wait, no wasted job time). Check the log for vLLM's own "GPU
    # KV cache size: N tokens" line to see what actually fit, and lower this
    # to match if it ever errors again. Test via run_llm_server.sbatch (server
    # only, no pipeline) before trusting a value in a full run_pipeline.sbatch
    # job. LLM_CONTEXT_WINDOW in .env must match whatever value ends up
    # working here, since that's what coder_agent._bounded_max_tokens sizes
    # completions against client-side.
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
            --max-model-len 262144 \
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
