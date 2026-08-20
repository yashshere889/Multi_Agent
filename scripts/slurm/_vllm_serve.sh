#!/bin/bash
# Shared vLLM launch/health-check helpers, sourced by run_llm_server.sbatch and
# run_pipeline.sbatch so the serve flags live in exactly one place.
#
# Barkla has no vLLM module (`module avail` only ships nemotron/nano-12b-v2,
# which is a transformers container, not an OpenAI-compatible server), so we
# run the official vLLM image under Apptainer. Build it once with
# scripts/slurm/build_vllm_sif.sh before submitting anything, and `module load
# apptainer/1.3.6` first — apptainer is not on PATH by default (Barkla §17.1.1).
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
    # --max-model-len 131072 and TP=1 are both measured, not assumed. On Barkla
    # job 10274103, Qwen3-Coder-30B-A3B served on a single 80GB card at
    # --gpu-memory-utilization 0.90 reported:
    #
    #     Model loading took 56.9342 GiB
    #     GPU KV cache size: 150,272 tokens
    #
    # i.e. the KV budget comfortably exceeds the 131072 requested, so the window
    # is real rather than silently clipped. LLM_CONTEXT_WINDOW in .env must
    # match this number — that is what every agent sizes completions against
    # client-side, and a mismatch buys a 400 from the server instead of a
    # completion.
    #
    # Why TP=1 when --gres asks for 2 GPUs: the model fits one card with room to
    # spare, so sharding it across both would spend the second card to buy
    # context this pipeline does not need — and leave generated experiments with
    # no GPU at all. TP=1 leaves GPU 1 free for them. Serving at TP=2 would
    # allow this model's full 262144 if some future run needs it; change TP in
    # the caller and LLM_CONTEXT_WINDOW together.
    #
    # 0.90 rather than 0.95: at TP=1 the weights alone are ~57GiB of the card's
    # 80GB, and the tighter margin left too little room for the CUDA graphs
    # captured after the KV cache is sized.
    #
    # No --trust-remote-code: vLLM resolves this model natively as
    # Qwen3MoeForCausalLM (confirmed in the job log above), so the flag would
    # buy nothing and enable arbitrary model-repo code execution for free.
    #
    # Deliberately no --served-model-name: vLLM then advertises the model under
    # its full HF id, which is what LLM_MODEL is set to. Alias it and every
    # request 404s.
    #
    # No --reasoning-parser: Qwen3-Coder-Instruct is not a reasoning model and
    # emits no <think> trace to parse. llm_json.strip_reasoning still strips one
    # if a future model produces it, so a server started without a parser is
    # safe either way.
    apptainer exec --nv \
        --bind /mnt/fastscratch,/mnt/scratch \
        --env HF_HOME="$HF_HOME" \
        "$VLLM_SIF" \
        vllm serve "$MODEL" \
            --host 0.0.0.0 \
            --port "$PORT" \
            --tensor-parallel-size "$TP" \
            --max-model-len 131072 \
            --gpu-memory-utilization 0.90 \
            --max-num-seqs 8 \
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
