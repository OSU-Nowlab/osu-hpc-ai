# The serving examples' environment, sourced on every node: the paths and the site's settings in one place.
# The defaults are placeholders under $HOME, for a site of GB200 NVL4 nodes (4 GPUs and 4 InfiniBand rails per node,
# compute nodes offline); a variable set before sourcing wins. Another site sets the paths and NCCL_IB_HCA, and drops
# NCCL_OVERRIDE where PyTorch's own NCCL works (Horizon pins one NCCL build per process).
#
#   sglang_env    SGLang's environment: its venv, the patched SGLang, the nimble package
#   vllm_env      vLLM's environment: its venv, the patched vLLM overlay, the nimble package

: "${MODEL:=$HOME/nimble-assets/models/DeepSeek-V3}"   # DeepSeek-V3 (FP8), the model these were tested on
: "${NIMBLE_SITE:=}"                            # where the nimble-gpu wheel is installed ('' = the venv)
: "${SGLANG_VENV:=$HOME/venvs/sglang}"          # SGLang's dependencies
: "${SGLANG_DIR:=$HOME/sglang-nimble}"          # made by the frameworks bundle's sglang/apply.sh DEST
: "${VLLM_VENV:=$HOME/venvs/vllm}"              # vLLM 0.30.0 as pip installs it
: "${VLLM_SITE:=$HOME/vllm-nimble-site}"        # made by the frameworks bundle's vllm/apply.sh SITE --from-wheel WHEEL
: "${VLLM_EXTRAS:=$HOME/vllm-extras}"           # DeepGEMM for vLLM's block-FP8 layers (pip --target)
: "${NCCL_OVERRIDE:=$HOME/nimble-assets/pyoverride2}"   # one NCCL build, in the pip layout torch loads first
: "${SGLANG_CACHE_SEED:=$HOME/nimble-assets/jit/sglang_cache_seed}"   # compiled kernels (cold start: ~20 min)
: "${VLLM_CACHE_SEED:=$HOME/vllm_cache_seed}"
: "${HF_HOME:=$HOME/nimble-assets/hf_home}"     # Hugging Face's cache (the nodes are offline)
export MODEL NIMBLE_SITE SGLANG_DIR VLLM_SITE HF_HOME

_site_common() {
  module load cuda/13.2 >/dev/null 2>&1 || true
  export CC=gcc CXX=g++                                           # the JIT builds' host compilers
  export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_4,mlx5_5 TOKENIZERS_PARALLELISM=false
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1                  # the model and its tokenizer come from disk
  if [ -n "$NCCL_OVERRIDE" ]; then
    export LD_PRELOAD="$NCCL_OVERRIDE/nvidia/nccl/lib/libnccl.so.2" NCCL_DIR="$NCCL_OVERRIDE/nvidia/nccl"
  fi
}

sglang_env() {
  _site_common
  source "$SGLANG_VENV/bin/activate"
  export PYTHONPATH="${NCCL_OVERRIDE:+$NCCL_OVERRIDE:}$SGLANG_DIR/python${NIMBLE_SITE:+:$NIMBLE_SITE}"
  # JIT caches on node-local disk (ranks compiling into one shared-file-system cache race), seeded from scratch
  export SGLANG_CACHE_DIR="/tmp/nimble_examples_sglang_$USER"
  export EP_JIT_CACHE_DIR="$SGLANG_CACHE_DIR/deep_ep_jit" TRITON_CACHE_DIR="$SGLANG_CACHE_DIR/triton" \
         TORCHINDUCTOR_CACHE_DIR="$SGLANG_CACHE_DIR/inductor"
  mkdir -p "$SGLANG_CACHE_DIR" "$EP_JIT_CACHE_DIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
  [ -d "$SGLANG_CACHE_SEED" ] && cp -an "$SGLANG_CACHE_SEED/." "$SGLANG_CACHE_DIR/" 2>/dev/null
  rm -rf "$SGLANG_CACHE_DIR/deep_gemm/locks" "$SGLANG_CACHE_DIR/deep_gemm/tmp"
  return 0
}

vllm_env() {
  _site_common
  source "$VLLM_VENV/bin/activate"
  export PYTHONPATH="${NCCL_OVERRIDE:+$NCCL_OVERRIDE:}$VLLM_EXTRAS:$VLLM_SITE${NIMBLE_SITE:+:$NIMBLE_SITE}"
  export NCCL_CUMEM_ENABLE=1 VLLM_ENGINE_READY_TIMEOUT_S=3600     # a cold start compiles longer than vLLM's 600 s
  # The path the seed was built at: FlashInfer's JIT builds record absolute paths, so a seed copied anywhere else
  # recompiles their MoE kernel (~15 min, and only on the ranks that received tokens).
  local cache="/tmp/vllm_cache_$USER"
  export VLLM_CACHE_ROOT="$cache" FLASHINFER_WORKSPACE_BASE="$cache/flashinfer" TRITON_CACHE_DIR="$cache/triton" \
         TORCHINDUCTOR_CACHE_DIR="$cache/inductor" DG_JIT_CACHE_DIR="$cache/deep_gemm" EP_JIT_CACHE_DIR="$cache/deep_ep_jit"
  mkdir -p "$FLASHINFER_WORKSPACE_BASE" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$DG_JIT_CACHE_DIR"
  [ -d "$VLLM_CACHE_SEED" ] && cp -an "$VLLM_CACHE_SEED/." "$cache/" 2>/dev/null
  rm -rf "$DG_JIT_CACHE_DIR/locks" "$DG_JIT_CACHE_DIR/tmp"
  return 0
}
