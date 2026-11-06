# The serving examples' environment, sourced on every node: the paths and the site's settings in one place.
# The defaults are the paths examples/README.md makes, for a site of GB200 NVL4 nodes (4 GPUs and 4 InfiniBand rails per
# node, compute nodes offline); a variable set before sourcing wins. Another site sets the paths and NCCL_IB_HCA, and
# NCCL_OVERRIDE='' to use PyTorch's own NCCL.
#
#   sglang_env    SGLang's environment: its venv, the patched SGLang, the nimble package
#   vllm_env      vLLM's environment: its venv, the patched vLLM overlay, the nimble package

: "${MODEL:=${SCRATCH:-$HOME}/models/DeepSeek-V3}"   # DeepSeek-V3 (FP8)
: "${NIMBLE_SITE:=}"                            # where the nimble-gpu wheel is installed ('' = the venv)
: "${SGLANG_VENV:=$HOME/sglang-venv}"           # SGLang's dependencies
: "${SGLANG_DIR:=$HOME/sglang-nimble}"          # made by patches/sglang/apply.sh DEST
: "${VLLM_VENV:=$HOME/vllm-venv}"               # vLLM 0.30.0 as pip installs it
: "${VLLM_SITE:=$HOME/vllm-nimble-site}"        # made by patches/vllm/apply.sh SITE --from-wheel WHEEL
: "${VLLM_EXTRAS:=$HOME/vllm-extras}"           # DeepGEMM for vLLM's block-FP8 layers (pip --target)
: "${NCCL_OVERRIDE=$HOME/nimble-assets/nccl}"   # one NCCL build, in pip's layout ('' = PyTorch's own)
: "${SGLANG_CACHE_SEED:=$HOME/nimble-assets/jit/sglang_cache_seed}"   # optional: compiled kernels copied in at start
: "${VLLM_CACHE_SEED:=$HOME/vllm_cache_seed}"   # compiled kernels, saved by vllm_server.sh for later jobs
: "${HF_HOME:=$HOME/nimble-assets/hf_home}"     # Hugging Face's cache (the nodes are offline)
: "${NCCL_IB_HCA:=mlx5_0,mlx5_1,mlx5_4,mlx5_5}"   # NCCL's NICs, the compute ones first
# vLLM's JIT caches, on node-local disk. Keep this path: FlashInfer's JIT builds record absolute paths, so the seed is
# reused only when restored here.
VLLM_CACHE_DIR="/tmp/vllm_cache_$USER"
export MODEL NIMBLE_SITE SGLANG_DIR VLLM_SITE HF_HOME

# site_check vllm|sglang: checks the paths a run needs before any node starts, and names the command that makes each
# missing one (examples/README.md has them all)
site_check() {
  local bad=0
  _need() { [ -e "$2" ] || { echo "site_env: $1=${!1} -- $3" >&2; bad=1; }; }
  _need MODEL "$MODEL/config.json" \
    "hf download deepseek-ai/DeepSeek-V3 --revision e815299b0bcbac849fa540c768ef21845365c9eb --local-dir $MODEL"
  [ -z "$NCCL_OVERRIDE" ] || _need NCCL_OVERRIDE "$NCCL_OVERRIDE/nvidia/nccl/lib/libnccl.so.2" \
    "pip install --no-deps --target $NCCL_OVERRIDE nvidia-nccl-cu13==2.30.7"
  if [ "$1" = vllm ]; then
    _need VLLM_VENV "$VLLM_VENV/bin/activate" "set it to the venv with vLLM 0.30.0 (examples/README.md, 3.)"
    _need VLLM_SITE "$VLLM_SITE/vllm" "patches/vllm/apply.sh $VLLM_SITE --from-wheel WHEEL"
    _need VLLM_EXTRAS "$VLLM_EXTRAS/deep_gemm" \
      "pip install --no-deps --target $VLLM_EXTRAS sgl-deep-gemm==0.1.7 apache-tvm-ffi==0.1.11"
  else
    _need SGLANG_VENV "$SGLANG_VENV/bin/activate" "set it to the venv with SGLang (examples/README.md, 4.)"
    _need SGLANG_DIR "$SGLANG_DIR/python/sglang" "patches/sglang/apply.sh $SGLANG_DIR"
  fi
  return $bad
}

_site_common() {
  module load cuda/13.2 >/dev/null 2>&1 || true
  export CC=gcc CXX=g++                                           # the JIT builds' host compilers
  export NCCL_IB_HCA TOKENIZERS_PARALLELISM=false
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1                  # the model and its tokenizer come from disk
  if [ -n "$NCCL_OVERRIDE" ]; then
    export LD_PRELOAD="$NCCL_OVERRIDE/nvidia/nccl/lib/libnccl.so.2" NCCL_DIR="$NCCL_OVERRIDE/nvidia/nccl"
  fi
}

sglang_env() {
  _site_common
  source "$SGLANG_VENV/bin/activate"
  export PYTHONPATH="${NCCL_OVERRIDE:+$NCCL_OVERRIDE:}$SGLANG_DIR/python${NIMBLE_SITE:+:$NIMBLE_SITE}"
  # JIT caches on node-local disk, copied from SGLANG_CACHE_SEED when it exists
  export SGLANG_CACHE_DIR="/tmp/nimble_examples_sglang_$USER"
  export EP_JIT_CACHE_DIR="$SGLANG_CACHE_DIR/deep_ep_jit" TRITON_CACHE_DIR="$SGLANG_CACHE_DIR/triton" \
         TORCHINDUCTOR_CACHE_DIR="$SGLANG_CACHE_DIR/inductor" SGLANG_JIT_CACHE_DIR="$SGLANG_CACHE_DIR/jit" \
         FLASHINFER_WORKSPACE_BASE="$SGLANG_CACHE_DIR/flashinfer"
  mkdir -p "$SGLANG_CACHE_DIR" "$EP_JIT_CACHE_DIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" \
           "$SGLANG_JIT_CACHE_DIR" "$FLASHINFER_WORKSPACE_BASE"
  [ -d "$SGLANG_CACHE_SEED" ] && cp -an "$SGLANG_CACHE_SEED/." "$SGLANG_CACHE_DIR/" 2>/dev/null
  rm -rf "$SGLANG_CACHE_DIR/deep_gemm/locks" "$SGLANG_CACHE_DIR/deep_gemm/tmp"
  return 0
}

vllm_env() {
  _site_common
  source "$VLLM_VENV/bin/activate"
  export PYTHONPATH="${NCCL_OVERRIDE:+$NCCL_OVERRIDE:}$VLLM_EXTRAS:$VLLM_SITE${NIMBLE_SITE:+:$NIMBLE_SITE}"
  export NCCL_CUMEM_ENABLE=1 VLLM_ENGINE_READY_TIMEOUT_S=3600     # a cold start compiles longer than vLLM's 600 s
  local cache="$VLLM_CACHE_DIR"
  export VLLM_CACHE_ROOT="$cache" FLASHINFER_WORKSPACE_BASE="$cache/flashinfer" TRITON_CACHE_DIR="$cache/triton" \
         TORCHINDUCTOR_CACHE_DIR="$cache/inductor" DG_JIT_CACHE_DIR="$cache/deep_gemm" EP_JIT_CACHE_DIR="$cache/deep_ep_jit"
  mkdir -p "$FLASHINFER_WORKSPACE_BASE" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$DG_JIT_CACHE_DIR"
  [ -d "$VLLM_CACHE_SEED" ] && cp -an "$VLLM_CACHE_SEED/." "$cache/" 2>/dev/null
  rm -rf "$DG_JIT_CACHE_DIR/locks" "$DG_JIT_CACHE_DIR/tmp"
  return 0
}

# vllm_save_caches HOST...: copies each node's compiled kernels into VLLM_CACHE_SEED, one node at a time, so the jobs
# after it start in minutes; the dedup mode starts from them
vllm_save_caches() {
  [ -n "$VLLM_CACHE_SEED" ] || return 0
  mkdir -p "$VLLM_CACHE_SEED"
  local h
  for h in "$@"; do srun -N 1 -n 1 -w "$h" cp -an "$VLLM_CACHE_DIR/." "$VLLM_CACHE_SEED/" 2>/dev/null; done
  rm -rf "$VLLM_CACHE_SEED/deep_gemm/locks" "$VLLM_CACHE_SEED/deep_gemm/tmp"
}
