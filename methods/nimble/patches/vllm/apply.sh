#!/bin/bash
# NIMBLE's backends in vLLM -- `--all2all-backend nimble` (NIMBLE's MoE dispatcher over its AllToAllV) and `nccl_a2a`
# (the same dispatcher over NCCL's all-to-all, the baseline): patch a vLLM package (BASE's version only). The patch
# series (patches/*.patch, applied in order to vLLM's own files) adds the backends' names, the manager's construction,
# the prepare / finalize factory entry, the CUDA-graph switch and the TRT-LLM experts' admission; the classes come from
# the nimble package (nimble.integrations.vllm): put <NIMBLE>/python on PYTHONPATH.
#   apply.sh SITE                      patch the vllm/ package under SITE (e.g. <venv>/lib/python3.12/site-packages)
#   apply.sh SITE --from-wheel WHEEL   first unpack BASE's wheel (sha256-checked) into a new directory SITE: an overlay
#                                      to put ahead of a venv's site-packages on PYTHONPATH, leaving the venv untouched
#   apply.sh SITE --check              verify only (non-zero when the series is not applied)
# Never run it while a job uses the installation.
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
SP=${1:?usage: apply.sh SITE [--check | --from-wheel WHEEL]}
MODE=${2:-apply}
if [ "$MODE" = "--from-wheel" ]; then
  WHEEL=${3:?usage: apply.sh SITE --from-wheel WHEEL}
  [ ! -e "$SP" ] || { echo "ABORT: $SP exists; --from-wheel makes a new directory"; exit 2; }
  want_sha=$(sed -n 's/^sha256=//p' "$HERE/BASE")
  [ "$(sha256sum "$WHEEL" | cut -c1-64)" = "$want_sha" ] || { echo "ABORT: $WHEEL is not BASE's wheel"; exit 2; }
  mkdir -p "$SP"
  python3 -m zipfile -e "$WHEEL" "$SP"
  echo "unpacked $(basename "$WHEEL") into $SP"
  MODE=apply
fi
V=$SP/vllm
[ -d "$V" ] || { echo "ABORT: no vllm package under $SP"; exit 2; }
want=$(sed -n 's/^vllm==//p' "$HERE/BASE")
have=$(sed -n 's/^Version: //p' "$SP"/vllm-*.dist-info/METADATA 2>/dev/null | head -1)
[ "$have" = "$want" ] || { echo "ABORT: vLLM $have under $SP; the series is for $want (BASE)"; exit 2; }
applied() {  # every patch of the series applies in reverse: it is in
  local p
  for p in "$HERE"/patches/*.patch; do
    patch -d "$SP" -p1 -R --dry-run -s -f < "$p" > /dev/null 2>&1 || return 1
  done
}
if [ "$MODE" = "--check" ]; then
  applied || { echo "vLLM under $SP: the series is not applied"; exit 1; }
  echo "vLLM $have under $SP: NIMBLE's series is applied (the classes come from the nimble package)"
  exit 0
fi
[ "$MODE" = apply ] || { echo "apply.sh: unknown option $MODE" >&2; exit 2; }
if applied; then
  echo "the series is already applied"
else
  for p in "$HERE"/patches/*.patch; do
    patch -d "$SP" -p1 --dry-run -s -f < "$p" > /dev/null 2>&1 || {
      echo "ABORT: $(basename "$p") does not apply (another series is in: reinstall BASE's wheel)"; exit 2; }
  done
  for p in "$HERE"/patches/*.patch; do patch -d "$SP" -p1 -s -f < "$p"; echo "applied $(basename "$p")"; done
fi
find "$V/distributed/device_communicators" "$V/model_executor/layers/fused_moe" "$V/config" -name __pycache__ -type d \
  -prune -exec rm -rf {} + 2>/dev/null || true
"$0" "$SP" --check
