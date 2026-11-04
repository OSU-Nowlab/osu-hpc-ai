#!/bin/bash
# NIMBLE's SGLang integration: clone the public SGLang, check out the commit the patches are made for (BASE) and apply
# them. The serving patches add the nimble and nccl_a2a MoE all-to-all backends, which import NIMBLE's dispatcher from
# the nimble package (nimble.integrations.sglang). --harness also applies NIMBLE's CI harness: seeded prompts, the
# greedy tokens' dump and the per-layer timing hooks, with their timing instrument.
#
#   apply.sh DEST [--harness]      DEST: a new directory
#
# Then install DEST's SGLang as usual, for example pip install -e DEST/python.
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
DEST=${1:?usage: apply.sh DEST [--harness]}
case "${2:-}" in "" | --harness) ;; *) echo "usage: apply.sh DEST [--harness]" >&2; exit 2 ;; esac
BASE=$(head -1 "$HERE/BASE")
git clone -q https://github.com/sgl-project/sglang "$DEST"
git -C "$DEST" checkout -q "$BASE"
for p in "$HERE"/serving/*.patch; do git -C "$DEST" apply "$p"; done
if [ "${2:-}" = --harness ]; then
  for p in "$HERE"/harness/*.patch; do git -C "$DEST" apply "$p"; done
  cp "$HERE/harness/a2a_timing.py" "$DEST/python/sglang/srt/layers/moe/token_dispatcher/"
fi
echo "apply.sh: SGLang ${BASE:0:9} with NIMBLE's patches in $DEST"
