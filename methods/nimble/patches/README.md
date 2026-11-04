# NIMBLE 1.0.0: the SGLang and vLLM patches

`nimble.integrations`, in the `nimble-gpu` 1.0.0 wheel, runs NIMBLE's MoE dispatcher inside SGLang and vLLM.
The frameworks recognise its backends only after the patches here are applied. Install the wheel in the framework's
environment first:

```bash
pip install nimble_gpu-1.0.0-py3-none-manylinux_2_28_aarch64.whl
```

## SGLang, commit `fbf8f1d`

- **Apply:** `sglang/apply.sh DEST` clones the public SGLang into the new directory `DEST`, checks out the base
  commit (`sglang/BASE`) and applies `sglang/serving/`: the `nimble` and `nccl_a2a` MoE all-to-all backends and one
  upstream fix. Then install `DEST`'s SGLang as you normally would, for example `pip install -e DEST/python`.
- **Run:** `--moe-a2a-backend nimble`, or `nccl_a2a` for the same dispatcher over NCCL, with `--disable-cuda-graph`.
  Tested with DP attention and expert parallelism (`--tp-size N --ep-size N --dp-size N --enable-dp-attention`),
  DeepGEMM experts (`--moe-runner-backend deep_gemm`) and DeepSeek-V3 FP8.
- **Mode:** `SGLANG_NIMBLE_DISPATCHER=fused` (the default) or `dedup`. The dedup mode also needs `NIMBLE_DEDUP=1` on
  every rank; `SGLANG_NIMBLE_MAX_TOKENS` bounds its tokens per rank and call (default 16384).

## vLLM 0.30.0

- **Apply:** `vllm/apply.sh SITE` patches the `vllm/` package of an installed vLLM 0.30.0 under `SITE`, a
  site-packages directory. `vllm/apply.sh SITE --from-wheel WHEEL` first unpacks vLLM 0.30.0's wheel (checked against
  `vllm/BASE`) into a new directory `SITE`: an overlay to put ahead of the environment's own packages on `PYTHONPATH`.
  `--check` only verifies.
- **Run:** `--all2all-backend nimble`, or `nccl_a2a` for the same dispatcher over NCCL, with
  `--enable-expert-parallel` and data parallelism across the GPUs (`--data-parallel-size N --tensor-parallel-size 1`).
  The patch turns CUDA graphs off for these backends.
- **Mode:** `VLLM_NIMBLE_DISPATCHER=fused` (the default) or `dedup`, which also needs `NIMBLE_DEDUP=1` on every rank.
- **Expert kernel:** vLLM's default FP8 MoE kernel (FlashInfer's TRT-LLM) runs untuned at expert-parallel batch
  sizes. `--moe-backend deep_gemm` brought vLLM's MoE compute to SGLang's level in our measurements.
