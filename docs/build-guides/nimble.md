# Installing NIMBLE

This guide explains how to install NIMBLE and how to enable it as the MoE all-to-all backend of SGLang and vLLM.

## Overview

[NIMBLE](https://github.com/OSU-Nowlab/NIMBLE) is a GPU AllToAllV for irregular traffic, with an MoE dispatcher for SGLang and vLLM. It plans each AllToAllV from the byte matrix every rank passes: which bytes go directly and which are relayed through another GPU of the node onto a less loaded NIC. One kernel per call on each GPU moves the data, and one CPU thread per NIC posts the RDMA writes.

NIMBLE ships as a pre-built wheel: there is nothing to compile, and it does not need MVAPICH-Plus.

## Prerequisites

- **System**: Linux aarch64 with glibc 2.28 or later
- **GPUs**: 4 per node, with the NVIDIA driver
- **Network**: RDMA NICs with rdma-core
- **Python**: 3.10-3.15

## Installation

### Option 1: From the release (Recommended)

```bash
pip install https://github.com/OSU-Nowlab/NIMBLE/releases/download/v1.0.0/nimble_gpu-1.0.0-py3-none-manylinux_2_28_aarch64.whl
```

### Option 2: From the submodule

```bash
cd osu-hpc-ai
git submodule update --init frameworks/nimble
pip install frameworks/nimble/wheel/nimble_gpu-1.0.0-py3-none-manylinux_2_28_aarch64.whl
```

## Verification

```bash
python -c "import nimble; print(nimble.__version__)"    # 1.0.0

# One node, 4 GPUs
torchrun --nproc-per-node 4 examples/nimble/python/01_all_to_all_single.py
# 01_all_to_all_single: OK on 4 ranks (...)
```

On several nodes: `sbatch -N NODES examples/nimble/python/launch_slurm.sh FILE`. All the examples: [examples/nimble](../../examples/nimble/README.md).

## SGLang and vLLM

Install the wheel in the framework's environment first, then apply NIMBLE's patch to the framework, from `frameworks/nimble/patches` (also in the release as `nimble_frameworks-1.0.0.tar.gz`).

### SGLang (commit `fbf8f1d`)

```bash
frameworks/nimble/patches/sglang/apply.sh DEST    # clones SGLang into the new directory DEST and applies the patches
pip install -e DEST/python
```

Run with `--moe-a2a-backend nimble` and `--disable-cuda-graph`, with expert parallelism and DP attention (`--tp-size N --ep-size N --dp-size N --enable-dp-attention`).

### vLLM 0.30.0

```bash
frameworks/nimble/patches/vllm/apply.sh SITE      # SITE: the site-packages directory of an installed vLLM 0.30.0
```

Run with `--all2all-backend nimble` and `--enable-expert-parallel`, with data parallelism across the GPUs (`--data-parallel-size N --tensor-parallel-size 1`).

In both, `nccl_a2a` runs the same dispatcher over NCCL, and `SGLANG_NIMBLE_DISPATCHER=dedup` / `VLLM_NIMBLE_DISPATCHER=dedup` with `NIMBLE_DEDUP=1` on every rank selects the dedup mode. Complete server scripts: [examples/serving](https://github.com/OSU-Nowlab/NIMBLE/tree/main/examples/serving).

## More

- [NIMBLE README](https://github.com/OSU-Nowlab/NIMBLE#readme): the APIs, every example and the performance results
- [The patches' README](https://github.com/OSU-Nowlab/NIMBLE/blob/main/patches/README.md): every option of the SGLang and vLLM integration
