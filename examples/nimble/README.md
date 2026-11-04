# NIMBLE Examples

The Python API in [python/](python/), and DeepSeek-V3 served by SGLang and vLLM with NIMBLE as the MoE all-to-all backend in [serving/](serving/). Run them in this order: the Python examples on one node, then on two nodes, then vLLM, then SGLang. The commands are for Slurm: add `-p PARTITION -A ACCOUNT` where your site needs them. If your nodes have RDMA ports besides the compute NICs, export `NCCL_IB_HCA` with the compute NICs first (for example `mlx5_0,mlx5_1,mlx5_4,mlx5_5`).

## 0. Set up, once

```bash
git clone https://github.com/OSU-Nowlab/NIMBLE.git
NIMBLE=$PWD/NIMBLE                       # in osu-hpc-ai: NIMBLE=$PWD/frameworks/nimble
python3 -m venv $HOME/nimble-venv
$HOME/nimble-venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu130
$HOME/nimble-venv/bin/pip install https://github.com/OSU-Nowlab/NIMBLE/releases/download/v1.0.0/nimble_gpu-1.0.0-py3-none-manylinux_2_28_aarch64.whl
$HOME/nimble-venv/bin/python -c "import nimble; print(nimble.__version__)"     # 1.0.0
```

## 1. Python examples, one node

```bash
srun -N 1 --pty bash                     # a shell on a compute node with 4 GPUs
cd $NIMBLE/examples/python
R="$HOME/nimble-venv/bin/python -m torch.distributed.run --nproc-per-node 4"
$R 01_all_to_all_single.py
$R 02_alltoallv_counts.py
$R 03_streams.py
$R 04_subgroups.py
$R 05_no_torch_distributed.py
$R 06_errors.py
$R 07_moe_dispatcher.py                       # FP8, fused, over NIMBLE
$R 07_moe_dispatcher.py --dtype bf16
$R 07_moe_dispatcher.py --transport torch     # the same dispatcher over NCCL
exit
```

Each prints one line that ends in `OK`.

| Example | What it shows |
|---|---|
| [01_all_to_all_single.py](python/01_all_to_all_single.py) | NIMBLE as a drop-in for `torch.distributed.all_to_all_single` |
| [02_alltoallv_counts.py](python/02_alltoallv_counts.py) | variable sizes (AllToAllV) with the count table every rank already holds |
| [03_streams.py](python/03_streams.py) | NIMBLE on a side stream, overlapped with compute; one communicator per stream |
| [04_subgroups.py](python/04_subgroups.py) | communicators over `torch.distributed` subgroups |
| [05_no_torch_distributed.py](python/05_no_torch_distributed.py) | NIMBLE without `torch.distributed`, on raw device pointers |
| [06_errors.py](python/06_errors.py) | errors and shutdown: what raises where |
| [07_moe_dispatcher.py](python/07_moe_dispatcher.py) | expert-parallel MoE with `nimble.moe.Dispatcher` |

## 2. Python examples, two nodes

```bash
cd $NIMBLE/examples/python
export PYTHON=$HOME/nimble-venv/bin/python
S="sbatch -N 2 launch_slurm.sh"
$S 01_all_to_all_single.py
$S 02_alltoallv_counts.py
$S 03_streams.py
$S 04_subgroups.py
$S 05_no_torch_distributed.py
$S 06_errors.py
$S 07_moe_dispatcher.py
$S 07_moe_dispatcher.py --dtype bf16
$S 07_moe_dispatcher.py --transport torch
NIMBLE_DEDUP=1 $S 07_moe_dispatcher.py --mode dedup     # dedup needs two or more nodes
squeue -u $USER                                         # wait until they are done
grep -h -E ": (OK|MISMATCH|UNEXPECTED)" slurm-*.out
```

## 3. vLLM

```bash
python3 -m venv $HOME/vllm-venv
$HOME/vllm-venv/bin/pip install vllm==0.30.0 https://github.com/OSU-Nowlab/NIMBLE/releases/download/v1.0.0/nimble_gpu-1.0.0-py3-none-manylinux_2_28_aarch64.whl
$HOME/vllm-venv/bin/pip download vllm==0.30.0 --no-deps -d $HOME/wheels
$NIMBLE/patches/vllm/apply.sh $HOME/vllm-nimble-site --from-wheel $HOME/wheels/vllm-0.30.0-cp38-abi3-manylinux_2_28_aarch64.whl
cd $NIMBLE/examples/serving
export MODEL=/path/to/DeepSeek-V3 VLLM_VENV=$HOME/vllm-venv VLLM_SITE=$HOME/vllm-nimble-site   # every setting: site_env.sh
S="sbatch -N 2"
$S vllm_server.sh                             # fused
MODE=dedup $S vllm_server.sh                  # dedup
squeue -u $USER
grep -h "client: OK\|client: FAILED" slurm-*.out
```

## 4. SGLang

```bash
$NIMBLE/patches/sglang/apply.sh $HOME/sglang-nimble
python3 -m venv $HOME/sglang-venv
$HOME/sglang-venv/bin/pip install -e $HOME/sglang-nimble/python https://github.com/OSU-Nowlab/NIMBLE/releases/download/v1.0.0/nimble_gpu-1.0.0-py3-none-manylinux_2_28_aarch64.whl
cd $NIMBLE/examples/serving
export MODEL=/path/to/DeepSeek-V3 SGLANG_VENV=$HOME/sglang-venv SGLANG_DIR=$HOME/sglang-nimble   # every setting: site_env.sh
S="sbatch -N 2"
$S sglang_server.sh                           # fused
MODE=dedup $S sglang_server.sh                # dedup
$S sglang_offline.sh                          # sglang.Engine, without an HTTP server
squeue -u $USER
grep -h "client: OK\|client: FAILED\|sglang_offline: OK\|sglang_offline: FAILED" slurm-*.out
```

A serving run passes when it prints `client: OK; N/M correct ...` (or `sglang_offline: OK; ...`). Each job can take up to 45 minutes on 2 nodes; each node's server log is under `logs/`.
