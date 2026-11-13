# TACC Horizon Setup Guide

This guide covers building and running the OSU HPC-AI PyTorch stack on TACC Horizon's Grace Blackwell (GB200) nodes: MVAPICH-Plus (CUDA-aware MPI) plus the OSU-Nowlab PyTorch fork, used for PyTorch DDP with the MPI backend. It also shows the NCCL launch we use as the vendor baseline.

No pre-built wheels are available for Horizon; build from source as described below.

## Table of Contents

- [Cluster Overview](#cluster-overview)
- [Tested Software Stack](#tested-software-stack)
- [Horizon Notes](#horizon-notes)
- [Modules](#modules)
- [Step 1: Build MVAPICH-Plus](#step-1-build-mvapich-plus)
- [Step 2: Build PyTorch](#step-2-build-pytorch)
- [Step 3: Runtime Environment](#step-3-runtime-environment)
- [Running PyTorch DDP](#running-pytorch-ddp)
- [Queues](#queues)
- [Additional Resources](#additional-resources)

## Cluster Overview

| Component     | Details                                                                                                                                       |
| ------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| Architecture  | ARM64 (aarch64)                                                                                                                               |
| Node          | NVIDIA GB200 NVL4                                                                                                                             |
| CPU           | 2x NVIDIA Grace, 72 cores each (144 cores per node)                                                                                           |
| CPU Memory    | 478 GiB LPDDR5X per Grace CPU                                                                                                                 |
| GPUs          | 4x NVIDIA Blackwell (GB200) per node, 185 GiB HBM3 each, NVLink between all pairs                                                             |
| Interconnect  | NVIDIA InfiniBand, XDR fabric, four 800 Gb/s connections per node; each node shows 4 InfiniBand HCAs (`mlx5_0`, `mlx5_1`, `mlx5_4`, `mlx5_5`) |
| Job Scheduler | SLURM                                                                                                                                         |
| Status        | Supported and tested (PyTorch 2.14.1)                                                                                                         |

## Tested Software Stack

These versions were used for the PyTorch 2.14.1 release runs (GPT-2 / nanoGPT DDP, 1-64 nodes, 4-256 GPUs).

| Component       | Version                                                                                                                                                                    |
| --------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| PyTorch         | [OSU-Nowlab/pytorch](https://github.com/OSU-Nowlab/pytorch) tag [`hpc_ai_v1.1`](https://github.com/OSU-Nowlab/pytorch/tree/hpc_ai_v1.1) (PyTorch 2.14.1, commit `8bca794`) |
| MVAPICH-Plus    | CUDA build, `ch4:ucx`                                                                                                                                                      |
| CUDA            | 13.2 (`module load cuda/13.2`, from NVIDIA HPC SDK 26.5), driver 595.71.05                                                                                                 |
| NCCL (baseline) | 2.29.3 (ships with NVIDIA HPC SDK 26.5)                                                                                                                                    |
| cuDNN           | 9.26.0.51 for CUDA 13 (`cudnn-linux-sbsa-9.26.0.51_cuda13-archive`, downloaded from NVIDIA)                                                                                |
| Compilers       | GCC 15.3.0 (`module load gcc/15.3.0`)                                                                                                                                      |
| UCX             | 1.21.0 (`module load ucx/1.21.0`)                                                                                                                                          |
| CMake           | 4.4.0 (`module load cmake/4.4.0`)                                                                                                                                          |
| Python          | 3.12 (uv virtual environment)                                                                                                                                              |

## Horizon Notes

- **4 GPUs per node.** Run one rank per GPU (4 ranks per node), with local rank *i* on `cuda:i`.
- **Use the nearest HCA.** Each rank should send over the InfiniBand HCA next to its GPU (`UCX_NET_DEVICES`, set per rank by the wrapper in [Step 3](#step-3-runtime-environment)).
- **Build on a compute node.** The login nodes have no `libcuda.so.1`, so PyTorch's GPU checks fail there. Compute nodes have no internet access, so install Python packages from a login node first, then build offline on the compute node.

## Modules

Use the same modules to build and to run. Pin the versions: the login default is now a newer NVIDIA HPC SDK / CUDA 13.3 / UCX 1.22.0 with Open MPI, so start from `module purge`.

```bash
module purge
module load gcc/15.3.0 cuda/13.2 ucx/1.21.0 cmake/4.4.0
```

`cuda/13.2` sets `CUDA_HOME` to the HPC SDK 26.5 install and puts its NCCL 2.29.3 (`comm_libs/13.2/nccl/lib`) on `LD_LIBRARY_PATH`. Don't load the `nvidia/26.x` compiler module for these builds: it sets `CC` / `CXX` to the NVIDIA HPC compilers.

## Step 1: Build MVAPICH-Plus

Download MVAPICH-Plus from the official site: [https://mvapich.cse.ohio-state.edu/downloads/](https://mvapich.cse.ohio-state.edu/downloads/). If a pre-built CUDA / InfiniBand package for Horizon is available, you can use it and skip to [Step 2](#step-2-build-pytorch). Otherwise, build from source on a compute node:

```bash
module purge
module load gcc/15.3.0 cuda/13.2 ucx/1.21.0

export MVP_HOME=/path/to/install/mvapich-plus-cuda13.2   # choose an install prefix

# Build in node-local /tmp, not on the shared file system.
mkdir -p /tmp/${USER}-build-mvp && cd /tmp/${USER}-build-mvp
tar xf /path/to/mvapich-plus-<version>.tar.gz && cd mvapich-plus-<version>
# (from a git checkout instead of a release tarball, run ./autogen.sh first)

./configure \
    --prefix=$MVP_HOME \
    --with-device=ch4:ucx \
    --with-ucx=$TACC_UCX_DIR \
    --enable-cuda \
    --with-cuda=$CUDA_HOME \
    NVCC=$CUDA_HOME/bin/nvcc \
    NVCC_FLAGS="-gencode arch=compute_100,code=sm_100" \
    --with-ch4-shmmods=hybrid \
    --enable-fast=opt \
    --enable-g=none \
    --enable-error-checking=no \
    --disable-omb \
    CC=gcc CXX=g++ FC=gfortran F77=gfortran \
    FFLAGS=-fallow-argument-mismatch \
    CFLAGS="-Wall -pipe -std=gnu17" \
    CXXFLAGS="-std=gnu++17" \
    LDFLAGS="-lstdc++"
make -j 64 install

$MVP_HOME/bin/mpichversion     # check: ch4:ucx device, CUDA enabled
```

`NVCC_FLAGS` builds MVAPICH-Plus's GPU kernels for Blackwell (`sm_100`) only.

## Step 2: Build PyTorch

### 2.1 Get the source

```bash
git clone -b hpc_ai_v1.1 https://github.com/OSU-Nowlab/pytorch.git
cd pytorch
git submodule sync
git submodule update --init --recursive
```

If you cloned `osu-hpc-ai` with `--recursive`, the same source is at `frameworks/pytorch`.

### 2.2 cuDNN

Horizon has no cuDNN module. Download the cuDNN 9.26.0.51 archive for CUDA 13 on Linux SBSA (aarch64) from [NVIDIA](https://developer.nvidia.com/cudnn-downloads) and unpack it:

```bash
tar xf cudnn-linux-sbsa-9.26.0.51_cuda13-archive.tar.xz -C /path/to
export CUDNN_HOME=/path/to/cudnn-linux-sbsa-9.26.0.51_cuda13-archive
```

### 2.3 Python environment

On a **login node** (compute nodes have no internet). We use [uv](https://docs.astral.sh/uv/); any Python 3.12 virtual environment works.

```bash
uv venv --python 3.12 /path/to/venv/torch2.14.1-mvp
source /path/to/venv/torch2.14.1-mvp/bin/activate

cd pytorch
uv pip install -r requirements.txt      # from the PyTorch source tree
# also install your training script's packages now (for nanoGPT: numpy, tiktoken, ...)
```

### 2.4 Build environment

On a **compute node**:

```bash
module purge
module load gcc/15.3.0 cuda/13.2 ucx/1.21.0 cmake/4.4.0
source /path/to/venv/torch2.14.1-mvp/bin/activate

export MVP_HOME=/path/to/install/mvapich-plus-cuda13.2
export MPI_HOME=$MVP_HOME
export PATH=$MVP_HOME/bin:$PATH
export LD_LIBRARY_PATH=$MVP_HOME/lib:$LD_LIBRARY_PATH
export CPATH=$MVP_HOME/include:${CPATH:-}
export MPI_C_COMPILER=$MVP_HOME/bin/mpicc
export MPI_CXX_COMPILER=$MVP_HOME/bin/mpicxx

export NCCL_HOME=$(dirname $CUDA_HOME)/../comm_libs/13.2/nccl   # NCCL 2.29.3 from HPC SDK 26.5
export CUDNN_HOME=/path/to/cudnn-linux-sbsa-9.26.0.51_cuda13-archive

# Skip nvcc's host-compiler version check (GCC 15.3); our builds set this.
export NVCC_APPEND_FLAGS=-allow-unsupported-compiler
```

### 2.5 Build and install

```bash
cd pytorch
MAX_JOBS=96 \
USE_CUDA=1 \
USE_MPI=1 \
USE_CUDA_MPI=1 \
USE_DISTRIBUTED=1 \
USE_SYSTEM_NCCL=1 \
NCCL_INCLUDE_DIR=$NCCL_HOME/include \
NCCL_LIB_DIR=$NCCL_HOME/lib \
CUDNN_ROOT=$CUDNN_HOME \
CUDNN_INCLUDE_DIR=$CUDNN_HOME/include \
CUDNN_LIBRARY=$CUDNN_HOME/lib \
USE_CUFILE=0 \
TORCH_CUDA_ARCH_LIST=10.0 \
CMAKE_POLICY_VERSION_MINIMUM=3.5 \
    uv pip install --no-build-isolation -v .
```

**Build flags:**

- `TORCH_CUDA_ARCH_LIST=10.0`: build for Blackwell (GB200) only, which also shortens the build.
- `USE_MPI=1`: build the MPI backend against `MPI_HOME`.
- `USE_CUDA_MPI=1`: pass GPU buffers directly to MPI (CUDA-aware MPI).
- `USE_SYSTEM_NCCL=1` + `NCCL_*`: use the HPC SDK's NCCL for the `nccl` backend (the baseline) instead of building PyTorch's bundled copy.
- `USE_CUFILE=0`: skip GPUDirect Storage.
- `CMAKE_POLICY_VERSION_MINIMUM=3.5`: lets some third-party projects configure with CMake 4.x.

### 2.6 Verify

On a compute node:

```bash
cd /tmp    # not inside the source tree
python -c "
import os, torch, torch.distributed as dist
print('torch', torch.__version__, 'git', torch.version.git_version[:7])
print('cuda', torch.version.cuda, 'gpus', torch.cuda.device_count())
print('mpi', dist.is_mpi_available(), 'nccl', dist.is_nccl_available(), torch.cuda.nccl.version())
"
ldd $(python -c "import torch, os; print(os.path.dirname(torch.__file__))")/lib/libtorch_cpu.so | grep libmpi
```

Expected: a `2.14.1` version with git `8bca794`, CUDA `13.2`, 4 GPUs, `mpi True`, `nccl True`, NCCL `(2, 29, 3)`, and `libmpi` resolving to `$MVP_HOME/lib`.

## Step 3: Runtime Environment

Save this as `~/hpc-ai-horizon-env.sh` and source it in your job scripts:

```bash
#!/bin/bash
module purge
module load gcc/15.3.0 cuda/13.2 ucx/1.21.0

export MVP_HOME=/path/to/install/mvapich-plus-cuda13.2
export PATH=$MVP_HOME/bin:$PATH
export LD_LIBRARY_PATH=$MVP_HOME/lib:/path/to/cudnn-linux-sbsa-9.26.0.51_cuda13-archive/lib:$LD_LIBRARY_PATH

source /path/to/venv/torch2.14.1-mvp/bin/activate
```

`$MVP_HOME/lib` must come first on `LD_LIBRARY_PATH`, so that PyTorch loads this MVAPICH-Plus build's `libmpi`.

Save this per-rank wrapper as `~/hpc-ai-horizon-rank.sh` and make it executable (`chmod +x`). `mpirun` runs it on every rank, in front of the training command:

```bash
#!/bin/bash
# torch.distributed rank variables, from MVAPICH-Plus's Hydra launcher
export RANK=$PMI_RANK WORLD_SIZE=$PMI_SIZE
export LOCAL_RANK=$MPI_LOCALRANKID LOCAL_WORLD_SIZE=$MPI_LOCALNRANKS
# The InfiniBand HCA next to this rank's GPU (GPU index = local rank)
NICS=(mlx5_0:1 mlx5_1:1 mlx5_4:1 mlx5_5:1)
export UCX_NET_DEVICES=${NICS[$LOCAL_RANK]}
exec "$@"
```

The training script must put each rank on GPU `LOCAL_RANK`, for example `torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))` (nanoGPT does this). Every rank sees all 4 GPUs; otherwise all 4 ranks run on GPU 0.

## Running PyTorch DDP

### MVAPICH-Plus (MPI backend)

The environment below is the one we use on Horizon.

```bash
#!/bin/bash
#SBATCH -J ddp-mvp
#SBATCH -A <project_id>
#SBATCH -p <partition>
#SBATCH -N 2
#SBATCH -t 01:00:00

source ~/hpc-ai-horizon-env.sh

# GPU-aware MPI
export MPIR_CVAR_ENABLE_GPU=1
export MPICH_GPU_SUPPORT_ENABLED=1
export MVP_ENABLE_GPU=1
export MPIR_CVAR_REDUCE_KERNEL_MAX_NUM_CTAS=32
export MPIR_CVAR_REDUCE_KERNEL_MAX_NUM_THREADS=544
# UCX / InfiniBand
export UCX_TLS=^dc                       # don't use the DC transport
export UCX_IB_PCI_RELAXED_ORDERING=on
export MPIR_CVAR_PMI_VERSION=1

# Multi-node allreduce: two-level composition (GPU IPC within a node, network between nodes)
if [ "$SLURM_JOB_NUM_NODES" -gt 1 ]; then
    export MPIR_CVAR_ALLREDUCE_COMPOSITION=8
fi

HOSTS=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | paste -sd,)
NP=$((SLURM_JOB_NUM_NODES * 4))

mpirun -hosts $HOSTS -n $NP -ppn 4 -bind-to core:36 \
    ~/hpc-ai-horizon-rank.sh python -u train.py --backend=mpi
```

On 2 or more nodes, MVAPICH-Plus picks the algorithm for the inter-node step of the allreduce by message size. You can try a specific one for your model's gradient bucket sizes:

```bash
export MPIR_CVAR_ALLREDUCE_INTRA_ALGORITHM=osu_gpu_rsa      # or osu_gpu_direct, osu_gpu_ring
export MPIR_CVAR_INIT_SET_DEVICE_BY_LOCAL_RANK=1            # required with PyTorch when forcing an algorithm
```

### NCCL baseline (nccl backend)

Same launcher, binding and rank wrapper, so both backends run with the same process layout:

```bash
#!/bin/bash
#SBATCH -J ddp-nccl
#SBATCH -A <project_id>
#SBATCH -p <partition>
#SBATCH -N 2
#SBATCH -t 01:00:00

source ~/hpc-ai-horizon-env.sh

export NCCL_IB_HCA==mlx5_0:1,mlx5_1:1,mlx5_4:1,mlx5_5:1   # the 4 InfiniBand HCAs only (leading '=' = exact names)
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500

HOSTS=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | paste -sd,)
NP=$((SLURM_JOB_NUM_NODES * 4))

mpirun -hosts $HOSTS -n $NP -ppn 4 -bind-to core:36 \
    ~/hpc-ai-horizon-rank.sh python -u train.py --backend=nccl
```

## Queues

The Horizon User Guide (a draft during early access) lists `gpu-dev` (quick tests, 2 h), `gpu`, `gpu-large` and `gpu-shared` (1-3 GPUs of a node) for the GB200 nodes. During early access our runs used the `debug` partition, the only GPU partition open at the time. Check `sinfo` and the [Horizon User Guide](https://docs.tacc.utexas.edu/hpc/horizon/) for the current partitions and limits.

```bash
# Interactive, 1 node (4 GPUs)
salloc -A <project_id> -p gpu-dev -N 1 -t 0:30:00

# Batch
sbatch -A <project_id> -p gpu -N 8 -t 1:00:00 job.sbatch
```

## Additional Resources

- **PyTorch DDP with MPI**: [../user-guides/pytorch-ddp.md](../user-guides/pytorch-ddp.md)
- **Troubleshooting Guide**: [../troubleshooting.md](../troubleshooting.md)
- **TACC Horizon User Guide**: [https://docs.tacc.utexas.edu/hpc/horizon/](https://docs.tacc.utexas.edu/hpc/horizon/)
- **Official HPC-AI Userguide**: [https://hpc-ai.engineering.osu.edu/userguide/](https://hpc-ai.engineering.osu.edu/userguide/)
- **MVAPICH-Plus Downloads**: [https://mvapich.cse.ohio-state.edu/downloads/](https://mvapich.cse.ohio-state.edu/downloads/)
