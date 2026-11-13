# OLCF Frontier Setup Guide

This guide covers building and running the OSU HPC-AI PyTorch stack on OLCF Frontier: MVAPICH-Plus (ROCm-aware MPI) plus the OSU-Nowlab PyTorch fork, used for PyTorch DDP with the MPI backend. It also shows the RCCL launch we use as the vendor baseline.

No pre-built wheels are available for Frontier; build from source as described below.

## Table of Contents

- [Cluster Overview](#cluster-overview)
- [Tested Software Stack](#tested-software-stack)
- [Frontier Notes](#frontier-notes)
- [Modules](#modules)
- [Step 1: Build MVAPICH-Plus](#step-1-build-mvapich-plus)
- [Step 2: Build PyTorch](#step-2-build-pytorch)
- [Step 3: Runtime Environment](#step-3-runtime-environment)
- [Running PyTorch DDP](#running-pytorch-ddp)
- [Queues](#queues)
- [Additional Resources](#additional-resources)

## Cluster Overview

| Component | Details |
|-----------|---------|
| Architecture | x86_64 |
| CPU | 1x AMD EPYC 7A53 "Trento", 64 cores (56 usable; 8 are reserved for the system) |
| CPU Memory | 512 GB DDR4 |
| GPUs | 4x AMD MI250X per node, 2 GCDs each = 8 GPUs per node, 64 GB HBM2e per GCD |
| Interconnect | HPE Slingshot-11, 4 NICs per node (200 Gb/s each) |
| Job Scheduler | SLURM |
| Status | Supported and tested (PyTorch 2.14.1) |

## Tested Software Stack

These versions were used for the PyTorch 2.14.1 release runs (GPT-2 / nanoGPT DDP, 1-64 nodes, 8-512 GCDs).

| Component | Version |
|-----------|---------|
| PyTorch | [OSU-Nowlab/pytorch](https://github.com/OSU-Nowlab/pytorch) tag [`hpc_ai_v1.1`](https://github.com/OSU-Nowlab/pytorch/tree/hpc_ai_v1.1) (PyTorch 2.14.1, commit `8bca794`) |
| MVAPICH-Plus | 5.1 (pre-release), ROCm build, `ch4:ofi` |
| ROCm | 7.13.0 (`module load rocm/7.13.0`) |
| RCCL (baseline) | 2.28.3 (ships with ROCm 7.13.0) + `rccl-net-plugin/1.0` (aws-ofi-nccl 1.19.2) |
| Compilers | `PrgEnv-gnu/8.6.0`, `gcc-native/13.2` (host), `amdclang` from ROCm (MVAPICH-Plus) |
| libfabric | 2.3.1 (`/opt/cray/libfabric/2.3.1`) |
| Python | 3.12 (uv virtual environment) |
| mpi4py | 4.1.2, built from source against MVAPICH-Plus |

## Frontier Notes

- **8 GPUs per node.** Each MI250X has 2 Graphics Compute Dies (GCDs), and ROCm exposes each GCD as its own GPU, so `torch.cuda.device_count()` returns 8 on a full node. Run one rank per GCD (8 ranks per node).
- **Usable cores.** Frontier reserves one core per L3 region for the system, so 56 of the 64 cores are usable. That is 7 cores per rank at 8 ranks per node (`-c 7`, `OMP_NUM_THREADS=7`).

## Modules

Use the same modules to build and to run. Pin the versions: the Frontier default CPE changed on 2026-11-10, and an unpinned `module load PrgEnv-gnu` picks up the new default.

```bash
module reset
module load PrgEnv-gnu/8.6.0 gcc-native/13.2
module load rocm/7.13.0
module load craype-accel-amd-gfx90a
module unload cray-mpich        # MVAPICH-Plus replaces Cray MPICH
```

## Step 1: Build MVAPICH-Plus

Download MVAPICH-Plus from the official site: [https://mvapich.cse.ohio-state.edu/downloads/](https://mvapich.cse.ohio-state.edu/downloads/). If a pre-built ROCm / Slingshot package for Frontier is available, you can use it and skip to [Step 2](#step-2-build-pytorch). Otherwise, build from source:

```bash
module reset
module load cpe/24.11 PrgEnv-gnu/8.6.0 gcc-native/13.2
module load rocm/7.13.0 craype-accel-amd-gfx90a
module unload cray-mpich

# Keep compiler temp files on node-local disk: with TMPDIR on Lustre,
# amdgcn-link can abort with "malloc(): corrupted top size".
export TMPDIR=/tmp/${USER}-build-mvp
mkdir -p "$TMPDIR"

export MVP_HOME=/path/to/install/mvapich-plus-rocm7.13.0   # choose an install prefix

cd mvapich-plus-<version>
./configure \
    --prefix=$MVP_HOME \
    --with-device=ch4:ofi \
    --with-libfabric=/opt/cray/libfabric/2.3.1 \
    --enable-rocm \
    --with-hip=/opt/rocm-7.13.0 \
    --enable-rcclomb \
    --with-rccl=/opt/rocm-7.13.0 \
    CC=$ROCM_PATH/bin/amdclang \
    CXX=$ROCM_PATH/bin/amdclang++ \
    LDFLAGS="-lstdc++" \
    CFLAGS="-pipe -Wno-implicit-function-declaration -Wno-incompatible-pointer-types -Wno-int-conversion"
make -j 32
make install

$MVP_HOME/bin/mpichversion     # check: ch4:ofi device, ROCm enabled
```

## Step 2: Build PyTorch

### 2.1 Get the source

```bash
git clone -b hpc_ai_v1.1 https://github.com/OSU-Nowlab/pytorch.git
cd pytorch
git submodule sync
git submodule update --init --recursive
```

If you cloned `osu-hpc-ai` with `--recursive`, the same source is at `frameworks/pytorch`.

### 2.2 Build environment

```bash
module reset
module load PrgEnv-gnu/8.6.0 gcc-native/13.2
module load rocm/7.13.0 craype-accel-amd-gfx90a
module unload cray-mpich

export MVP_HOME=/path/to/install/mvapich-plus-rocm7.13.0
export MPI_HOME=$MVP_HOME
export PATH=$MVP_HOME/bin:$PATH
export LD_LIBRARY_PATH=$MVP_HOME/lib:$LD_LIBRARY_PATH
export CPATH=$MVP_HOME/include:${CPATH:-}
export MPI_C_COMPILER=$MVP_HOME/bin/mpicc
export MPI_CXX_COMPILER=$MVP_HOME/bin/mpicxx

export CMAKE_PREFIX_PATH=/opt/rocm-7.13.0${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}

# Pin the host compilers so nested CMake projects do not fall back to /usr/bin/gcc.
export CC=/opt/cray/pe/gcc-native/13/bin/gcc
export CXX=/opt/cray/pe/gcc-native/13/bin/g++
export CMAKE_C_COMPILER=$CC CMAKE_CXX_COMPILER=$CXX
export CMAKE_ARGS="-DCMAKE_POLICY_VERSION_MINIMUM=3.5"
```

### 2.3 Python environment and mpi4py

We use [uv](https://docs.astral.sh/uv/); any Python 3.12 virtual environment works.

```bash
uv venv --python 3.12 /path/to/venv/torch2.14.1-mvp
source /path/to/venv/torch2.14.1-mvp/bin/activate

uv pip install -r requirements.txt      # from the PyTorch source tree

# mpi4py must be built against MVAPICH-Plus, not taken from a binary wheel.
MPICC=$MVP_HOME/bin/mpicc uv pip install --no-cache --no-binary mpi4py mpi4py==4.1.2
```

### 2.4 Build and install

```bash
cd pytorch
python tools/amd_build/build_amd.py      # hipify the CUDA sources for ROCm

export _GLIBCXX_USE_CXX11_ABI=1
MAX_JOBS=32 \
USE_ROCM=1 \
USE_MPI=1 \
USE_CUDA_MPI=1 \
USE_DISTRIBUTED=1 \
USE_XNNPACK=0 \
BUILD_TEST=0 \
BUILD_MOBILE_BENCHMARK=0 \
BUILD_MOBILE_TEST=0 \
PYTORCH_ROCM_ARCH=gfx90a \
    uv pip install --no-build-isolation -v .
```

**Build flags:**

- `USE_ROCM=1`, `PYTORCH_ROCM_ARCH=gfx90a`: build for MI250X.
- `USE_MPI=1`: build the MPI backend against `MPI_HOME`.
- `USE_CUDA_MPI=1`: pass GPU buffers directly to MPI (GPU-aware MPI).
- `USE_XNNPACK=0`, `BUILD_TEST=0`: skip parts we do not need, and shorten the build.

### 2.5 Verify

```bash
python -c "
import torch, torch.distributed as dist
print('torch', torch.__version__, 'git', torch.version.git_version)
print('hip', torch.version.hip)
print('mpi', dist.is_mpi_available(), 'nccl', dist.is_nccl_available())
print('rccl', torch.cuda.nccl.version())
"
```

Expected: a `2.14.1` version, a `7.13` HIP version, `mpi True`, `nccl True`, and RCCL `(2, 28, 3)`. Run the GPU check on a compute node: `torch.cuda.device_count()` should return 8.

## Step 3: Runtime Environment

Save this as `~/hpc-ai-frontier-env.sh` and source it in your job scripts:

```bash
#!/bin/bash
module reset
module load PrgEnv-gnu/8.6.0 gcc-native/13.2
module load rocm/7.13.0 craype-accel-amd-gfx90a
module unload cray-mpich

export MVP_HOME=/path/to/install/mvapich-plus-rocm7.13.0
export PATH=$MVP_HOME/bin:$PATH
export LD_LIBRARY_PATH=$MVP_HOME/lib:$LD_LIBRARY_PATH

source /path/to/venv/torch2.14.1-mvp/bin/activate

export OMP_NUM_THREADS=7
# Keep the MIOpen kernel cache on node-local disk, one per job.
export MIOPEN_USER_DB_PATH=/tmp/${USER}-miopen-cache-${SLURM_JOB_ID:-none}
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH
```

## Running PyTorch DDP

### MVAPICH-Plus (MPI backend)

The environment below is the tuned set used in the release runs. See [PyTorch DDP with MPI: Frontier tuning](../user-guides/pytorch-ddp.md#amd-mi250x-slingshot-olcf-frontier) for what each variable does.

```bash
#!/bin/bash
#SBATCH -J ddp-mvp
#SBATCH -A <project_id>
#SBATCH -N 2
#SBATCH -t 01:00:00
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --network=disable_rdzv_get

source ~/hpc-ai-frontier-env.sh

# MVAPICH-Plus collectives (sizes in bytes)
export MPIR_CVAR_ENABLE_GPU=1
export MPICH_GPU_SUPPORT_ENABLED=1
export MPIR_CVAR_CH4_OFI_ENABLE_HMEM=1
export MPIR_CVAR_CH4_OFI_ENABLE_MR_HMEM=1
export MPIR_CVAR_ALLREDUCE_COMPOSITION=8
export MPIR_CVAR_ALLREDUCE_GPU_RING_THRESHOLD=67108864
export MPIR_CVAR_ALLREDUCE_PIPELINE_2LVL_GPU_THRESHOLD=8388608
export MPIR_CVAR_ALLREDUCE_PIPELINE_2LVL_GPU_NUM_CHUNKS=1
export MPIR_CVAR_ALLREDUCE_INTRA_ALGORITHM=osu_gpu_rsa
export MPIR_CVAR_FORCE_GPU_QUERY_ATTR=1
export MPIR_CVAR_ALLGATHER_COMPOSITION=1
export MVP_ENABLE_GPU_PTRCACHE=0
export HSA_XNACK=1

# Slingshot / libfabric (same values as the rccl-net-plugin/1.0 module)
export HSA_FORCE_FINE_GRAIN_PCIE=1
export FI_MR_CACHE_MONITOR=kdreg2
export FI_CXI_DISABLE_HOST_REGISTER=1
export FI_CXI_DEFAULT_CQ_SIZE=131072
export FI_CXI_DEFAULT_TX_SIZE=2048
export FI_CXI_RDZV_PROTO=alt_read
export FI_CXI_RDZV_EAGER_SIZE=0
export FI_CXI_RDZV_THRESHOLD=0
export FI_CXI_RDZV_GET_MIN=0
export FI_CXI_RX_MATCH_MODE=hybrid

NP=$((SLURM_JOB_NUM_NODES * 8))
# Inside a Slurm job, mpirun starts its per-node processes with srun and takes
# the node list from Slurm. These srun options give each node all 56 usable
# cores and all 8 GPUs.
export HYDRA_LAUNCHER_EXTRA_ARGS="-c 56 --gpus-per-node=8 --gpu-bind=none"

mpirun -np $NP -ppn 8 python -u train.py --backend=mpi
```

`--network=disable_rdzv_get` is required because `FI_CXI_RDZV_PROTO=alt_read` is set.

Every rank sees all 8 GPUs (`--gpu-bind=none`), so the training script must select its own GPU from the node-local rank, for example `torch.cuda.set_device(int(os.environ["MPI_LOCALRANKID"]))`. Otherwise all 8 ranks run on GPU 0.

### RCCL baseline (nccl backend)

```bash
#!/bin/bash
#SBATCH -J ddp-rccl
#SBATCH -A <project_id>
#SBATCH -N 2
#SBATCH -t 01:00:00
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --network=disable_rdzv_get

source ~/hpc-ai-frontier-env.sh
module load rccl-net-plugin/1.0       # aws-ofi-nccl plus its FI_CXI_* / NCCL_* settings

if [ "$SLURM_JOB_NUM_NODES" -eq 1 ]; then export NCCL_NET=Socket; else export NCCL_NET=OFI; fi
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500

NP=$((SLURM_JOB_NUM_NODES * 8))
srun -N $SLURM_JOB_NUM_NODES -n $NP -c 7 --gpus-per-task=1 --gpu-bind=closest --mpi=pmi2 \
    python -u train.py --backend=nccl
```

## Queues

| Partition / QOS | Use |
|-----------------|-----|
| `-p batch` | Production runs. The walltime limit depends on the job's node count. |
| `-p batch -q debug` | Quick tests. 2 h max, 1 job per user. |
| `-p extended` | Smaller jobs (up to 64 nodes) with longer walltime. |

`debug` is a QOS on the `batch` partition, not a partition.

```bash
# Interactive, 1 node (8 GCDs), debug QOS
salloc -A <project_id> -p batch -q debug -t 0:30:00 -N 1 --gpus-per-node=8 --network=disable_rdzv_get

# Batch
sbatch -A <project_id> -p batch -N 8 -t 1:00:00 job.sbatch
```

Check the [OLCF Frontier User Guide](https://docs.olcf.ornl.gov/systems/frontier_user_guide.html#scheduling-policy) for the current limits.

## Additional Resources

- **PyTorch DDP with MPI**: [../user-guides/pytorch-ddp.md](../user-guides/pytorch-ddp.md)
- **Troubleshooting Guide**: [../troubleshooting.md](../troubleshooting.md)
- **OLCF Frontier User Guide**: [https://docs.olcf.ornl.gov/systems/frontier_user_guide.html](https://docs.olcf.ornl.gov/systems/frontier_user_guide.html)
- **ROCm Documentation**: [https://rocm.docs.amd.com/](https://rocm.docs.amd.com/)
- **Official HPC-AI Userguide**: [https://hpc-ai.engineering.osu.edu/userguide/](https://hpc-ai.engineering.osu.edu/userguide/)
- **MVAPICH-Plus Downloads**: [https://mvapich.cse.ohio-state.edu/downloads/](https://mvapich.cse.ohio-state.edu/downloads/)
