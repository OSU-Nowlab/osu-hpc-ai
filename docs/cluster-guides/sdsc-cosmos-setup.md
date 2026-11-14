# SDSC Cosmos Setup Guide

This guide covers building and running the OSU HPC-AI PyTorch stack on SDSC Cosmos (MI300 Nodes): MVAPICH-Plus (CUDA-aware MPI) plus the OSU-Nowlab PyTorch fork, used for PyTorch DDP with the MPI backend. It also shows the RCCL launch we use as the vendor baseline.

No pre-built wheels are available for Horizon; build from source as described below.

## Table of Contents

- [Cluster Overview](#cluster-overview)
- [Tested Software Stack](#tested-software-stack)
- [Integrated GPU Architecture](#Integrated-GPU-Architecture)
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
| Architecture | x86_64 (AMD EPYC) |
| CPUs | AMD EPYC (96 cores @ 3.7 GHz) |
| Memory | 512 GB HBM3 (unified CPU+GPU) |
| GPUs | Integrated AMD CDNA3 APU (MI310X, 128 GB HBM3 per APU) |
| Interconnect | HPE Cray Slingshot-11 (200 Gb/s) |
| Status | Validated |

## Tested Software Stack

These versions were used for the PyTorch 2.14.1 release runs (GPT-2 / nanoGPT DDP, 1-16 nodes, 4-64 APUs).

| Component | Version |
|-----------|---------|
| PyTorch | [OSU-Nowlab/pytorch](https://github.com/OSU-Nowlab/pytorch) tag [`hpc_ai_v1.1`](https://github.com/OSU-Nowlab/pytorch/tree/hpc_ai_v1.1) (PyTorch 2.14.1, commit `8bca794`) |
| MVAPICH-Plus | 5.1 (pre-release), ROCm build, `ch4:ofi` |
| ROCm | 7.0.0 (`rocm/7.0.0`) |
| RCCL (baseline) | 2.26.6 (ships with ROCm 7.0.0) + `nccl-net-plugin` (aws-ofi-nccl) |
| Compilers | `PrgEnv-gnu/8.7.0`, `gcc-native/13.2` (host), `amdclang` from ROCm (MVAPICH-Plus) |
| libfabric | 2.3.1 (`/opt/cray/libfabric/2.3.1`) |
| Python | 3.12 (conda virtual environment) |

## Integrated GPU Architecture

Cosmos uses AMD CDNA3 APUs where the CPU and GPU share HBM3 memory. This is different from discrete GPU setups:

- **No PCIe**: CPU and GPU access the same physical HBM3 memory without a PCIe bus
- **Unified addressing**: GPU tensors can be directly accessed from CPU without explicit copies
- **No separate GPU memory**: All 128 GB is shared between compute and GPU workloads

This means `cudaMalloc` allocates from the shared HBM3 pool — there is no separate "GPU VRAM" vs "CPU RAM".

## Modules

```bash
module load rocm/7.0.0                          # ROCm for CDNA3
module swap PrgEnv-cray/8.7.0 PrgEnv-gnu/8.7.0  # Or PrgEnv-cray / PrgEnv-amd depending on Cosmos config
module unload cray-mpich/9.1.0                  # MVAPICH-Plus replaces Cray MPICH 
module load craype-accel-amd-gfx942

# Cosmos cluster has some missing dynamic linkage thus we manually create symbolic links to those missing library packages
mkdir -p $HOME/lib
ln -sf /usr/lib64/libnl-3.so.200.25.0 $HOME/lib/libnl-3.so
ln -sf /usr/lib64/libjson-c.so.5  $HOME/lib/libjson-c.so
export LD_LIBRARY_PATH=$HOME/lib:$LD_LIBRARY_PATH
```

Check available modules:

```bash
module avail
module avail rocm
```

## Step 1: Build MVAPICH-Plus
Download MVAPICH-Plus from the official site: [https://mvapich.cse.ohio-state.edu/downloads/](https://mvapich.cse.ohio-state.edu/downloads/). If a pre-built package for Cosmos is available, you can use it and skip to [Step 2](#step-2-build-pytorch). Otherwise, build from source on a compute node:

```bash
# load modules
module load rocm/7.0.0
module swap PrgEnv-cray/8.7.0 PrgEnv-gnu/8.7.0
module unload cray-mpich/9.1.0
module load craype-accel-amd-gfx942

export LD_LIBRARY_PATH=$HOME/lib:$LD_LIBRARY_PATH

export MVP_HOME=/path/to/install/mvapich-plus-cuda13.2   # choose an install prefix

./autogen.sh && ./configure \
    --prefix=$MVP_HOME CC=/opt/cray/pe/gcc-native/14/bin/gcc CXX=/opt/cray/pe/gcc-native/14/bin/g++ \
    --with-device=ch4:ofi \
    --with-pmi=pmi2 --with-pmi2=/opt/cray/pe/pmi/default --with-pm=no \
    --with-libfabric=/opt/cray/libfabric/2.3.1 \
    --enable-threads=multiple \
    --enable-thread-cs=per-vci \
    --with-hip=$ROCM_PATH --enable-hip \
    --with-rocm=$ROCM_PATH --enable-rocm \
    --enable-rcclomb --with-rccl=$ROCM_PATH \
    --enable-unified-gpu \
    --enable-fast=opt --disable-cl --disable-opencl --disable-gl --disable-nvml \
    LDFLAGS="-L$HOME/lib"

make -j16 install
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


### 2.2 Environment Setup

```bash
# Install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p /path/to/miniconda3                # Choose your path to miniconda3

source /path/to/miniconda3/bin/activate

# set the following flags to avoid broken environment
conda config --set auto_activate_base false
conda config --set channel_priority strict

# Create and activate a conda virtual environment
conda create -n name-your-venv python=3.12 -y                                   # Choose a name for your conda virtual environment
conda activate name-your-venv

cd pytorch
pip install -r requirements.txt      # from the PyTorch source tree
# also install your training script's packages now (for nanoGPT: numpy, tiktoken, ...)
```

### 2.3 Load Environment Variables

```bash
#load MPI related environment variables
export CPATH=$MVP_HOME/include:$CPATH
export PATH=$MVP_HOME/bin:$PATH
export LD_LIBRARY_PATH=$MVP_HOME/lib64:$LD_LIBRARY_PATH
export LD_PRELOAD=$MVP_HOME/lib64/libmpi.so
export MPI_C_COMPILER=$MVP_HOME/bin/mpicc
export MPI_CXX_COMPILER=$MVP_HOME/bin/mpicxx
export CFLAGS="-I$MVP_HOME/include $CFLAGS"
export CXXFLAGS="-I$MVP_HOME/include $CXXFLAGS"
export LDFLAGS="-L$MVP_HOME/lib64 $LDFLAGS"

#load cmake related environment variables 
export _GLIBCXX_USE_CXX11_ABI=1
export CMAKE_CXX_COMPILER=/opt/cray/pe/gcc-native/14/bin/g++
export CMAKE_C_COMPILER=/opt/cray/pe/gcc-native/14/bin/gcc
export CMAKE_PREFIX_PATH="${MVP_HOME}:$LD_LIBRARY_PATH:$CPATH:$PATH:${CMAKE_PREFIX_PATH}"

#load RCCL related environment variables
export RCCL_HOME=$ROCM_PATH
export RCCL_INCLUDE_DIR=$RCCL_HOME/include/rccl
export RCCL_LIB_DIR=$RCCL_HOME/lib
export USE_ROCM=1
export USE_NCCL=1
export USE_SYSTEM_NCCL=1
```

### 2.4 Pytorch installation

```bash
cd pytorch
python tools/amd_build/build_amd.py # hipify the CUDA sources for ROCm

MAX_JOBS=16 \
USE_XNNPACK=0 \
USE_MSLK=0 \
USE_MPI=1 \
USE_CUDA_MPI=1 \
CC=/opt/cray/pe/gcc-native/14/bin/gcc \
CXX=/opt/cray/pe/gcc-native/14/bin/g++ \
USE_DISTRIBUTED=1 \
BUILD_TEST=0 \
BUILD_MOBILE_BENCHMARK=0 \
BUILD_MOBILE_TEST=0 \
PYTORCH_ROCM_ARCH="gfx942" CMAKE_POLICY_VERSION_MINIMUM=3.5 \
python setup.py develop

```
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
Expected: a `2.14.1` version, a `7.0.0` HIP version, `mpi True`, `nccl True`, and RCCL `(2, 26, 6)`. Run the GPU check on a compute node: `torch.cuda.device_count()` should return 4.

## Step 3: Runtime Environment

Save this as `~/hpc-ai-consmos-env.sh` and source it in your job scripts:

```bash
#!/bin/bash
module load rocm/7.0.0
module swap PrgEnv-cray/8.7.0 PrgEnv-gnu/8.7.0
module unload cray-mpich/9.1.0
module load craype-accel-amd-gfx942

export LD_LIBRARY_PATH=$HOME/lib:$LD_LIBRARY_PATH

export MVP_HOME=/path/to/install/mvapich-plus-cuda13.2
export CPATH=$MVP_HOME/include:$CPATH
export PATH=$MVP_HOME/bin:$PATH
export LD_LIBRARY_PATH=$MVP_HOME/lib64:$LD_LIBRARY_PATH
export LD_PRELOAD=$MVP_HOME/lib64/libmpi.so

source /path/to/miniconda3/bin/activate
conda activate your-venv

export HF_HOME=$HOME/.cache/huggingface
export MPICH_GPU_SUPPORT_ENABLED=1

export NCCL_NET_PLUGIN=ofi
export FI_MR_CACHE_MONITOR=userfaultfd
```

## Running PyTorch DDP

```bash
#!/bin/bash
#SBATCH --job-name=ddp-test
#SBATCH --output=ddp-test.out
#SBATCH --error=ddp-test.err
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time 00:30:00

NP=$((${SLURM_JOB_NUM_NODES} * 4))
NODES=$SLURM_JOB_NUM_NODES

echo $NP
echo $NODES

echo "**************************************************************"
echo "----------STARTING MPI DDP---------------"
echo "**************************************************************"

srun -n $NP -N $NODES bash -c '
  source ~/hpc-ai-consmos-env.sh
  MPIR_CVAR_ENABLE_GPU=1 \
  MPIR_CVAR_ALLREDUCE_INTRA_ALGORITHM=osu_gpu_rsa \
  MPIR_CVAR_ALLREDUCE_COMPOSITION=8 \
  MPIR_CVAR_ALLREDUCE_IPC_MSG_SIZE_THRESHOLD=131072 \
  python train.py --backend mpi
'
wait


echo "**************************************************************"
echo "----------STARTING RCCL DDP---------------"
echo "**************************************************************"


srun -n $NP -N $NODES bash -c '
  source ~/hpc-ai-cosmos-env.sh
  python train.py --backend nccl
'
wait

echo "**************************************************************"
echo "----------COMPLETED DDP TESTS---------------"
echo "**************************************************************"
```

## Queues

```bash
#  Interactive, 1 node
salloc --nodes=1 --time 04:00:00 --exclusive

# Batch
sbatch jobscriptfile    # setup jobscript file as per https://www.sdsc.edu/systems/cosmos/user_guide.html#narrow-wysiwyg-4 
```
Check the [SDSC Cosmos User Guide](https://www.sdsc.edu/systems/cosmos/user_guide.html) for the more details.


## Additional Resources

- **Troubleshooting Guide**: [../troubleshooting.md](../troubleshooting.md)
- **SDSC Cosmos User Guide**: [https://www.sdsc.edu/support/user_guides/cosmos.html](https://www.sdsc.edu/support/user_guides/cosmos.html)
- **ROCm Documentation**: [https://rocm.docs.amd.com/](https://rocm.docs.amd.com/)
- **Official HPC-AI Userguide**: [https://hpc-ai.engineering.osu.edu/userguide/](https://hpc-ai.engineering.osu.edu/userguide/)
