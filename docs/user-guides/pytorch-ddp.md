# PyTorch Distributed Data Parallel with MPI

This guide covers running PyTorch Distributed Data Parallel (DDP) training using the MPI backend with MVAPICH-Plus.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Initializing the MPI Backend](#initializing-the-mpi-backend)
- [Running DDP Training](#running-ddp-training)
- [MVAPICH-Plus Tuning](#mvapich-plus-tuning)
- [Common Patterns](#common-patterns)
- [Troubleshooting](#troubleshooting)

## Overview

PyTorch's Distributed Data Parallel (DDP) enables training across multiple GPUs and nodes. The MPI backend, powered by MVAPICH-Plus, provides:

- **CUDA-aware MPI**: Direct GPU-to-GPU communication without CPU staging
- **GPUDirect RDMA**: Low-latency transfers over InfiniBand/Slingshot
- **Vendor-neutral**: Works across NVIDIA and AMD GPUs
- **HPC integration**: Native compatibility with SLURM, PBS, and other job schedulers

> **System-specific setup**: build steps and job scripts for each cluster are in the cluster guides: [OLCF Frontier](../cluster-guides/olcf-frontier-setup.md), [TACC Horizon](../cluster-guides/tacc-horizon-setup.md), [TACC Vista](../cluster-guides/tacc-vista-setup.md), [SDSC Cosmos](../cluster-guides/sdsc-cosmos-setup.md), [MRI](../cluster-guides/mri-setup.md).

## Prerequisites

### MVAPICH-Plus

MVAPICH-Plus is the preferred MPI runtime for DDP training with PyTorch on GPUs and CPUs.

- **Download**: [https://mvapich.cse.ohio-state.edu/downloads/](https://mvapich.cse.ohio-state.edu/downloads/)
- **User Guide**: [https://mvapich-docs.readthedocs.io/en/mvapich-plus/](https://mvapich-docs.readthedocs.io/en/mvapich-plus/)

### PyTorch with GPU-Aware MPI

We provide an open-source PyTorch branch with enhanced GPU-Aware MPI support:

- **Repository**: [https://github.com/OSU-Nowlab/pytorch](https://github.com/OSU-Nowlab/pytorch) (tag `hpc_ai_v1.1`, PyTorch 2.14.1)
- **Official PyTorch Build Guide**: [https://github.com/pytorch/pytorch#installation](https://github.com/pytorch/pytorch#installation)

To enable GPU-Aware MPI when building, add `USE_CUDA_MPI=1` to your setup command:

```bash
USE_CUDA_MPI=1 USE_MPI=1 python setup.py install
```

On AMD GPUs, the same flag enables ROCm-aware MPI. For the full ROCm build on Frontier, see the [OLCF Frontier Setup Guide](../cluster-guides/olcf-frontier-setup.md#step-2-build-pytorch).

## Quick Start

### Basic DDP Script

```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Initialize MPI backend
dist.init_process_group(backend='mpi')

# Get rank and device
rank = dist.get_rank()
device = torch.device(f'cuda:{rank % torch.cuda.device_count()}')
torch.cuda.set_device(device)

# Create model and wrap with DDP
model = YourModel().to(device)
model = DDP(model)

# Training loop
for batch in dataloader:
    outputs = model(batch.to(device))
    loss = criterion(outputs, labels.to(device))
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

# Cleanup
dist.destroy_process_group()
```

### Launch with mpirun

```bash
# Single node, 4 GPUs
mpirun -np 4 python train.py

# Multi-node (2 nodes, 4 GPUs each)
mpirun -np 8 -ppn 4 -hostfile hosts.txt python train.py
```

### Launch with srun (SLURM)

```bash
# Single node
srun -n 4 --gpus-per-task=1 python train.py

# Multi-node
srun -N 2 -n 8 --gpus-per-node=4 python train.py
```

## Initializing the MPI Backend

### Standard Initialization

```python
import torch.distributed as dist

# MPI backend handles rank/world_size automatically from mpirun/srun
dist.init_process_group(backend='mpi')

# Get process information
rank = dist.get_rank()
world_size = dist.get_world_size()
```

### With Explicit Configuration

```python
import os
import torch.distributed as dist

# MPI provides these automatically, but you can access them.
# MVAPICH-Plus mpirun sets PMI_* / MPI_LOCALRANKID; srun sets SLURM_*.
rank = int(os.environ.get('PMI_RANK', os.environ.get('SLURM_PROCID', 0)))
world_size = int(os.environ.get('PMI_SIZE', os.environ.get('SLURM_NTASKS', 1)))
local_rank = int(os.environ.get('MPI_LOCALRANKID', os.environ.get('SLURM_LOCALID', 0)))

dist.init_process_group(backend='mpi')
```

### Verifying MPI Availability

```python
import torch.distributed as dist

# Check if MPI backend is available
if dist.is_mpi_available():
    print("MPI backend is available")
else:
    print("MPI backend not available - was PyTorch built with USE_MPI=1?")
```

## Running DDP Training

### Single Node Multi-GPU

```bash
# Launch with mpirun
mpirun -np 4 python train.py

# Or with srun
srun -n 4 --gpus-per-task=1 python train.py
```

### Multi-Node Training

Create a hostfile listing your nodes:

```text
# hosts.txt
node1 slots=4
node2 slots=4
```

Launch:

```bash
# With mpirun
mpirun -np 8 -ppn 4 -hostfile hosts.txt python train.py

# With srun (SLURM handles node allocation)
srun -N 2 -n 8 --ntasks-per-node=4 python train.py
```

### SLURM Batch Script Example

```bash
#!/bin/bash
#SBATCH -N 2                      # Number of nodes
#SBATCH -n 8                      # Total tasks (GPUs)
#SBATCH --ntasks-per-node=4       # GPUs per node
#SBATCH --gpus-per-node=4         # Request GPUs
#SBATCH -t 4:00:00                # Time limit
#SBATCH -p gpu                    # GPU partition

# Load environment
source ~/setup_hpc_ai.sh

# Run training
srun python train.py
```

## MVAPICH-Plus Tuning

MVAPICH-Plus provides environment variables (CVARs) to optimize collective operations.

### General Recommendations

```bash
# Enable GPU-aware MPI
export MPIR_CVAR_ENABLE_GPU=1
```

### NVIDIA TACC-Vista InfiniBand Networks (NDR/HDR)

```bash
# Note that the osu_gpu_direct algorithm is optimized for Vista's 1GPU per node setup
export MPIR_CVAR_ENABLE_GPU=1
export MPIR_CVAR_ALLREDUCE_INTRA_ALGORITHM=osu_gpu_direct
export MPIR_CVAR_ALLREDUCE_THROTTLE=4
export MPIR_CVAR_ALLREDUCE_COMPOSITION=2
export MPIR_CVAR_ALLREDUCE_IPC_MSG_SIZE_THRESHOLD=65536
```

### AMD MI250X Slingshot (OLCF Frontier)

These settings were used for the PyTorch 2.14.1 release runs on Frontier: GPT-2 DDP, bf16, 1-64 nodes, 8 ranks per node, ROCm 7.13.0. For the full build steps and job scripts, see the [OLCF Frontier Setup Guide](../cluster-guides/olcf-frontier-setup.md).

**MVAPICH-Plus:**

```bash
export MPIR_CVAR_ENABLE_GPU=1                                 # GPU-aware MPI
export MPICH_GPU_SUPPORT_ENABLED=1
export MPIR_CVAR_CH4_OFI_ENABLE_HMEM=1                        # pass GPU buffers to libfabric
export MPIR_CVAR_CH4_OFI_ENABLE_MR_HMEM=1                     # register GPU memory with the NIC
export MPIR_CVAR_ALLREDUCE_COMPOSITION=8                      # two-level allreduce: GPU IPC within a node, network between nodes
export MPIR_CVAR_ALLREDUCE_INTRA_ALGORITHM=osu_gpu_rsa        # network-side allreduce algorithm
export MPIR_CVAR_ALLREDUCE_GPU_RING_THRESHOLD=67108864        # 64 MB: messages at or above this use the intra-node IPC ring
export MPIR_CVAR_ALLREDUCE_PIPELINE_2LVL_GPU_THRESHOLD=8388608  # 8 MB
export MPIR_CVAR_ALLREDUCE_PIPELINE_2LVL_GPU_NUM_CHUNKS=1
export MPIR_CVAR_FORCE_GPU_QUERY_ATTR=1
export MPIR_CVAR_ALLGATHER_COMPOSITION=1
export MVP_ENABLE_GPU_PTRCACHE=0
export HSA_XNACK=1
```

**Slingshot / libfabric** (the same values the `rccl-net-plugin/1.0` module sets for RCCL, so both backends run with the same network settings):

```bash
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
```

### NVIDIA GB200 InfiniBand (TACC Horizon)

Horizon nodes have 4 GB200 GPUs and 4 InfiniBand HCAs, and we run one rank per GPU. For the build steps, the per-rank wrapper and the job scripts, see the [TACC Horizon Setup Guide](../cluster-guides/tacc-horizon-setup.md).

**MVAPICH-Plus:**

```bash
export MPIR_CVAR_ENABLE_GPU=1                     # GPU-aware MPI
export MPICH_GPU_SUPPORT_ENABLED=1
export MVP_ENABLE_GPU=1
export MPIR_CVAR_REDUCE_KERNEL_MAX_NUM_CTAS=32    
export MPIR_CVAR_REDUCE_KERNEL_MAX_NUM_THREADS=544
export MPIR_CVAR_PMI_VERSION=1                    # PMI-1 with the Hydra mpirun (what our Horizon runs used)
export MPIR_CVAR_ALLREDUCE_COMPOSITION=8          # 2+ nodes only: two-level allreduce, GPU IPC within a node, network between nodes
```

### AMD Slingshot (SDSC Cosmos)

```bash
export MPIR_CVAR_ENABLE_GPU=1
export MPIR_CVAR_PMI_VERSION=2
export MPIR_CVAR_CH4_OFI_ENABLE_HMEM=1
export MPIR_CVAR_CH4_OFI_ENABLE_MR_HMEM=1
export MPIR_CVAR_ALLREDUCE_COMPOSITION=8
export MPIR_CVAR_ALLREDUCE_INTRA_ALGORITHM=osu_gpu_rsa
```

### Setting CVARs in Job Scripts

With mpirun:
```bash
mpirun -np 4 \
    -genv MPIR_CVAR_ENABLE_GPU=1 \
    -genv MPIR_CVAR_ALLREDUCE_INTRA_ALGORITHM=osu_gpu_direct \
    python train.py
```

With srun:
```bash
srun -n 4 --export=ALL,MPIR_CVAR_ENABLE_GPU=1,MPIR_CVAR_ALLREDUCE_INTRA_ALGORITHM=osu_gpu_direct \
    python train.py
```

## Common Patterns

### Gradient Accumulation

```python
accumulation_steps = 4

for i, batch in enumerate(dataloader):
    outputs = model(batch)
    loss = criterion(outputs, labels) / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### Mixed Precision Training

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in dataloader:
    optimizer.zero_grad()

    with autocast():
        outputs = model(batch)
        loss = criterion(outputs, labels)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### Checkpointing

```python
# Save checkpoint (only on rank 0)
if dist.get_rank() == 0:
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.module.state_dict(),  # Note: .module for DDP
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, 'checkpoint.pt')

# Synchronize before continuing
dist.barrier()
```

### Data Loading

```python
from torch.utils.data import DataLoader, DistributedSampler

# Create distributed sampler
sampler = DistributedSampler(
    dataset,
    num_replicas=dist.get_world_size(),
    rank=dist.get_rank(),
    shuffle=True
)

# Create dataloader with sampler
dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    sampler=sampler,
    num_workers=4,
    pin_memory=True
)

# Update sampler epoch for proper shuffling
for epoch in range(num_epochs):
    sampler.set_epoch(epoch)
    for batch in dataloader:
        # training loop
```

## Troubleshooting

### MPI Backend Not Available

**Error**: `RuntimeError: MPI backend not available`

**Cause**: PyTorch was not built with MPI support.

**Solution**: Verify build flags and rebuild if needed:
```bash
python -c "import torch.distributed as dist; print(dist.is_mpi_available())"
# Should print: True
```

### CUDA Device Mismatch

**Error**: `RuntimeError: Expected all tensors to be on the same device`

**Cause**: Model and data on different GPUs.

**Solution**: Set device based on local rank:
```python
local_rank = int(os.environ.get('MPI_LOCALRANKID', os.environ.get('SLURM_LOCALID', 0)))
device = torch.device(f'cuda:{local_rank % torch.cuda.device_count()}')
torch.cuda.set_device(device)
```

### Hanging on init_process_group

**Cause**: Network connectivity issues or mismatched world sizes.

**Solution**:
1. Verify all nodes can communicate
2. Check hostfile matches SLURM allocation
3. Ensure MPI environment is consistent across nodes

### Out of Memory

**Cause**: Batch size too large for GPU memory.

**Solution**:
1. Reduce batch size per GPU
2. Use gradient accumulation
3. Enable mixed precision training

## Additional Resources

- [MVAPICH-Plus Documentation](https://mvapich-docs.readthedocs.io/en/mvapich-plus/)
- [MVAPICH-Plus Downloads](https://mvapich.cse.ohio-state.edu/downloads/)
- [PyTorch Distributed Overview](https://pytorch.org/tutorials/beginner/dist_overview.html)
- [HPC-AI Project Homepage](https://hpc-ai.engineering.osu.edu/)
- [OSU-Nowlab PyTorch Fork](https://github.com/OSU-Nowlab/pytorch)
