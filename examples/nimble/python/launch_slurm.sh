#!/bin/bash
# Run a NIMBLE script on N nodes of 4 GPUs: one torchrun per node, started by srun.
#
#   sbatch -N 2 [-p PARTITION -A ACCOUNT] launch_slurm.sh 01_all_to_all_single.py [ARGS...]
#   NIMBLE_DEDUP=1 PYTHON=/path/to/venv/bin/python sbatch -N 4 launch_slurm.sh 07_moe_dispatcher.py --mode dedup
#
# The environment reaches every rank (srun exports it), so NIMBLE_* variables set before sbatch apply. Add
# --gpus-per-node=4 where the cluster schedules GPUs as resources. nimble.moe's kernels are Triton's, compiled on
# first use: Triton gets a cache on each node's /tmp unless TRITON_CACHE_DIR is set (ranks of several nodes compiling
# into one cache on a shared file system can fail with "Stale file handle") and gcc for its C helpers (a site's
# default CC, such as NVHPC's nvc, rejects Triton's flags).
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --time=00:10:00
set -eu
HOSTS=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
MASTER_ADDR=${HOSTS[0]}
MASTER_PORT=${MASTER_PORT:-29500}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/tmp/triton_cache_$USER} CC=gcc
srun --cpu-bind=none --ntasks-per-node=1 bash -c \
  "${PYTHON:-python3} -m torch.distributed.run --nnodes $SLURM_NNODES --node-rank \$SLURM_NODEID \
   --nproc-per-node 4 --master-addr $MASTER_ADDR --master-port $MASTER_PORT $*"
