"""NIMBLE without torch.distributed, on raw device pointers (CuPy, JAX, custom allocators, other launchers).

nimble.init() with no process group reads RANK, WORLD_SIZE, LOCAL_RANK, LOCAL_WORLD_SIZE and MASTER_ADDR from the
environment (torchrun sets them) and rendezvous at NIMBLE_BOOTSTRAP, or tcp://MASTER_ADDR:(MASTER_PORT + 1). A custom
launcher passes them as keywords instead:
    nimble.init(rank=r, world_size=n, local_rank=l, local_world_size=4, bootstrap="tcp://host0:29600")

Communicator.alltoallv_bytes() is the C call underneath nimble.alltoallv: send and receive device pointers and the
nranks-by-nranks byte table (source by row), the same on every rank. Here the application knows the table without a
count exchange: rank s sends (s + d + 1) KiB to rank d.

    torchrun --nproc-per-node 4 05_no_torch_distributed.py
"""
import numpy as np
import torch

import nimble

comm = nimble.init()                                  # no process group: the torchrun environment
rank, world, device = comm.rank, comm.world_size, comm.device
kib = 1024
table = np.array([[(s + d + 1) * kib for d in range(world)] for s in range(world)], dtype=np.uint64)

# Any device memory works; here torch allocates it. Byte b of the block for rank d holds (rank + d + b) mod 251.
send = torch.cat([(torch.arange(int(table[rank, d]), device=device) + rank + d) % 251
                  for d in range(world)]).to(torch.uint8)
recv = torch.empty(int(table[:, rank].sum()), dtype=torch.uint8, device=device)
stream = torch.cuda.current_stream(device)
comm.alltoallv_bytes(send.data_ptr(), recv.data_ptr(), table, stream.cuda_stream)
stream.synchronize()
comm.raise_async_error()                              # raises NimbleError if the collective failed on the GPU

# Received bytes are packed by source: the block from rank s holds (s + rank + b) mod 251.
want = torch.cat([(torch.arange(int(table[s, rank]), device=device) + s + rank) % 251
                  for s in range(world)]).to(torch.uint8)
ok = torch.equal(recv, want)
print(f"05_no_torch_distributed: rank {rank}: {'OK' if ok else 'MISMATCH'} ({recv.numel()} bytes from {world} ranks)")
comm.destroy()                                        # collective
