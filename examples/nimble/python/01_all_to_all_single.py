"""Quickstart: NIMBLE as a drop-in for torch.distributed.all_to_all_single.

    torchrun --nproc-per-node 4 01_all_to_all_single.py            # one node; more nodes: launch_slurm.sh
"""
import torch
import torch.distributed as dist

import nimble

dist.init_process_group("nccl")          # torchrun sets RANK, WORLD_SIZE, LOCAL_RANK, MASTER_ADDR, MASTER_PORT
nimble.init()                            # collective: the default communicator over the world, on cuda:LOCAL_RANK
rank, world = dist.get_rank(), dist.get_world_size()
device = torch.device("cuda", torch.cuda.current_device())

# Equal split: rank r sends rows [d * chunk, (d + 1) * chunk) of `inp` to rank d.
chunk, width = 64, 1024
inp = torch.randn(world * chunk, width, device=device)
out = torch.empty_like(inp)

# 1. The drop-in: the same arguments as torch.distributed.all_to_all_single (async_op is not offered).
nimble.all_to_all_single(out, inp)
torch.cuda.current_stream().synchronize()      # the call is stream-ordered: synchronize before reading `out`

# 2. The same exchange when every rank knows the count table (rows[src][dst]): no count exchange inside the call.
rows = torch.full((world, world), chunk)
out_fast = torch.empty_like(inp)
nimble.alltoallv(out_fast, inp, [chunk] * world, [chunk] * world, matrix=rows)
torch.cuda.current_stream().synchronize()

ref = torch.empty_like(inp)
dist.all_to_all_single(ref, inp)               # NCCL, for comparison
torch.cuda.synchronize()
ok = torch.tensor([int(torch.equal(out, ref) and torch.equal(out_fast, ref))], device=device)
dist.all_reduce(ok, op=dist.ReduceOp.MIN)
if rank == 0:
    print(f"01_all_to_all_single: {'OK' if ok.item() else 'MISMATCH'} on {world} ranks "
          f"({nimble.__version__}, {nimble.version_string()})")

nimble.destroy()                         # collective, before the process group goes
dist.destroy_process_group()
