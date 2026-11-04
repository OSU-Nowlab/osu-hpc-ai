"""Communicators over torch.distributed subgroups: here one per node (NVLink only) beside one over the world.

A NIMBLE communicator spans whole nodes (4 GPUs each), so a subgroup is a set of whole nodes. Create it with
set_default=False and pass it as comm=; ranks and the count table are the group's. Every rank creates every
torch.distributed group, as torch requires.

    torchrun --nproc-per-node 4 04_subgroups.py                    # one node: the node group is the world
"""
import os

import torch
import torch.distributed as dist

import nimble

dist.init_process_group("nccl")
rank, world = dist.get_rank(), dist.get_world_size()
gpus = int(os.environ["LOCAL_WORLD_SIZE"])            # 4 on GB200 nodes
device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))

world_comm = nimble.init(set_default=False)
node_groups = [dist.new_group(list(range(n * gpus, (n + 1) * gpus))) for n in range(world // gpus)]
my_group = node_groups[rank // gpus]
node_comm = nimble.init(group=my_group, set_default=False)        # collective over the node's ranks
assert (node_comm.rank, node_comm.world_size) == (dist.get_rank(my_group), gpus)


def exchange(comm, group, rows_per_pair, width):
    """An equal-split exchange on `comm`, checked against NCCL on the same group."""
    n = comm.world_size
    table = torch.full((n, n), rows_per_pair)
    x = torch.randn(n * rows_per_pair, width, device=device)
    y = torch.empty_like(x)
    nimble.alltoallv(y, x, [rows_per_pair] * n, [rows_per_pair] * n, matrix=table, comm=comm)
    ref = torch.empty_like(x)
    dist.all_to_all_single(ref, x, group=group)
    torch.cuda.synchronize()
    return torch.equal(y, ref)


ok = exchange(node_comm, my_group, 1024, 2048) and exchange(world_comm, None, 128, 2048)
ok = torch.tensor([int(ok)], device=device)
dist.all_reduce(ok, op=dist.ReduceOp.MIN)
if rank == 0:
    print(f"04_subgroups: {'OK' if ok.item() else 'MISMATCH'}; {world // gpus} node communicator(s) and the world's")

node_comm.destroy()                                   # collective over the node
world_comm.destroy()                                  # collective over the world
dist.destroy_process_group()
