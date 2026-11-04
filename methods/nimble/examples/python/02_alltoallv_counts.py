"""Variable sizes (AllToAllV) with the count table: the MoE token shuffle, sequence-parallel resharding, KV transfer.

Each rank sends a different number of rows to each rank. Applications exchange these counts anyway; handing NIMBLE
the full table (`matrix=`, rows[src][dst]) lets it plan the call without a second collective on the critical path.

    torchrun --nproc-per-node 4 02_alltoallv_counts.py
"""
import torch
import torch.distributed as dist

import nimble

dist.init_process_group("nccl")
nimble.init()
rank, world = dist.get_rank(), dist.get_world_size()
device = torch.device("cuda", torch.cuda.current_device())
hidden = 7168

# 1. This rank's send counts, in rows, one entry per destination (here random, as a router would produce them).
torch.manual_seed(1000 + rank)
send_counts = torch.randint(0, 512, (world,), device=device)

# 2. The count table, the same on every rank: one all_gather of the send counts.
table = torch.empty(world, world, dtype=send_counts.dtype, device=device)
dist.all_gather_into_tensor(table, send_counts)
table = table.cpu()                                # host copy; NIMBLE plans on the host
input_splits = table[rank].tolist()                # rows this rank sends to each rank
output_splits = table[:, rank].tolist()            # rows this rank receives from each rank

# 3. `inp` is packed by destination rank, `out` by source rank, along dim 0. Any dtype, any trailing shape;
#    tensors must be contiguous CUDA tensors on this rank's device. No registration: any allocation works.
inp = torch.randn(sum(input_splits), hidden, device=device).to(torch.bfloat16)
out = torch.empty(sum(output_splits), hidden, dtype=torch.bfloat16, device=device)
nimble.alltoallv(out, inp, output_splits, input_splits, matrix=table)

# 4. Without `matrix`, NIMBLE first exchanges the rows itself (one extra collective and a host wait).
out_auto = torch.empty_like(out)
nimble.alltoallv(out_auto, inp, output_splits, input_splits)
torch.cuda.current_stream().synchronize()

ref = torch.empty_like(out)
dist.all_to_all_single(ref, inp, output_split_sizes=output_splits, input_split_sizes=input_splits)
torch.cuda.synchronize()
ok = torch.tensor([int(torch.equal(out, ref) and torch.equal(out_auto, ref))], device=device)
dist.all_reduce(ok, op=dist.ReduceOp.MIN)
if rank == 0:
    print(f"02_alltoallv_counts: {'OK' if ok.item() else 'MISMATCH'}; rank 0 sent {input_splits} rows of "
          f"{hidden} bf16")

nimble.destroy()
dist.destroy_process_group()
