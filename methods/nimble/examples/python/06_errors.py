"""Errors and shutdown: what raises where, and a pattern that never leaves ranks hanging.

- Argument errors (shapes, dtypes, devices, splits that disagree with the table) raise ValueError on the calling
  rank before anything is enqueued; the communicator stays usable.
- The library's own refusals raise nimble.NimbleError with .code and .name (e.g. NIMBLE_ERROR_INVALID_ARGUMENT).
- Failures on the GPU or the network are latched: comm.async_error() returns the code (0 = none),
  comm.raise_async_error() raises it. Check after synchronizing.
- destroy() is collective. If a rank cannot continue, abort() first: the others' calls then fail instead of waiting.

    torchrun --nproc-per-node 4 06_errors.py
"""
import torch
import torch.distributed as dist

import nimble

dist.init_process_group("nccl")
comm = nimble.init()
rank, world = dist.get_rank(), dist.get_world_size()
device = torch.device("cuda", comm.device)
table = torch.full((world, world), 16)
splits = [16] * world
x = torch.randn(world * 16, 512, device=device)
y = torch.empty_like(x)
seen = []

# 1. Argument errors: raised before the call reaches the library, on the calling rank only.
for bad in (dict(output=y.half()),                              # dtype differs from the input
            dict(output=torch.empty(world * 16, 256, device=device)),   # trailing dimensions differ
            dict(input=x.t()),                                  # not contiguous
            dict(input_splits=[17] + splits[1:]),               # splits disagree with the table
            dict(output=y.cpu())):                              # not a CUDA tensor
    args = dict(output=y, input=x, input_splits=splits) | bad
    try:
        nimble.alltoallv(args["output"], args["input"], splits, args["input_splits"], matrix=table)
    except ValueError as error:
        seen.append(type(error).__name__)

# 2. A refusal by the library itself: it accepts one explicit stream or the legacy default stream, never
#    cudaStreamPerThread (raw handle 2). Refusals like this one come before any work is admitted, so the communicator
#    stays usable. (Faults found after a call returns -- a buffer smaller than the table says, a GPU or network
#    failure -- latch the communicator instead; see 3.)
bytes_table = (table * x[0].numel() * x.element_size()).numpy()
try:
    comm.alltoallv_bytes(x.data_ptr(), y.data_ptr(), bytes_table, 2)
except nimble.NimbleError as error:
    seen.append(error.name)

# 3. The communicator is still healthy: a normal call, then the latched asynchronous state.
nimble.alltoallv(y, x, splits, splits, matrix=table)
torch.cuda.current_stream().synchronize()
comm.raise_async_error()                             # no-op when nothing failed (comm.async_error() == 0)

# 4. Shutdown that cannot hang: agree on success first; abort before destroy if any rank failed.
ok = torch.tensor([int(seen == ["ValueError"] * 5 + ["NIMBLE_ERROR_INVALID_ARGUMENT"])], device=device)
dist.all_reduce(ok, op=dist.ReduceOp.MIN)
if not ok.item():
    comm.abort()                                     # the other ranks' pending calls fail rather than wait
try:
    nimble.destroy()                                 # collective; after abort() it raises the latched error
except nimble.NimbleError as error:
    print(f"rank {rank}: destroy after abort: {error.name}")
if rank == 0:
    print(f"06_errors: {'OK' if ok.item() else 'UNEXPECTED'}; rank 0 saw {seen}")
dist.destroy_process_group()
