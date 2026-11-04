"""Streams: NIMBLE on a side stream, overlapped with compute; back-to-back calls; one communicator per stream.

A communicator is bound to the CUDA stream of its first call. Later calls on another stream raise ValueError before
anything is enqueued, so a second stream needs a second communicator.

    torchrun --nproc-per-node 4 03_streams.py
"""
import torch
import torch.distributed as dist

import nimble

dist.init_process_group("nccl")
comm = nimble.init()                                 # the default communicator; its stream is chosen below
rank, world = dist.get_rank(), dist.get_world_size()
device = torch.device("cuda", torch.cuda.current_device())
rows = torch.full((world, world), 256)               # 256 rows of 4096 floats per pair
splits = [256] * world
compute = torch.cuda.current_stream()
comm_stream = torch.cuda.Stream()

# Produce the input on the compute stream, hand it to the communication stream with an event.
x = torch.randn(world * 256, 4096, device=device)
y = torch.empty_like(x)
ready = torch.cuda.Event()
ready.record(compute)
comm_stream.wait_event(ready)

# `stream=` takes a torch.cuda.Stream (or an ExternalStream, or a raw cudaStream_t integer); on a stream other than
# the current one NIMBLE calls record_stream on both tensors, so the allocator keeps them until the transfer is done.
nimble.alltoallv(y, x, splits, splits, matrix=rows, stream=comm_stream)   # binds `comm` to comm_stream

# Independent compute overlaps with the transfer.
a = torch.randn(4096, 4096, device=device)
b = a @ a

# Consume `y` on the compute stream only after the transfer: an event, no host wait.
done = torch.cuda.Event()
done.record(comm_stream)
compute.wait_event(done)
checksum = y.sum()

# Back-to-back calls need no synchronization in between: they execute in stream order. Inside
# `with torch.cuda.stream(...)`, stream=None means that (current) stream.
outs = [torch.empty_like(x) for _ in range(4)]
with torch.cuda.stream(comm_stream):
    for out in outs:
        nimble.alltoallv(out, x, splits, splits, matrix=rows)
comm_stream.synchronize()

# The communicator stays bound to comm_stream: the compute stream is refused before anything is enqueued.
try:
    nimble.alltoallv(outs[0], x, splits, splits, matrix=rows, stream=compute)
    refused = False
except ValueError:
    refused = True

# A second stream gets its own communicator (collective; each holds its own buffers).
other_stream = torch.cuda.Stream()
other_stream.wait_stream(compute)                    # `x` was produced on the compute stream
comm2 = nimble.init(set_default=False)
z = torch.empty_like(x)
nimble.alltoallv(z, x, splits, splits, matrix=rows, comm=comm2, stream=other_stream)
other_stream.synchronize()

torch.cuda.synchronize()
ok = refused and all(torch.equal(o, y) for o in outs) and torch.equal(z, y) and torch.isfinite(checksum).item()
ok = torch.tensor([int(ok)], device=device)
dist.all_reduce(ok, op=dist.ReduceOp.MIN)
if rank == 0:
    print(f"03_streams: {'OK' if ok.item() else 'MISMATCH'}; {comm} bound to stream {comm.bound_stream:#x}")

comm2.destroy()                                      # collective
nimble.destroy()
dist.destroy_process_group()
