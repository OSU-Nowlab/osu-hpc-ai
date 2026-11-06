"""Expert-parallel MoE with nimble.moe.Dispatcher: route, dispatch, run the local experts, combine.

dispatch sends each token once to every rank that holds one of its top-k experts (DeepEP's normal layout): the row
carries the token's activations and scales, its top-k ids as this rank's local expert indices (-1 for experts that
live elsewhere) and their weights (0 there). The experts return, per received row, their outputs already weighted
and summed over that row's local experts; combine sends them back and sums each token over its ranks.

    torchrun --nproc-per-node 4 07_moe_dispatcher.py                    # fused mode, FP8 with 128-element scales
    torchrun --nproc-per-node 4 07_moe_dispatcher.py --dtype bf16       # BF16 activations, no scales
    torchrun --nproc-per-node 4 07_moe_dispatcher.py --transport torch  # the same dispatcher over NCCL, for comparison
    NIMBLE_DEDUP=1 sbatch -N 2 launch_slurm.sh 07_moe_dispatcher.py --mode dedup   # two or more nodes
"""
import argparse
import os

import torch
import torch.distributed as dist

import nimble
from nimble.moe import Dispatcher

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=("fused", "dedup"), default="fused")
ap.add_argument("--transport", choices=("nimble", "torch"), default="nimble")
ap.add_argument("--dtype", choices=("fp8", "bf16"), default="fp8")
ap.add_argument("--tokens", type=int, default=2048, help="tokens per rank and call")
args = ap.parse_args()

dist.init_process_group("nccl")
rank, world = dist.get_rank(), dist.get_world_size()
device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
torch.cuda.set_device(device)
E, H, K, T = 64, 2048, 8, args.tokens            # experts, hidden size, top-k, tokens on this rank
fp8 = args.dtype == "fp8"

# One communicator per process serves every MoE layer; the dedup mode needs NIMBLE_DEDUP=1 before it is created.
ep_group = dist.group.WORLD                      # the expert-parallel group: expert e lives on rank e // (E / world)
comm = nimble.init(group=ep_group, set_default=False) if args.transport == "nimble" else None
dispatcher = Dispatcher(ep_group, comm=comm, num_experts=E, hidden_size=H, topk=K,
                        x_dtype=torch.float8_e4m3fn if fp8 else torch.bfloat16,
                        scale_dtype=torch.float32 if fp8 else None, scale_cols=H // 128 if fp8 else 0,
                        expert_alignment=128,    # expert_counts rounded up to 128 (DeepGEMM's contiguous layout)
                        mode=args.mode, transport=args.transport, max_tokens=T)

# The router: top-k experts per token and their weights; a masked slot is -1.
torch.manual_seed(rank)
x = torch.randn(T, H, device=device).to(torch.bfloat16)
logits = x.float() @ torch.randn(H, E, device=device, generator=torch.Generator(device).manual_seed(7))
topk_weights, topk_ids = logits.topk(K, dim=1)
topk_weights = topk_weights.softmax(dim=1)
topk_ids[::7, K - 1] = -1                        # every 7th token uses K - 1 experts


def quantize(v):
    """FP8 E4M3 with one float32 scale per 128 elements of a token (the layout DeepGEMM's FP8 GEMM reads)."""
    blocks = v.float().view(v.shape[0], H // 128, 128)
    scale = blocks.abs().amax(-1).clamp(min=1e-4) / 448.0
    return (blocks / scale[..., None]).to(torch.float8_e4m3fn).view(v.shape[0], H), scale


def dequantize(q, scale):
    return (q.float().view(q.shape[0], H // 128, 128) * scale[..., None]).view(q.shape[0], H)


x_in, x_scales = quantize(x) if fp8 else (x, None)
out = dispatcher.dispatch(x_in, x_scales, topk_ids, topk_weights)

# The local experts. Stand-in: expert e multiplies its input by (1 + e / E). A real layer runs its grouped GEMM here;
# out.expert_counts gives each local expert's rows (rounded up to expert_alignment).
hidden = dequantize(out.hidden, out.scales) if fp8 else out.hidden.float()
first = rank * (E // world)                      # this rank's first global expert
gain = torch.where(out.topk_ids >= 0, 1 + (first + out.topk_ids) / E, 0.0)
expert_rows = (hidden * (out.topk_weights * gain).sum(1, keepdim=True)).to(torch.bfloat16)

y = dispatcher.combine(expert_rows, out.handle)  # [T, H] bf16: each token's weighted sum over its experts
torch.cuda.synchronize()

# Check against the same layer computed on this rank alone.
source = dequantize(x_in, x_scales) if fp8 else x.float()
want = source * (topk_weights * torch.where(topk_ids >= 0, 1 + topk_ids / E, 0.0)).sum(1, keepdim=True)
error = ((y.float() - want).norm() / want.norm()).item()
ok = torch.tensor([int(error < 1e-2)], device=device)
dist.all_reduce(ok, op=dist.ReduceOp.MIN)
if rank == 0:
    print(f"07_moe_dispatcher: {'OK' if ok.item() else 'MISMATCH'}; mode {args.mode}, transport {args.transport}, "
          f"{args.dtype}: {out.hidden.shape[0]} rows received, local expert rows {out.expert_counts}, "
          f"relative error {error:.1e}")

if comm is not None:
    comm.destroy()
dist.destroy_process_group()
