"""SGLang's offline engine, sglang.Engine (the Python API, no HTTP server), with NIMBLE's MoE dispatcher.

Every node runs this script (sglang_offline.sh starts it there). The engine spans all of them: data-parallel attention
and expert parallelism over 4 x nodes GPUs. sglang.Engine() returns on node 0, which generates; on every other node
it runs that node's ranks until the job stops them.
"""
import os

import sglang as sgl
from transformers import AutoTokenizer


def main():
    nodes, node = int(os.environ.get("SLURM_NNODES", "1")), int(os.environ.get("SLURM_NODEID", "0"))
    ep = 4 * nodes
    engine = sgl.Engine(model_path=os.environ["MODEL"], trust_remote_code=True,
                        tp_size=ep, ep_size=ep, dp_size=ep, enable_dp_attention=True,
                        moe_a2a_backend=os.environ.get("BACKEND", "nimble"),   # or "nccl_a2a": the same over NCCL
                        moe_runner_backend="deep_gemm", disable_cuda_graph=True, mem_fraction_static=0.70,
                        nnodes=nodes, node_rank=node, dist_init_addr=os.environ["DIST_INIT_ADDR"])

    # Node 0 only from here.
    tokenizer = AutoTokenizer.from_pretrained(os.environ["MODEL"], trust_remote_code=True)
    pairs = [(12 + i, 23 + 3 * i) for i in range(32)]
    prompts = [tokenizer.apply_chat_template(
        [{"role": "user", "content": f"What is {a} * {b}? Reply with the number only."}],
        tokenize=False, add_generation_prompt=True) for a, b in pairs]
    outputs = engine.generate(prompts, {"temperature": 0, "max_new_tokens": 16})
    correct = sum(str(a * b) in out["text"].replace(",", "") for (a, b), out in zip(pairs, outputs))
    for (a, b), out in list(zip(pairs, outputs))[:3]:
        print(f"sglang_offline: {a} * {b} -> {out['text'].strip()!r}")
    print(f"sglang_offline: {'OK' if correct >= 0.9 * len(pairs) else 'FAILED'}; {correct}/{len(pairs)} correct",
          flush=True)
    engine.shutdown()


if __name__ == "__main__":
    main()
