#!/bin/bash
# vLLM serving DeepSeek-V3 with NIMBLE's MoE dispatcher on every GPU of the allocation (one data-parallel engine per
# GPU, expert parallelism over 4 x nodes GPUs), then client.py against it, then shutdown. Run sbatch from this
# directory:
#
#   sbatch -N 2 vllm_server.sh                       # --all2all-backend nimble, fused (one AllToAllV per direction)
#   MODE=dedup sbatch -N 2 vllm_server.sh            # the dedup mode (two or more nodes; sets NIMBLE_DEDUP=1)
#   BACKEND=nccl_a2a sbatch -N 2 vllm_server.sh      # the same dispatcher over NCCL, for an A/B
#   CLIENT=0 sbatch -N 2 vllm_server.sh              # keep serving on the first node's port until the job ends
#
# Paths and site settings: site_env.sh. Each node's log: logs/vllm_<backend>_<mode>_<job>/node<i>.log. The NIMBLE
# backends turn CUDA graphs off (the patch does it). The batch and length settings are the tested ones, which the
# seeded compile caches match: a first start without them compiles and tunes for up to ~20 minutes.
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --time=00:45:00
set -u
HERE=${EXAMPLES_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}}
source "$HERE/site_env.sh"
BACKEND=${BACKEND:-nimble} MODE=${MODE:-fused} PORT=${PORT:-8000} RPC_PORT=${RPC_PORT:-13345} CLIENT=${CLIENT:-1}
HOSTS=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
EP=$((4 * ${#HOSTS[@]})) HEAD=${HOSTS[0]}
LOG=${LOG_DIR:-$HERE/logs}/vllm_${BACKEND}_${MODE}_$SLURM_JOB_ID
mkdir -p "$LOG"
DEDUP=0; [ "$MODE" = dedup ] && DEDUP=1
echo "vllm_server: $BACKEND ($MODE) on ${HOSTS[*]}, EP $EP; logs $LOG"

# One server per node: node 0 runs the API server and its GPUs' engines, the others their engines, headless.
srun --cpu-bind=none --ntasks-per-node=1 --kill-on-bad-exit=1 -o "$LOG/node%n.log" bash -c "
  source '$HERE/site_env.sh' && vllm_env
  export VLLM_NIMBLE_DISPATCHER=$MODE NIMBLE_DEDUP=$DEDUP
  if [ \$SLURM_NODEID = 0 ]; then EXTRA=''; else EXTRA=\"--headless --data-parallel-start-rank \$((4 * SLURM_NODEID))\"; fi
  exec vllm serve '$MODEL' --trust-remote-code --tensor-parallel-size 1 \
    --data-parallel-size $EP --data-parallel-size-local 4 --data-parallel-address $HEAD --data-parallel-rpc-port $RPC_PORT \
    --enable-expert-parallel --all2all-backend $BACKEND \
    --max-num-batched-tokens 8192 --max-num-seqs 72 --max-model-len 10240 --no-enable-prefix-caching \
    --gpu-memory-utilization 0.85 --host 0.0.0.0 --port $PORT \$EXTRA" &
SERVER=$!

ready() { python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:$PORT/health', timeout=5)" 2>/dev/null; }
until ready; do
  kill -0 $SERVER 2>/dev/null || { echo "vllm_server: the server exited before it was ready"; exit 1; }
  sleep 10
done
echo "vllm_server: ready at http://$HEAD:$PORT"
[ "$CLIENT" = 0 ] && { wait $SERVER; exit $?; }
python3 "$HERE/client.py" --url "http://127.0.0.1:$PORT"
STATUS=$?
kill $SERVER; wait $SERVER 2>/dev/null
exit $STATUS
