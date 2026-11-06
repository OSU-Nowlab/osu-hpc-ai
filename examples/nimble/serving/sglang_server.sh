#!/bin/bash
# SGLang serving DeepSeek-V3 with NIMBLE's MoE dispatcher on every GPU of the allocation (data-parallel attention and
# expert parallelism over 4 x nodes GPUs), then client.py against it, then shutdown. Run sbatch from this directory:
#
#   sbatch -N 2 sglang_server.sh                     # --moe-a2a-backend nimble, fused (one AllToAllV per direction)
#   MODE=dedup sbatch -N 2 sglang_server.sh          # the dedup mode (two or more nodes; sets NIMBLE_DEDUP=1)
#   BACKEND=nccl_a2a sbatch -N 2 sglang_server.sh    # the same dispatcher over NCCL, for comparison
#   CLIENT=0 sbatch -N 2 sglang_server.sh            # keep serving on the first node's port until the job ends
#
# Paths and site settings: site_env.sh. Each node's log: logs/sglang_<backend>_<mode>_<job>/node<i>.log.
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --time=00:45:00
set -u
HERE=${EXAMPLES_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}}
source "$HERE/site_env.sh"
site_check sglang || exit 1
BACKEND=${BACKEND:-nimble} MODE=${MODE:-fused} PORT=${PORT:-30000} DIST_PORT=${DIST_PORT:-30500} CLIENT=${CLIENT:-1}
HOSTS=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
NODES=${#HOSTS[@]} EP=$((4 * ${#HOSTS[@]})) HEAD=${HOSTS[0]}
LOG=${LOG_DIR:-$HERE/logs}/sglang_${BACKEND}_${MODE}_$SLURM_JOB_ID
mkdir -p "$LOG"
DEDUP=0; [ "$MODE" = dedup ] && DEDUP=1
LARGE=""; [ "$NODES" -ge 8 ] && LARGE="--moe-dense-tp-size 1 --dist-timeout 3600"
echo "sglang_server: $BACKEND ($MODE) on ${HOSTS[*]}, EP $EP; logs $LOG"

# One server process per node: node 0 serves HTTP, the others join it at --dist-init-addr.
srun --cpu-bind=none --ntasks-per-node=1 --kill-on-bad-exit=1 -o "$LOG/node%n.log" bash -c "
  source '$HERE/site_env.sh' && sglang_env
  export SGLANG_NIMBLE_DISPATCHER=$MODE NIMBLE_DEDUP=$DEDUP
  exec python -m sglang.launch_server --model-path '$MODEL' --trust-remote-code \
    --tp-size $EP --ep-size $EP --dp-size $EP --enable-dp-attention \
    --moe-a2a-backend $BACKEND --moe-runner-backend deep_gemm --disable-cuda-graph --mem-fraction-static 0.70 \
    --nnodes $NODES --node-rank \$SLURM_NODEID --dist-init-addr $HEAD:$DIST_PORT --host 0.0.0.0 --port $PORT $LARGE" &
SERVER=$!

ready() { python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:$PORT/health', timeout=5)" 2>/dev/null; }
until ready; do
  kill -0 $SERVER 2>/dev/null || { echo "sglang_server: the server exited before it was ready"; exit 1; }
  sleep 10
done
echo "sglang_server: ready at http://$HEAD:$PORT"
[ "$CLIENT" = 0 ] && { wait $SERVER; exit $?; }
python3 "$HERE/client.py" --url "http://127.0.0.1:$PORT"
STATUS=$?
kill $SERVER; wait $SERVER 2>/dev/null
exit $STATUS
