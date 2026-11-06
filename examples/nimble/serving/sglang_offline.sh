#!/bin/bash
# Run sglang_offline.py (sglang.Engine with NIMBLE's MoE dispatcher) on every node of the allocation. Node 0 generates
# and checks; when it is done, srun stops the other nodes (--wait). Run sbatch from this directory:
#
#   sbatch -N 2 sglang_offline.sh                    # NIMBLE, fused
#   MODE=dedup sbatch -N 2 sglang_offline.sh         # the dedup mode (two or more nodes; sets NIMBLE_DEDUP=1)
#
# Paths and site settings: site_env.sh. Each node's log: logs/sglang_offline_<mode>_<job>/node<i>.log.
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --time=00:45:00
set -u
HERE=${EXAMPLES_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}}
source "$HERE/site_env.sh"
site_check sglang || exit 1
MODE=${MODE:-fused} DIST_PORT=${DIST_PORT:-30600}
HEAD=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
LOG=${LOG_DIR:-$HERE/logs}/sglang_offline_${MODE}_$SLURM_JOB_ID
mkdir -p "$LOG"
DEDUP=0; [ "$MODE" = dedup ] && DEDUP=1
srun --cpu-bind=none --ntasks-per-node=1 --kill-on-bad-exit=1 --wait=120 -o "$LOG/node%n.log" bash -c "
  source '$HERE/site_env.sh' && sglang_env
  export SGLANG_NIMBLE_DISPATCHER=$MODE NIMBLE_DEDUP=$DEDUP DIST_INIT_ADDR=$HEAD:$DIST_PORT BACKEND=${BACKEND:-nimble}
  exec python '$HERE/sglang_offline.py'"
grep -h "sglang_offline:" "$LOG/node0.log"
grep -q "sglang_offline: OK" "$LOG/node0.log"
