#!/bin/bash
#SBATCH --job-name=pinn_gen
#SBATCH --account=education-ae-msc-ae
#SBATCH --partition=compute-p1
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-cpu=2G
#SBATCH --array=0-99
#SBATCH --requeue
#SBATCH --output=/home/resquivelpecas/Thesis/logs/pinngen_%A_%a.out
#
# PINN LABEL GENERATION -- open loop. One dispersed handover state per seed, one two-stage SCvx
# solve, one npz of labels. This is NOT the closed-loop Monte Carlo (mc_array.slurm); there is no
# navigation, no actuator and no re-plan, and the two datasets must not share an output directory.
#
# SEEDS ARE DERIVED, NOT LISTED. seed = SEED_BASE + task*PER + slot + 1, so the grid is reproducible
# from the job id alone and extending the dataset is a matter of raising SEED_BASE or the array
# range. Every seed is recorded inside its own npz.
#
# WHY THE CASES ARE PACKED. Measured association limits: MaxJobs 8 RUNNING, MaxSubmit 100, MaxWall
# 1 day, MaxArraySize 1001, and compute-p1 caps memory at 3996 MB per CPU. A 1-core-per-task array
# would therefore trickle 8 cases at a time. Each task instead takes a whole 48-core compute-p1 node
# and runs 48 cases side by side, one per core, with MC_SLOT telling each which core to pin itself
# to BEFORE it imports gravity -- the TBB pool is sized from the affinity mask at first import and
# honours no thread environment variable. Unpinned they would fan out over the node and thrash, and
# the solve would scatter 0.57 % in fuel run to run.
#
# 100 tasks x 48 = 4800 CASES PER SUBMISSION, the ceiling MaxSubmit imposes. For more, let the first
# array drain and submit again with SEED_BASE moved past it; the script skips seeds it already has,
# so re-submitting the same range is safe and fills gaps rather than repeating work.
#
# IDEMPOTENT. A seed whose npz already exists is skipped, so re-running over the same range fills
# gaps and never repeats work.
#
# EXIT CODES ARE REPORTED PER CASE. `wait` with no arguments returns 0 even when children died;
# waiting on each pid individually turns a silent death into a line that names the seed and signal.
#
# PRECONDITION: nominal.npz must exist -- it carries the handover attitude and throttle every case
# is dispersed around. Make it once, on a login node or a short interactive job:
#     python pinn_case.py --make-nominal --out "Figures&data/$PINN_OUT"
#
# THE DISPERSION lives in pinn_case.py's defaults -- 1000 m, 1 m/s, 2 kg, 3 deg -- and every value
# is recorded inside each case file. EXTRA overrides them; argparse takes the last occurrence, so
# EXTRA always wins.
#
# OPTIONAL ENVIRONMENT:
#     PINN_OUT (default pinn)  PER (default 24)  SEED_BASE (default 0)  EXTRA (extra pinn_case args)
set -u
PINN_OUT=${PINN_OUT:-pinn}
PER=${PER:-48}
SEED_BASE=${SEED_BASE:-0}
EXTRA=${EXTRA:-}
QUEUE=""
LOGS=/home/resquivelpecas/Thesis/logs/pinn
OUT="Figures&data/$PINN_OUT"

module load 2025 >/dev/null 2>&1
module load miniconda3 >/dev/null 2>&1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p "$LOGS"
cd /home/resquivelpecas/Thesis/FullModel/SCvx

if [ ! -f "$OUT/nominal.npz" ]; then
  echo "FATAL: $OUT/nominal.npz missing -- run pinn_case.py --make-nominal --out \"$OUT\" first"
  exit 1
fi

echo "task $SLURM_ARRAY_TASK_ID  node $(hostname)  out=$OUT  per=$PER  seed_base=$SEED_BASE"
echo "  extra='$EXTRA'"

# A WORKER POOL, NOT A FIXED BATCH. Launching 48 cases and waiting for all of them makes a task
# cost the SLOWEST case, not the average: measured on the first run, ~1 % of cases ran to iters=200
# and took ~70 min against a 7 min median, and each one idled 47 cores for an hour -- about 31 % of
# the whole generation wall clock. Pulling from a queue instead means a straggler occupies one core
# while the others keep working, so task time tracks the MEAN case.
declare -a PID
running=0
launch() {
  local seed=$1 slot=$2
  MC_SLOT=$slot ~/envs/eros-pinn/bin/python -u -W ignore pinn_case.py \
      --seed "$seed" --nominal "$OUT/nominal.npz" --out "$OUT" $EXTRA \
      > "$LOGS/${PINN_OUT}_case_$(printf %06d "$seed").log" 2>&1 &
  PID[$slot]=$!
}
slot=0
for i in $(seq 0 $((PER-1))); do
  seed=$((SEED_BASE + SLURM_ARRAY_TASK_ID * PER + i + 1))
  name=$(printf "case_%06d" "$seed")
  [ -f "$OUT/$name.npz" ] && { echo "  skip $name"; continue; }
  QUEUE="$QUEUE $seed"
done
set -- $QUEUE
total=$#
echo "queue: $total seeds, $PER workers"
# fill the pool
for slot in $(seq 0 $((PER-1))); do
  [ $# -eq 0 ] && break
  launch "$1" "$slot"; shift
  sleep 1
done
# as a worker finishes, give it the next seed
fail=0
while [ ${#PID[@]} -gt 0 ]; do
  for slot in "${!PID[@]}"; do
    if ! kill -0 "${PID[$slot]}" 2>/dev/null; then
      wait "${PID[$slot]}" || fail=$((fail+1))
      unset "PID[$slot]"
      if [ $# -gt 0 ]; then launch "$1" "$slot"; shift; fi
    fi
  done
  sleep 5
done
echo "task $SLURM_ARRAY_TASK_ID done: $total seeds, $fail failed"
