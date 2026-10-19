#!/bin/bash
#SBATCH --job-name=mc_disp
#SBATCH --account=education-ae-msc-ae
#SBATCH --partition=compute-p1
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=2G
#SBATCH --array=0-20
#SBATCH --requeue
#SBATCH --output=/home/resquivelpecas/Thesis/logs/mcpack_%A_%a.out
#
# CLOSED-LOOP MONTE CARLO -- the grid is $CASES (default mc_cases.txt, written by mc_cases.py,
# which documents the design: 10 handover-dispersion configurations x 50 seeds).
#
# WHY THE CASES ARE PACKED.  The association allows 8 RUNNING JOBS and 100 SUBMITTED, so a 500-task
# array is not submittable and a 1-core-per-task array would trickle 8 cases at a time for ~28 h.
# Each task takes $PER cores of a node and runs that many cases side by side, one per core.
#
# WHY MC_SLOT.  polyhedral_gravity sizes its TBB pool ONCE, from the affinity mask in force when
# `gravity` is first imported, and honours no thread environment variable.  MC_SLOT tells mc_run.py
# which core of the allocation to pin itself to, which it does before that import -- indexing into
# the cpuset it actually got, because Slurm does not promise the core ids are 0..n-1.  Without the
# pin all the processes would fan out over the whole node and thrash.
#
# IDEMPOTENT.  A case whose json already exists is skipped, so re-running this over the same grid
# fills gaps and never repeats work.  That is the recovery path: regenerate the case list, submit,
# done.
#
# EXIT CODES ARE REPORTED PER CASE.  Job 865910 lost 27 cases inside tasks that Slurm called
# COMPLETED: the children died mid-descent, `wait` with no arguments then returned 0, and the
# failures were invisible.  Waiting on each pid individually turns a silent death into a line that
# names the case and the signal.
set -u
CASES=${CASES:-/home/resquivelpecas/Thesis/FullModel/SCvx/mc_cases.txt}
LOGS=/home/resquivelpecas/Thesis/logs/mc
PER=${PER:-24}
# MC_OUT keeps datasets apart: a run with different guidance gravity is a different experiment and
# must not land in the same folder.  EXTRA is appended to every case's closed_loop command line,
# and argparse takes the LAST occurrence of an option, so EXTRA can override mc_run.py's defaults.
MC_OUT=${MC_OUT:-mc}
EXTRA=${EXTRA:-}
export MC_OUT
OUT="Figures&data/$MC_OUT"

module load 2025 >/dev/null 2>&1
module load miniconda3 >/dev/null 2>&1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p "$LOGS"
cd /home/resquivelpecas/Thesis/FullModel/SCvx

START=$((SLURM_ARRAY_TASK_ID * PER + 1))
END=$((START + PER - 1))
echo "task $SLURM_ARRAY_TASK_ID  node $(hostname)  cases $START..$END of $(wc -l < "$CASES")"
echo "  cases=$CASES  out=$OUT  extra='$EXTRA'"

declare -a PID NAME
slot=0
while read -r L V S; do
  [ -z "${L:-}" ] && continue
  name="p${L}_v${V}_s${S}"
  if [ -f "$OUT/closed_loop_mc_${name}.json" ]; then
    echo "  skip (already have it)  $name"; continue
  fi
  echo "  core slot $slot  pos_sigma=$L  vel_sigma=$V  seed=$S"
  MC_SLOT=$slot ~/envs/eros-pinn/bin/python -u mc_run.py "$L" "$V" "$S" $EXTRA \
      > "$LOGS/${MC_OUT}_${name}.log" 2>&1 &
  PID[$slot]=$!; NAME[$slot]=$name
  slot=$((slot + 1))
  sleep 2                     # stagger the torch/mesh imports off the shared filesystem
done < <(sed -n "${START},${END}p" "$CASES")

echo "launched $slot cases, waiting"
fail=0
for i in $(seq 0 $((slot - 1))); do
  if wait "${PID[$i]}"; then :; else
    rc=$?
    # 128+n is death by signal n: 137 = SIGKILL (oom killer or scancel), 139 = SIGSEGV.
    echo "CASE FAILED: ${NAME[$i]} exit=$rc$([ "$rc" -gt 128 ] && echo " (signal $((rc - 128)))")"
    fail=$((fail + 1))
  fi
done
echo "task $SLURM_ARRAY_TASK_ID done: $slot launched, $fail failed"
