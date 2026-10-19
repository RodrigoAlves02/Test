#!/bin/bash
#SBATCH --job-name=tol_test
#SBATCH --account=education-ae-msc-ae
#SBATCH --partition=compute-p1,compute-p2
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=3G
#SBATCH --output=/home/resquivelpecas/Thesis/logs/pinn/toltest_%j.out
#
# ARE THE GENERATION TOLERANCES TIGHTER THAN THEY NEED TO BE?
#
# One process per seed, each solving that seed at all three levels in turn, so a level's cost is
# measured against its own answer on the same case rather than across cases. 10 of the seeds are
# ones the tight level REJECTED -- those are where the payoff would be, since a stalled case costs
# ~70 min and drags a whole 48-core task with it.
set -u
module load 2025 >/dev/null 2>&1
module load miniconda3 >/dev/null 2>&1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd /home/resquivelpecas/Thesis/FullModel/SCvx
OUT="Figures&data/tol_test"; mkdir -p "$OUT"
NOM="Figures&data/pinn/nominal.npz"
REJ="139 499 976 1500 2021 2607 3061 3398 3774 4291"
OKS="50 300 700 1100 1600 2100 2500 2900 3300 3700 4000 4300 4500 4700"
slot=0
for s in $REJ $OKS; do
  MC_SLOT=$slot ~/envs/eros-pinn/bin/python -u -W ignore tol_test.py "$NOM" "$OUT/seed_$s.json" "$s" \
      > "$OUT/seed_$s.log" 2>&1 &
  slot=$((slot+1)); sleep 1
done
echo "launched $slot seeds x 3 levels"
fail=0
for p in $(jobs -p); do wait "$p" || fail=$((fail+1)); done
echo "done, $fail failed"
