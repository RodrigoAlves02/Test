#!/bin/bash
#SBATCH --job-name=mc_smoke
#SBATCH --account=education-ae-msc-ae
#SBATCH --partition=compute-p1
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=1G
#SBATCH --output=/home/resquivelpecas/Thesis/logs/mcsmoke_%j.out
#
# END-TO-END CHECK of the packing before 500 cases are committed: one case from each family, run
# the way mc_array.slurm runs them (several per allocation, each pinned to its own core by
# MC_SLOT). What it proves: the pinning indexes the cpuset correctly, the dispersion branch is
# entered for a zero sigma on one axis and a nonzero one on the other, the OD error is drawn from
# the DISPERSED state, and the output filename carries both sigmas so (1000, 1) and (1000, 0)
# cannot overwrite each other.
set -u
LOGS=/home/resquivelpecas/Thesis/logs/mc
module load 2025 >/dev/null 2>&1
module load miniconda3 >/dev/null 2>&1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p "$LOGS"
cd /home/resquivelpecas/Thesis/FullModel/SCvx
slot=0
for case in "0 0 901" "1000 1 901" "0 1 901" "1000 0 901"; do
  set -- $case
  echo "core slot $slot  pos_sigma=$1  vel_sigma=$2  seed=$3"
  MC_SLOT=$slot ~/envs/eros-pinn/bin/python -u mc_run.py "$1" "$2" "$3" \
      > "$LOGS/smoke_p${1}_v${2}_s${3}.log" 2>&1 &
  slot=$((slot + 1)); sleep 1
done
wait
echo "smoke done"; ls -la 'Figures&data/mc/' 2>&1 | tail -8
