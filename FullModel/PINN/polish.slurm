#!/bin/bash
#SBATCH --job-name=pinn_pol
#SBATCH --account=education-ae-msc-ae
#SBATCH --partition=compute-p1,compute-p2
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=3G
#SBATCH --output=/home/resquivelpecas/Thesis/logs/pinn/polish_%j.out
#
# L-BFGS polish of an Adam checkpoint, on a FIXED batch. See polish.py for why the batch must be
# fixed -- the minibatched version in train.py stopped after one iteration and made things worse.
set -u
module load 2025 >/dev/null 2>&1
module load miniconda3 >/dev/null 2>&1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
cd /home/resquivelpecas/Thesis/FullModel/PINN
echo "host $(hostname)  args: ${ARGS:-}"
~/envs/eros-pinn/bin/python -u -W ignore polish.py ${ARGS:-}
