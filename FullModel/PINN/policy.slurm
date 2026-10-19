#!/bin/bash
#SBATCH --job-name=pinn_pol
#SBATCH --account=education-ae-msc-ae
#SBATCH --partition=compute-p1
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=3G
#SBATCH --output=/home/resquivelpecas/Thesis/logs/pinn/policy_%j.out
#
# THE DIRECT GUIDANCE POLICY. Far cheaper than the trajectory surrogate: the Bellman expansion adds
# nothing here, because every sub-trajectory's first node is already a node of its parent, so the
# dataset is ~379k (state, control) pairs rather than 19M (state, tau) pairs.
set -u
module load 2025 >/dev/null 2>&1
module load miniconda3 >/dev/null 2>&1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
cd /home/resquivelpecas/Thesis/FullModel/PINN
echo "host $(hostname)  cpus ${SLURM_CPUS_PER_TASK:-?}  args: ${ARGS:-}"
~/envs/eros-pinn/bin/python -u -W ignore policy.py ${ARGS:-}
