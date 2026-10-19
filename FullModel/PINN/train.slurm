#!/bin/bash
#SBATCH --job-name=pinn_tr
#SBATCH --account=education-ae-msc-ae
#SBATCH --partition=compute-p1
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=3G
#SBATCH --output=/home/resquivelpecas/Thesis/logs/pinn/train_%j.out
#
# ONE TRAINING RUN. Everything that shaped it lands in PINN/runs/<run_id>/, which is the point of
# the folder: a result that cannot be traced back to its config is not a result.
#
# CPU, NOT THE A100. The network is ~335k parameters and GPUs bill at 300x a CPU here, so a CPU run
# of under an hour is far cheaper than a five-minute GPU one for the same answer. Revisit when the
# sweep grows or the model does.
#
# TIME AND CPUS cover Adam PLUS a real L-BFGS polish now that the polish runs on a fixed
# batch and actually iterates. The n1000 run was 3 h 17 m of Adam on 8 cpus with a polish
# that stopped after one closure; the separate fixed-batch polish was 21 m on 16.
#
# ARGS goes straight to train.py, and every train.py flag is a recorded Config field, so nothing can
# be set here that does not appear in the run's config.json.
set -u
module load 2025 >/dev/null 2>&1
module load miniconda3 >/dev/null 2>&1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
cd /home/resquivelpecas/Thesis/FullModel/PINN
echo "host $(hostname)  cpus ${SLURM_CPUS_PER_TASK:-?}  args: ${ARGS:-}"
~/envs/eros-pinn/bin/python -u -W ignore train.py ${ARGS:-}
