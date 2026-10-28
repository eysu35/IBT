#!/bin/bash
#SBATCH --job-name=ibt_power
#SBATCH --output=logs/%x_%j.out
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
# Power / minimum-detectable-effect simulation (analysis/sensitivity_power.py); needs `make stats` first.
REPO=${SLURM_SUBMIT_DIR:-$(pwd)}
cd "$REPO"
source "${IBT_ENV:-$REPO/.venv/bin/activate}"
export MPLBACKEND=Agg
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p logs
python analysis/sensitivity_power.py
echo DONE
