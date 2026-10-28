#!/bin/bash
#SBATCH --job-name=ibt_bayes_precompute
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --array=0-2
# Held-out Bayesian fit, stage 1: cache the per-trial predictions per hypothesis space (array: full, stationary, oracle).
REPO=${SLURM_SUBMIT_DIR:-$(pwd)}
cd "$REPO"
source "${IBT_ENV:-$REPO/.venv/bin/activate}"
export MPLBACKEND=Agg
set -euo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p logs
SPACES=(full stationary oracle)
python bayesian/fit/fit_variants.py precompute --space "${SPACES[$SLURM_ARRAY_TASK_ID]}" --n-jobs "$SLURM_CPUS_PER_TASK"
echo DONE
