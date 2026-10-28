#!/bin/bash
#SBATCH --job-name=ibt_bayes_fit
#SBATCH --output=logs/%x_%j.out
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
# Held-out Bayesian fit, stage 2: 5-fold CV of the full model and its variants on the cached predictions.
REPO=${SLURM_SUBMIT_DIR:-$(pwd)}
cd "$REPO"
source "${IBT_ENV:-$REPO/.venv/bin/activate}"
export MPLBACKEND=Agg
set -euo pipefail
export OMP_NUM_THREADS=4
mkdir -p logs
python bayesian/fit/fit_variants.py fit
echo DONE
