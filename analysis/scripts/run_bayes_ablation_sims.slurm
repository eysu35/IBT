#!/bin/bash
#SBATCH --job-name=ibt_bayes_ablation_sims
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --array=0-4
# Study 1 simulations of the Bayesian model variants, one array task per variant; then: python analysis/bayes_ablations.py
REPO=${SLURM_SUBMIT_DIR:-$(pwd)}
cd "$REPO"
source "${IBT_ENV:-$REPO/.venv/bin/activate}"
export MPLBACKEND=Agg
set -euo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p logs
VARIANTS=(full no_hint no_hier_oracle no_hier neither)
python bayesian/ablations/simulate_variant.py --variant "${VARIANTS[$SLURM_ARRAY_TASK_ID]}" --n-envs 100 --n-jobs "$SLURM_CPUS_PER_TASK"
echo DONE
