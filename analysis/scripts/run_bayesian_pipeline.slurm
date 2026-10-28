#!/bin/bash
#SBATCH --job-name=ibt_bayes_sims
#SBATCH --output=logs/%x_%j.out
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
# Bayesian agent: fit gamma to Study 1 choices, simulate Study 1 and Study 2, redraw the figures.
REPO=${SLURM_SUBMIT_DIR:-$(pwd)}
cd "$REPO"
source "${IBT_ENV:-$REPO/.venv/bin/activate}"
export MPLBACKEND=Agg
set -euo pipefail
mkdir -p logs
echo '== [1/4] fit (bayesian/fit/data/model_comparison.json) =='
python bayesian/fit/fit.py
echo '== [2/4] Study 1 simulations (bayesian/data/bayesian_sim_trials.parquet) =='
rm -f bayesian/data/bayesian_sim_trials.parquet
python bayesian/simulate_bayesian.py --n-envs 100 --resim
echo '== [3/4] Study 2 simulations (bayesian/data/bayesian_timing_trials.parquet) =='
rm -f bayesian/data/bayesian_timing_trials.parquet
python bayesian/simulate_bayesian_timing.py
echo '== [4/4] statistics, tables, figures =='
make stats tables figures
echo DONE
