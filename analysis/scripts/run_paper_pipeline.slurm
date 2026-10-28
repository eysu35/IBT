#!/bin/bash
#SBATCH --job-name=ibt_paper
#SBATCH --output=logs/%x_%j.out
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
# Human and model statistics, tables and figures 1-5 from the shipped data (same as `make all`).
REPO=${SLURM_SUBMIT_DIR:-$(pwd)}
cd "$REPO"
source "${IBT_ENV:-$REPO/.venv/bin/activate}"
export MPLBACKEND=Agg
set -euo pipefail
mkdir -p logs
make data stats tables figures
echo DONE
