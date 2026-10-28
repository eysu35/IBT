#!/bin/bash
#SBATCH --job-name=ibt_llm_sims
#SBATCH --output=logs/%x_%j.out
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
# LLM simulations for every config in CONFIG_DIR -> RESULTS_DIR (GPU; HF token in ~/.hf_token). Example:
#   sbatch --export=ALL,CONFIG_DIR=centaur/config/run_all_centaur,RESULTS_DIR=centaur/data/raw centaur/scripts/run_simulations.slurm
REPO=${SLURM_SUBMIT_DIR:-$(pwd)}
cd "$REPO"
source "${IBT_ENV:-$REPO/.venv/bin/activate}"
export MPLBACKEND=Agg
set -euo pipefail
export HUGGING_FACE_HUB_TOKEN=$(cat ~/.hf_token)
export HF_HUB_ENABLE_HF_TRANSFER=0
mkdir -p logs "$RESULTS_DIR"
cd centaur
for CONFIG_FILE in "$REPO/$CONFIG_DIR"/*.json; do
  echo "config: $CONFIG_FILE"
  python run.py --config-file "$CONFIG_FILE" --results-dir "$REPO/$RESULTS_DIR/"
done
echo DONE
