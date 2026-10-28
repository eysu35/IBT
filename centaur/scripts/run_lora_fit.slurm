#!/bin/bash
#SBATCH --job-name=ibt_lora
#SBATCH --output=logs/%x_%j.out
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
# LoRA fine-tuning of Centaur on the Study 1 training split (CentaurIBT); run centaur/fit/make_data.py first.
REPO=${SLURM_SUBMIT_DIR:-$(pwd)}
cd "$REPO"
source "${IBT_ENV:-$REPO/.venv/bin/activate}"
export MPLBACKEND=Agg
set -euo pipefail
export HUGGING_FACE_HUB_TOKEN=$(cat ~/.hf_token)
export HF_HUB_ENABLE_HF_TRANSFER=0
mkdir -p logs
python centaur/fit/lora_fit.py --data-dir centaur/fit/data --label-smoothing 0.05 --wandb-project centaur-lora --wandb-name lora-smooth-0.05
echo DONE
