# Instructed Bandit Task (IBT)

Data, models and code for the Instructed Bandit Task: people play eight kinds of five-armed bandits
with or without a verbal hint, compared to a hierarchical Bayesian agent and LLM agents (Centaur,
CentaurIBT, LLaMA) given the same hints.

## Layout

```
behavioral/
  instructed_bandits/   Study 1 (8 tasks, hint from trial 0): data/raw, data/make_clean.py, analysis/hint_effects.py
  hint_time/            Study 2 (hint at trial 0/4/8/12):     data/raw, data/make_clean.py, analysis/hint_effects.py
  experiment/           task code (Smile web app, project files only)
  anonymize_raw.py      how the raw exports were anonymised (run once; originals not shared)
bayesian/               hierarchical Bayesian agent, simulations, ablations, tests
  fit/fit.py            gamma fit used by the simulations -> fit/data/model_comparison.json
  fit/fit_variants.py   held-out fit of the model and its lesions -> fit/variants/
centaur/                LLM agents: runner + configs, raw sims (data/raw, data/hint_time), LoRA and temperature fits
analysis/               result1..5_*.py -> figures/; tables/ (CSV tables the figures read); bayes_ablations.py;
                        sensitivity_power.py; scripts/ (Slurm)
constants.py            task order, hint texts, Bayesian priors
```

## Data

| | Participants | Games | Trials | Raw | Clean |
|---|---|---|---|---|---|
| Study 1 | 337 | 4,163 | 100,620 | `behavioral/instructed_bandits/data/raw/study1_participants.json` | `behavioral/instructed_bandits/data/behavioral_clean.parquet` |
| Study 2 | 300 | 3,150 | 63,000 | `behavioral/hint_time/data/raw/study2_participants.json` | `behavioral/hint_time/data/behavioral_clean.parquet` |

Raw files are anonymised session exports, one entry per participant: trial data (`studyData`), quiz
answers, build conditions, gender and age. `make_clean.py` flattens them to one row per trial and adds
`regret` (max arm mean minus chosen arm mean; `5 - reward` for linear contextual) and `optimal`.
Analyses use regret as a percentage of the game's arm spread.

Study 1 `batch` 2 (178 participants) are the hint-onset study's first-build sessions, where every hint
was shown at trial 0 (stationary and patch-foraging games only). Study 2 `cohort` is `onset0` (first
build, hint at trial 0, source of the no-hint games) or `timed` (hint at trial 4, 8 or 12).

Model simulations: `bayesian/data/*.parquet`, `centaur/data/*_clean.parquet`.

## Reproduce

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt     # Python 3.10, CPU
make all                            # data -> stats -> tables -> figures 1-5 (analysis/figures), ~10 min
make ablations                      # supplementary ablation figure + report
make test                           # Bayesian agent tests
```

Fits and simulations are shipped; to rerun them (submit from the repo root, edit the account and
environment lines):

| What | Script | Output |
|---|---|---|
| Bayesian gamma fit + Study 1/2 sims | `analysis/scripts/run_bayesian_pipeline.slurm` | `bayesian/fit/data/model_comparison.json`, `bayesian/data/` |
| Bayesian held-out fit + lesions | `bayesian/fit/variants/run_bayes_variants.slurm`, then `run_bayes_fit.slurm` | `bayesian/fit/variants/` |
| Lesion sims | `analysis/scripts/run_bayes_ablation_sims.slurm` | `bayesian/data/ablations/` |
| LLM sims (GPU) | `centaur/scripts/run_simulations.slurm` with `CONFIG_DIR` | `centaur/data/raw/`, `centaur/data/hint_time/` |
| CentaurIBT fine-tune (GPU) | `python centaur/fit/make_data.py`; `centaur/scripts/run_lora_fit.slurm` | `eys8549/centaur_IBT` on Hugging Face |
| LLM temperature fit | `python centaur/fit/temp_fit.py` | `centaur/fit/data/model_comparison.json` |
| Power analysis | `analysis/scripts/run_sensitivity_power.slurm` | `analysis/results/sensitivity_*` |

The Bayesian held-out fit and lesion simulations shipped here predate a fix that removed duplicated Study 1
trial rows; everything else is from the corrected data.
