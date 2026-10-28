# Bayesian model ablations (supplement)

Held-out fit: 5-fold CV over participants (seed 42), K = 500 posterior samples, T and γ fitted per fold on the training folds, log-likelihood of every held-out trial. Choices are scored in rank space (as in the simulations; bayesian/fit/fit.py, whose gamma the simulations use, scores displayed positions). Simulations: 100 agents per task x hint condition, γ = mean fitted γ over folds, T = None (normative Thompson policy, as in the paper); per-hint effects by OLS/HC3 vs the variant's own no-hint games, in % of arm spread.

## Summary

Held-out LL / trial and accuracy are computed per task and averaged over the eight tasks with equal weight (stationary and patch foraging have about twice as many participants as the other tasks, so a plain per-trial mean over-weights them; the trial-weighted numbers, the weighting of Fig. 5a and the main text, are given below). Δ LL is per participant: the participant's LL / trial per task, minus the full model's, averaged over the tasks they played (nats per trial).

| Variant | LL / trial (nats) ± 1.96 SE | Accuracy | Δ LL / trial vs full, per participant [95% CI] | paired t p | r with human per-hint effects | MAE vs human (% spread) |
|---|---|---|---|---|---|---|
| Fixed-type + hinted | -1.3059 ± 0.0403 | 0.590 | +0.0110 [+0.0081, +0.0139] | 1.4e-12 | 0.79 | 3.9 |
| Hierarchical hinted | -1.3149 ± 0.0344 | 0.593 | — | — | 0.73 | 4.1 |
| Non-hierarchical hinted | -1.3457 ± 0.0364 | 0.524 | -0.0072 [-0.0153, +0.0010] | 0.086 | 0.75 | 4.1 |
| Hierarchical no-hint | -1.4025 ± 0.0288 | 0.577 | -0.1086 [-0.1171, -0.1001] | 6.4e-79 | 0.09 | 6.8 |
| Non-hierarchical no-hint | -1.4249 ± 0.0277 | 0.505 | -0.1012 [-0.1121, -0.0903] | 6.2e-52 | 0.13 | 6.8 |
| Chance | -1.6094 | 0.200 | | | | |

Negative Δ = the ablation predicts choices worse than the full model. The Centaur and LLaMA reference lines in the figure are their overall (trial-weighted) held-out LL from the main model comparison, since no per-task likelihoods are stored for them.

Trial-weighted (plain mean over all held-out trials; the weighting used in Fig. 5a and the main-text model comparison; Δ = summed LL over the participant's held-out trials, nats): Fixed-type + hinted -1.2894 ± 0.0279 (Δ +3.7 [+2.8, +4.7], p = 9.3e-14); Hierarchical hinted -1.2995 ± 0.0273; Non-hierarchical hinted -1.3058 ± 0.0272 (Δ -2.3 [-5.7, +1.1], p = 0.19); Hierarchical no-hint -1.3898 ± 0.0253 (Δ -33.3 [-36.3, -30.2], p = 9e-65); Non-hierarchical no-hint -1.3868 ± 0.0228 (Δ -32.2 [-36.7, -27.6], p = 9.6e-35).

## Discrepancy with the paper's existing Bayesian numbers

- Paper (repo fit, display-position space): Bayesian + hints -1.4065 ± 0.0261, accuracy 0.529; Bayesian without hints -1.3902, accuracy 0.576.
- Ablation code (rank space, same trial-weighted mean): full -1.2995 ± 0.0273, accuracy 0.586; hierarchical no-hint -1.3898, accuracy 0.576.
- The difference is the choice space only (rank space here, displayed button positions in bayesian/fit/fit.py).

## Fitted parameters per fold

| Variant | fold | T | γ | test LL / trial | at bound |
|---|---|---|---|---|---|
| Fixed-type + hinted | 0 | 13.60 | 0.10 | -1.2724 | γ |
| Fixed-type + hinted | 1 | 13.48 | 0.10 | -1.3040 | γ |
| Fixed-type + hinted | 2 | 13.38 | 0.10 | -1.3256 | γ |
| Fixed-type + hinted | 3 | 13.48 | 0.10 | -1.3009 | γ |
| Fixed-type + hinted | 4 | 13.77 | 0.10 | -1.2436 | γ |
| Hierarchical hinted | 0 | 13.77 | 0.10 | -1.2810 | γ |
| Hierarchical hinted | 1 | 13.57 | 0.10 | -1.3173 | γ |
| Hierarchical hinted | 2 | 13.47 | 0.10 | -1.3371 | γ |
| Hierarchical hinted | 3 | 13.63 | 0.10 | -1.3043 | γ |
| Hierarchical hinted | 4 | 13.88 | 0.10 | -1.2573 | γ |
| Non-hierarchical hinted | 0 | 14.30 | 0.10 | -1.2800 | γ |
| Non-hierarchical hinted | 1 | 14.11 | 0.10 | -1.3119 | γ |
| Non-hierarchical hinted | 2 | 13.95 | 0.10 | -1.3447 | γ |
| Non-hierarchical hinted | 3 | 14.05 | 0.10 | -1.3231 | γ |
| Non-hierarchical hinted | 4 | 14.33 | 0.10 | -1.2694 | γ |
| Hierarchical no-hint | 0 | 14.42 |  | -1.3728 |  |
| Hierarchical no-hint | 1 | 14.16 |  | -1.4087 |  |
| Hierarchical no-hint | 2 | 14.07 |  | -1.4155 |  |
| Hierarchical no-hint | 3 | 14.16 |  | -1.4041 |  |
| Hierarchical no-hint | 4 | 14.55 |  | -1.3470 |  |
| Non-hierarchical no-hint | 0 | 15.70 |  | -1.3638 |  |
| Non-hierarchical no-hint | 1 | 15.46 |  | -1.3934 |  |
| Non-hierarchical no-hint | 2 | 15.29 |  | -1.4147 |  |
| Non-hierarchical no-hint | 3 | 15.32 |  | -1.4066 |  |
| Non-hierarchical no-hint | 4 | 15.74 |  | -1.3557 |  |

γ sits at its lower bound (0.1) in every fold of every hinted variant: it scales only the soft, history-based hints (hard hints have log-weights 0 / -inf, unchanged by γ), and the fit wants those as weak as allowed. Refitting Fixed-type + hinted, Hierarchical hinted, Non-hierarchical hinted with the lower bound widened to 0.001 (bayesian/fit/variants/gamma_wide/, run_gamma_wide.py; same caches, folds, seed): γ = 0.001 in every fold, T unchanged to two decimals, held-out LL / trial changes by at most 0.0002 nats, accuracy unchanged. The bound is immaterial.

## Held-out fit by hint vs no-hint games

| Variant | hint games LL | hint games acc | no-hint games LL | no-hint games acc |
|---|---|---|---|---|
| Fixed-type + hinted | -1.2797 ± 0.0329 | 0.586 | -1.3315 ± 0.0308 | 0.577 |
| Hierarchical hinted | -1.2895 ± 0.0314 | 0.586 | -1.3433 ± 0.0301 | 0.584 |
| Non-hierarchical hinted | -1.3057 ± 0.0315 | 0.545 | -1.3063 ± 0.0285 | 0.558 |
| Hierarchical no-hint | -1.4001 ± 0.0277 | 0.574 | -1.3444 ± 0.0288 | 0.584 |
| Non-hierarchical no-hint | -1.4044 ± 0.0255 | 0.530 | -1.3102 ± 0.0256 | 0.558 |

## Fine-grained behaviour: r with people over the 48 conditions (Fig. 3 metrics, bootstrap 95% CI)

| Variant | Switch rate | Choice entropy | Exploitation onset |
|---|---|---|---|
| Fixed-type + hinted | 0.74 [0.57, 0.86] | 0.83 [0.69, 0.91] | 0.80 [0.68, 0.89] |
| Hierarchical hinted | 0.75 [0.59, 0.86] | 0.85 [0.75, 0.93] | 0.82 [0.72, 0.91] |
| Non-hierarchical hinted | 0.85 [0.75, 0.91] | 0.81 [0.67, 0.90] | 0.94 [0.90, 0.97] |
| Hierarchical no-hint | 0.66 [0.47, 0.80] | 0.72 [0.56, 0.85] | 0.77 [0.63, 0.87] |
| Non-hierarchical no-hint | 0.80 [0.67, 0.87] | 0.60 [0.39, 0.77] | 0.91 [0.84, 0.95] |
| Paper table (bayesian vs people) | 0.76 [0.62, 0.87] | 0.85 [0.76, 0.93] | 0.84 [0.76, 0.92] |
| Paper table (centaur vs people) | 0.90 [0.86, 0.95] | 0.95 [0.92, 0.97] | 0.95 [0.92, 0.97] |

## Held-out fit by hint category (LL / trial; V = value, P = procedure, S = structure hints)

| Games | Fixed-type + hinted | Hierarchical hinted | Non-hierarchical hinted | Hierarchical no-hint | Non-hierarchical no-hint |
|---|---|---|---|---|---|
| V hints | -0.965 | -0.998 | -1.040 | -1.401 | -1.431 |
| P hints | -1.373 | -1.376 | -1.393 | -1.383 | -1.388 |
| S hints | -1.408 | -1.408 | -1.405 | -1.414 | -1.401 |
| no hint | -1.331 | -1.343 | -1.306 | -1.344 | -1.310 |

## Held-out fit by task (LL / trial)

| Task | Fixed-type + hinted | Hierarchical hinted | Non-hierarchical hinted | Hierarchical no-hint | Non-hierarchical no-hint |
|---|---|---|---|---|---|
| stationary | -0.892 | -0.902 | -0.908 | -1.146 | -1.135 |
| drifting | -1.055 | -1.130 | -1.065 | -1.287 | -1.218 |
| time_delayed | -1.396 | -1.345 | -1.382 | -1.317 | -1.418 |
| patch_foraging | -1.476 | -1.488 | -1.467 | -1.474 | -1.453 |
| two_context | -1.387 | -1.403 | -1.410 | -1.503 | -1.506 |
| three_context | -1.399 | -1.410 | -1.491 | -1.506 | -1.548 |
| bernoulli | -1.425 | -1.427 | -1.317 | -1.536 | -1.413 |
| linear_contextual | -1.436 | -1.431 | -1.726 | -1.466 | -1.710 |

## Where the hierarchy matters: per-task paired differences

Per participant, held-out LL summed over their trials in the task, first model minus second; mean over participants [95% CI], paired t p. Positive = the first model fits that task's choices better. The hierarchy (all eight task types in the hypothesis space vs stationary only) helps where behaviour depends on structure a stationary learner cannot represent (contexts, delayed value) and costs on the tasks people treat as simple bandits (bernoulli, drifting, patch foraging), where its type uncertainty makes it explore more than people do. The near-zero overall difference is the cancellation of these two reliable effects.

**Hierarchical hinted minus Non-hierarchical hinted**

| Task | all games: Δ [95% CI] | p | hint games: Δ | p | no-hint games: Δ | p |
|---|---|---|---|---|---|---|
| stationary | +0.6 [+0.2, +1.0] | 0.0045 | +1.0 | 4.3e-14 | -0.6 | 0.0048 |
| drifting | -2.8 [-3.9, -1.7] | 2.3e-06 | -2.6 | 2.1e-05 | -1.4 | 0.022 |
| time_delayed | +1.6 [+0.8, +2.3] | 0.00021 | +0.8 | 0.049 | +2.4 | 6.1e-08 |
| patch_foraging | -2.8 [-3.2, -2.3] | 4.2e-28 | -2.3 | 1e-21 | -0.7 | 3.9e-05 |
| two_context | +0.6 [-3.0, +4.2] | 0.76 | +2.1 | 0.21 | -4.1 | 0.1 |
| three_context | +7.2 [+3.1, +11.2] | 0.00067 | +7.7 | 9.4e-05 | -0.5 | 0.84 |
| bernoulli | -8.8 [-11.2, -6.5] | 1.5e-11 | -6.5 | 2.7e-08 | -10.6 | 9.4e-07 |
| linear_contextual | +12.9 [+8.5, +17.2] | 4.5e-08 | +13.3 | 5.7e-09 | +0.1 | 0.95 |

**Hierarchical no-hint minus Non-hierarchical no-hint**

| Task | all games: Δ [95% CI] | p | hint games: Δ | p | no-hint games: Δ | p |
|---|---|---|---|---|---|---|
| stationary | -1.0 [-1.8, -0.3] | 0.0063 | -0.9 | 0.0022 | -0.2 | 0.32 |
| drifting | -3.0 [-3.7, -2.3] | 1.4e-14 | -2.9 | 4.6e-16 | -1.2 | 0.026 |
| time_delayed | +4.3 [+3.5, +5.1] | 1.7e-19 | +3.8 | 8.5e-17 | +2.0 | 2.2e-07 |
| patch_foraging | -2.8 [-3.5, -2.1] | 5.5e-14 | -2.2 | 5.1e-13 | -0.8 | 3.4e-06 |
| two_context | +0.2 [-3.0, +3.4] | 0.91 | +1.8 | 0.25 | -4.2 | 0.079 |
| three_context | +3.7 [+0.1, +7.4] | 0.048 | +4.2 | 0.015 | -1.1 | 0.66 |
| bernoulli | -9.9 [-12.0, -7.7] | 4.7e-15 | -7.8 | 4.5e-12 | -10.1 | 5.7e-07 |
| linear_contextual | +10.7 [+6.9, +14.5] | 1.5e-07 | +11.0 | 1.9e-08 | +0.4 | 0.86 |

**Fixed-type + hinted minus Non-hierarchical hinted**

| Task | all games: Δ [95% CI] | p | hint games: Δ | p | no-hint games: Δ | p |
|---|---|---|---|---|---|---|
| stationary | +1.5 [+1.3, +1.6] | 3.8e-65 | +1.3 | 2e-67 | +0.3 | 3.3e-24 |
| drifting | +0.4 [+0.2, +0.6] | 0.00021 | +0.3 | 0.001 | +0.3 | 0.054 |
| time_delayed | -0.6 [-0.9, -0.3] | 0.00023 | -0.5 | 0.00056 | -0.3 | 0.17 |
| patch_foraging | -1.1 [-1.4, -0.9] | 1.7e-14 | -1.0 | 9.9e-15 | -0.2 | 0.015 |
| two_context | +1.9 [-1.8, +5.6] | 0.33 | +3.6 | 0.039 | -4.2 | 0.099 |
| three_context | +8.2 [+4.0, +12.4] | 0.0002 | +8.7 | 2.1e-05 | -0.5 | 0.84 |
| bernoulli | -8.7 [-11.0, -6.3] | 3.3e-11 | -6.4 | 4e-08 | -10.4 | 1.9e-06 |
| linear_contextual | +12.7 [+8.3, +17.1] | 9.1e-08 | +13.2 | 9.1e-09 | -0.2 | 0.92 |

**Fixed-type + hinted minus Hierarchical hinted**

| Task | all games: Δ [95% CI] | p | hint games: Δ | p | no-hint games: Δ | p |
|---|---|---|---|---|---|---|
| stationary | +0.9 [+0.5, +1.2] | 1.1e-05 | +0.2 | 0.045 | +0.9 | 2.3e-05 |
| drifting | +3.2 [+2.1, +4.4] | 1.4e-07 | +2.9 | 2.3e-06 | +1.8 | 0.0056 |
| time_delayed | -2.2 [-3.0, -1.3] | 1.7e-06 | -1.3 | 0.0018 | -2.7 | 1.7e-07 |
| patch_foraging | +1.6 [+1.3, +2.0] | 5.9e-15 | +1.3 | 2.8e-12 | +0.5 | 0.002 |
| two_context | +1.3 [+0.7, +1.9] | 1.2e-05 | +1.5 | 6.1e-08 | -0.2 | 0.69 |
| three_context | +1.0 [+0.4, +1.6] | 0.0019 | +1.0 | 0.0011 | -0.0 | 0.97 |
| bernoulli | +0.1 [+0.0, +0.3] | 0.045 | +0.1 | 0.14 | +0.2 | 0.084 |
| linear_contextual | -0.2 [-0.3, -0.1] | 6.4e-05 | -0.1 | 2.5e-09 | -0.3 | 0.089 |

**Hierarchical hinted minus Hierarchical no-hint**

| Task | all games: Δ [95% CI] | p | hint games: Δ | p | no-hint games: Δ | p |
|---|---|---|---|---|---|---|
| stationary | +22.9 [+21.0, +24.8] | 6.4e-72 | +23.6 | 1.4e-73 | +0.3 | 6.3e-27 |
| drifting | +6.8 [+5.6, +8.1] | 2e-20 | +7.6 | 3.5e-21 | +0.2 | 0.00039 |
| time_delayed | -1.2 [-2.1, -0.2] | 0.015 | -1.2 | 0.017 | -0.1 | 0.066 |
| patch_foraging | -1.9 [-2.8, -1.0] | 5e-05 | -1.9 | 8.3e-05 | -0.1 | 1.3e-05 |
| two_context | +8.2 [+5.3, +11.1] | 1.5e-07 | +9.1 | 9.2e-08 | -0.3 | 0.024 |
| three_context | +8.4 [+6.1, +10.8] | 1.4e-10 | +8.9 | 7.9e-11 | -0.2 | 0.17 |
| bernoulli | +8.7 [+5.7, +11.7] | 6.1e-08 | +9.4 | 6.2e-08 | +0.2 | 6.1e-09 |
| linear_contextual | +1.5 [+0.9, +2.2] | 1.3e-05 | +1.7 | 3e-06 | -0.4 | 8.5e-05 |

**Non-hierarchical hinted minus Non-hierarchical no-hint**

| Task | all games: Δ [95% CI] | p | hint games: Δ | p | no-hint games: Δ | p |
|---|---|---|---|---|---|---|
| stationary | +21.3 [+19.3, +23.3] | 3.7e-62 | +21.6 | 1.2e-62 | +0.7 | 8.2e-29 |
| drifting | +6.6 [+5.1, +8.2] | 3.5e-14 | +7.3 | 2.6e-14 | +0.4 | 0.005 |
| time_delayed | +1.5 [+0.6, +2.4] | 0.0011 | +1.8 | 0.00029 | -0.4 | 2e-05 |
| patch_foraging | -1.9 [-2.2, -1.6] | 3.2e-28 | -1.8 | 4e-31 | -0.2 | 0.0016 |
| two_context | +7.8 [+4.8, +10.8] | 9.7e-07 | +8.7 | 5.8e-07 | -0.3 | 0.095 |
| three_context | +5.0 [+3.0, +6.9] | 1.7e-06 | +5.4 | 4.9e-07 | -0.7 | 0.017 |
| bernoulli | +7.7 [+5.1, +10.3] | 5.4e-08 | +8.1 | 8.7e-08 | +0.6 | 0.00014 |
| linear_contextual | -0.7 [-0.8, -0.6] | 1.9e-17 | -0.7 | 2.7e-16 | -0.1 | 0.0077 |

## Hint effects by hint category, and the shared value hint across tasks

Pearson r between each model's simulated per-hint effects and people's, by hint category (bootstrap 95% CI over hints). Then the arm-naming value hint that appears with the same meaning in stationary, bernoulli and time delayed; effect = regret reduction vs the agent's own no-hint games, % of arm spread (people: LMM; models: OLS/HC3).

| Category | n | Fixed-type + hinted | Hierarchical hinted | Non-hierarchical hinted | Hierarchical no-hint | Non-hierarchical no-hint |
|---|---|---|---|---|---|---|
| Value | 12 | 0.66 [0.15, 0.94] | 0.49 [0.08, 0.89] | 0.67 [0.16, 0.98] | -0.42 [-0.83, 0.47] | -0.68 [-0.92, -0.03] |
| Procedure | 12 | 0.90 [-0.36, 0.99] | 0.87 [-0.47, 0.98] | 0.05 [-0.35, 0.47] | -0.10 [-0.92, 0.51] | 0.42 [-0.75, 0.86] |
| Structure | 16 | 0.11 [-0.12, 0.36] | 0.43 [0.13, 0.74] | 0.12 [-0.33, 0.59] | 0.12 [-0.23, 0.51] | 0.11 [-0.39, 0.54] |
| All | 40 | 0.79 [0.63, 0.92] | 0.73 [0.64, 0.88] | 0.75 [0.47, 0.94] | 0.09 [-0.17, 0.36] | 0.13 [-0.16, 0.43] |

| Task | People | Fixed-type + hinted | Hierarchical hinted | Non-hierarchical hinted | Hierarchical no-hint | Non-hierarchical no-hint |
|---|---|---|---|---|---|---|
| stationary | 12.9 [11.4, 14.5] | 11.9 | 13.8 | 11.9 | -0.0 | -0.0 |
| bernoulli | 15.7 [7.9, 23.5] | 25.2 | 25.6 | 18.2 | -0.0 | 0.0 |
| time_delayed | 24.5 [16.7, 32.4] | 30.9 | 34.6 | 33.5 | -0.0 | -0.0 |

## Simulated per-hint effects: agreement with people and with the full model

| Variant | subset | n | r (people) | ρ (people) | MAE (people) | r (full) | MAE (full) | mean |effect| |
|---|---|---|---|---|---|---|---|---|
| Fixed-type + hinted | all | 40 | 0.79 | 0.64 | 3.9 | 0.96 | 1.3 | 4.0 |
| Fixed-type + hinted | V | 12 | 0.66 | 0.64 | 6.3 | 0.92 | 2.0 | 12.8 |
| Fixed-type + hinted | P | 12 | 0.90 | 0.25 | 3.7 | 0.92 | 0.9 | 0.5 |
| Fixed-type + hinted | S | 16 | 0.11 | 0.16 | 2.3 | 0.26 | 1.1 | 0.1 |
| Fixed-type + hinted | non-V | 28 | 0.78 | 0.08 | 2.9 | 0.79 | 1.0 | 0.3 |
| Hierarchical hinted | all | 40 | 0.73 | 0.72 | 4.1 | 1.00 | 0.0 | 4.6 |
| Hierarchical hinted | V | 12 | 0.49 | 0.62 | 7.3 | 1.00 | 0.0 | 12.5 |
| Hierarchical hinted | P | 12 | 0.87 | 0.21 | 3.4 | 1.00 | 0.0 | 1.4 |
| Hierarchical hinted | S | 16 | 0.43 | 0.53 | 2.3 | 1.00 | 0.0 | 1.1 |
| Hierarchical hinted | non-V | 28 | 0.75 | 0.31 | 2.8 | 1.00 | 0.0 | 1.2 |
| Non-hierarchical hinted | all | 40 | 0.75 | 0.62 | 4.1 | 0.80 | 3.0 | 5.0 |
| Non-hierarchical hinted | V | 12 | 0.67 | 0.56 | 6.6 | 0.60 | 7.2 | 16.5 |
| Non-hierarchical hinted | P | 12 | 0.05 | 0.14 | 4.2 | 0.08 | 1.4 | 0.1 |
| Non-hierarchical hinted | S | 16 | 0.12 | 0.17 | 2.2 | 0.23 | 1.1 | 0.0 |
| Non-hierarchical hinted | non-V | 28 | -0.01 | 0.21 | 3.0 | 0.04 | 1.2 | 0.0 |
| Hierarchical no-hint | all | 40 | 0.09 | 0.11 | 6.8 | 0.11 | 4.6 | 0.0 |
| Hierarchical no-hint | V | 12 | -0.42 | -0.31 | 15.5 | -0.03 | 12.5 | 0.0 |
| Hierarchical no-hint | P | 12 | -0.10 | -0.34 | 4.1 | 0.04 | 1.4 | 0.0 |
| Hierarchical no-hint | S | 16 | 0.12 | 0.16 | 2.2 | 0.29 | 1.1 | 0.0 |
| Hierarchical no-hint | non-V | 28 | 0.11 | 0.12 | 3.0 | 0.19 | 1.2 | 0.0 |
| Non-hierarchical no-hint | all | 40 | 0.13 | 0.14 | 6.8 | 0.09 | 4.6 | 0.0 |
| Non-hierarchical no-hint | V | 12 | -0.68 | -0.61 | 15.5 | 0.13 | 12.5 | 0.0 |
| Non-hierarchical no-hint | P | 12 | 0.42 | -0.14 | 4.1 | 0.52 | 1.4 | 0.0 |
| Non-hierarchical no-hint | S | 16 | 0.11 | 0.09 | 2.2 | -0.06 | 1.1 | 0.0 |
| Non-hierarchical no-hint | non-V | 28 | 0.36 | 0.15 | 3.0 | 0.31 | 1.2 | 0.0 |
| paper_bayesian (repo fit gamma) | all | 40 | 0.73 | 0.75 | 4.2 | 1.00 | 0.1 | 4.7 |
| paper_bayesian (repo fit gamma) | V | 12 | 0.48 | 0.62 | 7.4 | 1.00 | 0.1 | 12.6 |
| paper_bayesian (repo fit gamma) | P | 12 | 0.81 | 0.40 | 3.4 | 0.94 | 0.4 | 1.7 |
| paper_bayesian (repo fit gamma) | S | 16 | 0.43 | 0.53 | 2.3 | 1.00 | 0.0 | 1.1 |
| paper_bayesian (repo fit gamma) | non-V | 28 | 0.73 | 0.37 | 2.8 | 0.95 | 0.2 | 1.3 |

V = value hints (name an arm), P = procedure / strategy hints, S = structure / dynamics hints (constants.HINT_LABELS).

## Task-level hint effects (any hint vs no hint, % of arm spread)

| Task | Fixed-type + hinted | Hierarchical hinted | Non-hierarchical hinted | Hierarchical no-hint | Non-hierarchical no-hint |
|---|---|---|---|---|---|
| stationary | 4.3 | 5.6 | 4.3 | 0.0 | 0.0 |
| drifting | 4.0 | 4.5 | 4.0 | 0.0 | 0.0 |
| time_delayed | 11.9 | 9.8 | 13.4 | 0.0 | 0.0 |
| patch_foraging | 0.1 | 1.3 | 0.1 | 0.0 | 0.0 |
| two_context | 2.5 | 3.3 | 9.3 | 0.0 | 0.0 |
| three_context | 2.2 | 2.8 | 4.9 | 0.0 | 0.0 |
| bernoulli | 5.0 | 5.3 | 3.5 | 0.0 | 0.0 |
| linear_contextual | 2.2 | 3.4 | 0.0 | 0.0 | 0.0 |

## Which hints change most under each ablation

**Fixed-type + hinted** (largest |Δ effect| vs full, % of arm spread):
- time_delayed: "{max2} is the best, even if it doesn't seem like it at first." +16.1 → +27.8 (Δ +11.6; mu hint)
- time_delayed: "{max2} is the best." +34.6 → +30.9 (Δ -3.7; mu hint)
- linear_contextual: "The first number has a negative impact on the points. Only choose buttons with a 0 as the first number." +3.1 → +1.2 (Δ -1.9; theta hint)
- stationary: "{max2} is the best." +13.8 → +11.9 (Δ -1.9; mu hint)
- two_context: "The button values change for square and circle." +1.7 → -0.0 (Δ -1.7; tau hint, undefined under this ablation)
- stationary: "Explore all buttons, then stick with the one that gave the most points." +1.4 → -0.2 (Δ -1.6; history hint)
  Undefined under this ablation (19 hints, effect should be ≈ 0): two_context: The button values change for square and circle.; drifting: The button values drift by a small number at each …; stationary: One button is the best.; drifting: The button values change gradually.; three_context: When you see a square or circle, the buttons are s…; patch_foraging: The points you get on the first pick of a button a…; patch_foraging: Each button value will decrease if you keep choosi…; linear_contextual: The three numbers under each button combine to cal…; linear_contextual: One of the three numbers under the buttons always …; two_context: Pay attention to the shapes.; two_context: Explore the buttons to find the best one for each …; two_context: Each shape corresponds to a different set of butto…; time_delayed: There is a time delay between your choice and the …; three_context: Explore the buttons to find the best one for each …; three_context: Pay attention to the shapes.; three_context: The button values change under each shape.; bernoulli: Choose a button multiple times to learn its probab…; bernoulli: The probability for returning points is fixed for …; bernoulli: There is a chance of returning 0 points even if th…

**Non-hierarchical hinted** (largest |Δ effect| vs full, % of arm spread):
- two_context: "When you see a square, {max2} is the best. When you see a circle, {min2} is the best." +12.3 → +46.5 (Δ +34.2; mu hint)
- time_delayed: "{max2} is the best, even if it doesn't seem like it at first." +16.1 → +33.5 (Δ +17.4; mu hint)
- three_context: "When you see a square, choose G. When you see a circle, choose A. When you see a triangle, explore all buttons to find the best." +10.5 → +24.5 (Δ +13.9; mu hint)
- bernoulli: "{max2} is most likely to return 100 points." +25.6 → +18.2 (Δ -7.5; mu hint)
- linear_contextual: "If the third number under the button is 1, you will get the most points." +5.9 → -0.0 (Δ -5.9; theta hint, undefined under this ablation)
- linear_contextual: "Choose the button with a 1 as its third number." +5.9 → +0.0 (Δ -5.9; theta hint, undefined under this ablation)
  Undefined under this ablation (23 hints, effect should be ≈ 0): linear_contextual: If the third number under the button is 1, you wil…; linear_contextual: Choose the button with a 1 as its third number.; linear_contextual: The first number has a negative impact on the poin…; two_context: The button values change for square and circle.; drifting: The button values drift by a small number at each …; stationary: One button is the best.; drifting: The button values change gradually.; three_context: When you see a square or circle, the buttons are s…; patch_foraging: The points you get on the first pick of a button a…; patch_foraging: Each button value will decrease if you keep choosi…; linear_contextual: The three numbers under each button combine to cal…; linear_contextual: One of the three numbers under the buttons always …; two_context: Explore the buttons to find the best one for each …; two_context: Pay attention to the shapes.; two_context: Each shape corresponds to a different set of butto…; time_delayed: There is a time delay between your choice and the …; three_context: Pay attention to the shapes.; three_context: The button values change under each shape.; three_context: Explore the buttons to find the best one for each …; time_delayed: The points correspond to the choice you made 3 tri…; bernoulli: The probability for returning points is fixed for …; bernoulli: Choose a button multiple times to learn its probab…; bernoulli: There is a chance of returning 0 points even if th…

**Hierarchical no-hint** (largest |Δ effect| vs full, % of arm spread):
- time_delayed: "{max2} is the best." +34.6 → -0.0 (Δ -34.6; mu hint)
- bernoulli: "{max2} is most likely to return 100 points." +25.6 → -0.0 (Δ -25.6; mu hint)
- time_delayed: "{max2} is the best, even if it doesn't seem like it at first." +16.1 → -0.0 (Δ -16.1; mu hint)
- stationary: "{max2} is the best." +13.8 → -0.0 (Δ -13.8; mu hint)
- two_context: "When you see a square, {max2} is the best. When you see a circle, {min2} is the best." +12.3 → -0.0 (Δ -12.3; mu hint)
- drifting: "{max2} begins as the best, but the button values change over trials." +11.0 → -0.0 (Δ -11.0; mu hint)

**Non-hierarchical no-hint** (largest |Δ effect| vs full, % of arm spread):
- time_delayed: "{max2} is the best." +34.6 → -0.0 (Δ -34.6; mu hint)
- bernoulli: "{max2} is most likely to return 100 points." +25.6 → +0.0 (Δ -25.6; mu hint)
- time_delayed: "{max2} is the best, even if it doesn't seem like it at first." +16.1 → -0.0 (Δ -16.1; mu hint)
- stationary: "{max2} is the best." +13.8 → -0.0 (Δ -13.8; mu hint)
- two_context: "When you see a square, {max2} is the best. When you see a circle, {min2} is the best." +12.3 → -0.0 (Δ -12.3; mu hint)
- drifting: "{max2} begins as the best, but the button values change over trials." +11.0 → -0.0 (Δ -11.0; mu hint)

## Hints whose meaning function cannot act under an ablation

- **Type known**: the 16 pure type-gate hints (kind = tau) select every posterior sample once the type is given, so their weight is constant and the predicted effect is 0 by construction. Value (mu), parameter (theta) and history hints are unaffected in form.
- **Stationary-only**: the type-gate hints (tau) have no type to gate, and the parameter hints (theta: the 3-trial delay and the linear-contextual feature hints) refer to parameters a stationary bandit does not have, so both kinds are undefined (constant weight, effect 0). The value and history hints act on the sampled arm means and are defined. The context-dependent value hints are defined but the stationary model has no context, so they reduce to "the best arm is the top-ranked one".
- **No-hint models**: γ = 0, every hint weight is constant, so all 40 effects are 0 up to simulation noise.

