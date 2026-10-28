# Sensitivity / power analysis (frequentist, two-sided)

Simulations per delta: 1000; grid (SD units): [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.65, 0.8, 1.0, 1.25]; detection |z| > 1.960 (two-sided p < .05). Runtime 26 min on 32 workers.

Each cell is simulated with the model the paper fits (see module docstring); tau / sigma are the participant and residual SDs of that fit on the real data. Outcome: regret as % of the game's arm spread. MDE = smallest true effect (% of spread) detected with 80% power. `mde_80_fdr` uses the per-family Benjamini–Hochberg threshold the paper's FDR step realised in the observed data:

- Study 1 task-level (8): largest rejected raw p = 0.0077 (|z| > 2.67)
- Study 1 per-hint (40): largest rejected raw p = 0.0078 (|z| > 2.66)
- Study 2 per-hint x onset (40): largest rejected raw p = 0.0036 (|z| > 2.91)

## MDE range per inferential level

- **Study 1 task-level (8 cells)**: MDE 80% power (p < .05) from 2.13 to 10.58 % of arm spread; 8/8 cells reached 80% within the grid.
- **Study 1 per-hint (40 cells)**: MDE 80% power (p < .05) from 2.22 to 13.25 % of arm spread; 40/40 cells reached 80% within the grid.
- **Study 2 per-hint x onset, post-onset window with pre-onset covariate (40 cells)**: MDE 80% power (p < .05) from 3.80 to 8.85 % of arm spread; 40/40 cells reached 80% within the grid.

## Study 1 task-level effects significant after FDR (4): observed |effect| vs MDE

| type | observed | SE | p_adj | stars | MDE_80 | MDE_80_fdr | |obs|/MDE | tau | sigma | n_subj | n_hint/n_nohint |
|---|---|---|---|---|---|---|---|---|---|---|---|
| stationary | 5.20 | 0.75 | 0.000 | *** | 2.13 | 2.67 | 2.44 | 6.23 | 9.91 | 317 | 775/240 |
| time_delayed | 14.14 | 3.86 | 0.001 | ** | 10.53 | 13.31 | 1.34 | 11.02 | 23.84 | 139 | 239/53 |
| two_context | 7.05 | 2.64 | 0.015 | * | 7.44 | 9.33 | 0.95 | 12.38 | 15.75 | 143 | 232/60 |
| linear_contextual | 10.36 | 3.68 | 0.013 | * | 10.58 | 13.36 | 0.98 | 13.70 | 18.30 | 138 | 265/37 |

## Study 1 per-hint effects significant after FDR (13): observed |effect| vs MDE

| type | ogHint | observed | SE | p_adj | stars | MDE_80 | MDE_80_fdr | |obs|/MDE | tau | sigma | n_subj | n_hint/n_nohint |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| stationary | {max1} is better than {min2}, {min1}, and {mid}. | 7.05 | 1.61 | 0.000 | *** | 4.46 | 5.61 | 1.58 | 6.98 | 8.41 | 317 | 47/240 |
| stationary | {max2} is the best. | 12.95 | 0.81 | 0.000 | *** | 2.29 | 2.85 | 5.65 | 6.98 | 8.41 | 317 | 220/240 |
| drifting | {max2} begins as the best, but the button values change over | 9.28 | 2.75 | 0.003 | ** | 7.82 | 9.74 | 1.19 | 9.94 | 11.92 | 141 | 45/59 |
| time_delayed | {max2} is the best. | 24.52 | 4.00 | 0.000 | *** | 11.01 | 13.80 | 2.23 | 8.91 | 19.74 | 139 | 59/53 |
| time_delayed | {max2} is the best, even if it doesn't seem like it at first | 37.23 | 4.09 | 0.000 | *** | 11.75 | 14.40 | 3.17 | 8.91 | 19.74 | 139 | 54/53 |
| patch_foraging | Find a few good buttons and alternate between them to maximi | 5.89 | 0.98 | 0.000 | *** | 2.79 | 3.46 | 2.11 | 7.50 | 10.27 | 316 | 233/237 |
| patch_foraging | The points you get on the first pick of a button is the most | -2.96 | 0.99 | 0.009 | ** | 2.79 | 3.49 | 1.06 | 7.50 | 10.27 | 316 | 221/237 |
| two_context | When you see a square, {max2} is the best. When you see a ci | 24.50 | 3.24 | 0.000 | *** | 9.25 | 11.60 | 2.65 | 12.76 | 13.69 | 143 | 45/60 |
| three_context | When you see a square, choose G. When you see a circle, choo | 17.90 | 2.78 | 0.000 | *** | 7.86 | 9.91 | 2.28 | 11.77 | 11.67 | 138 | 56/46 |
| bernoulli | There is a chance of returning 0 points even if the button i | -9.91 | 3.73 | 0.024 | * | 10.77 | 13.36 | 0.92 | 7.00 | 16.18 | 136 | 46/42 |
| bernoulli | {max2} is most likely to return 100 points. | 15.70 | 3.96 | 0.000 | *** | 11.14 | 13.89 | 1.41 | 7.00 | 16.18 | 136 | 35/42 |
| linear_contextual | Choose the button with a 1 as its third number. | 22.97 | 3.87 | 0.000 | *** | 11.24 | 13.98 | 2.04 | 11.65 | 15.31 | 138 | 49/37 |
| linear_contextual | If the third number under the button is 1, you will get the  | 27.05 | 3.71 | 0.000 | *** | 10.61 | 13.07 | 2.55 | 11.65 | 15.31 | 138 | 53/37 |

## Study 2 per-hint x onset effects significant after FDR (5): observed |effect| vs MDE

| type | hint_timing | ogHint | observed | SE | p_adj | stars | MDE_80 | MDE_80_fdr | |obs|/MDE | tau | sigma | n_subj | n_hint/n_nohint |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| stationary | 0 | {max2} is the best. | 11.27 | 1.37 | 0.000 | *** | 3.81 | 5.22 | 2.96 | 4.52 | 8.41 | 75 | 75/75 |
| stationary | 0 | {max1} and {max2} are better than the others. | 5.58 | 1.37 | 0.001 | *** | 3.82 | 5.28 | 1.46 | 4.52 | 8.41 | 75 | 75/75 |
| patch_foraging | 0 | Find a few good buttons and alternate between them to maximi | 6.24 | 1.57 | 0.001 | *** | 4.45 | 5.86 | 1.40 | 6.69 | 9.62 | 75 | 75/75 |
| patch_foraging | 0 | The points you get on the first pick of a button is the most | -4.57 | 1.57 | 0.029 | * | 4.47 | 5.86 | 1.02 | 6.69 | 9.62 | 75 | 75/75 |
| patch_foraging | 12 | Find a few good buttons and alternate between them to maximi | 8.37 | 2.81 | 0.029 | * | 7.99 | 10.74 | 1.05 | 8.10 | 15.69 | 262 | 83/75 |

## Significant effects smaller than their MDE (|observed| < MDE_80)


- S1 task: two_context / any hint: |observed| 7.05 vs MDE 7.44
- S1 task: linear_contextual / any hint: |observed| 10.36 vs MDE 10.58
- S1 hint: bernoulli / There is a chance of returning 0 points even if the button i: |observed| 9.91 vs MDE 10.77

## Cells that did not reach 80% power within the grid (p < .05 criterion)

none

## tau (participant SD) and sigma (residual SD) per simulated frame

| frame | tau | sigma | sd_total | n_subjects | n_games | fit |
|---|---|---|---|---|---|---|
| S1 task-level | stationary | 6.23 | 9.91 | 11.70 | 317 | 1015 | mixedlm |
| S1 task-level | drifting | 9.68 | 12.43 | 15.76 | 141 | 306 | mixedlm |
| S1 task-level | time_delayed | 11.02 | 23.84 | 26.26 | 139 | 292 | mixedlm |
| S1 task-level | patch_foraging | 7.41 | 10.70 | 13.01 | 316 | 1380 | mixedlm |
| S1 task-level | two_context | 12.38 | 15.75 | 20.04 | 143 | 292 | mixedlm |
| S1 task-level | three_context | 12.42 | 13.32 | 18.21 | 138 | 304 | mixedlm |
| S1 task-level | bernoulli | 6.27 | 18.05 | 19.11 | 136 | 272 | mixedlm |
| S1 task-level | linear_contextual | 13.70 | 18.30 | 22.86 | 138 | 302 | mixedlm |
| S1 per-hint | stationary | 6.98 | 8.41 | 10.93 | 317 | 1015 | mixedlm |
| S1 per-hint | drifting | 9.94 | 11.92 | 15.53 | 141 | 306 | mixedlm |
| S1 per-hint | time_delayed | 8.91 | 19.74 | 21.66 | 139 | 292 | mixedlm |
| S1 per-hint | patch_foraging | 7.50 | 10.27 | 12.72 | 316 | 1380 | mixedlm |
| S1 per-hint | two_context | 12.76 | 13.69 | 18.71 | 143 | 292 | mixedlm |
| S1 per-hint | three_context | 11.77 | 11.67 | 16.57 | 138 | 304 | mixedlm |
| S1 per-hint | bernoulli | 7.00 | 16.18 | 17.63 | 136 | 272 | mixedlm |
| S1 per-hint | linear_contextual | 11.65 | 15.31 | 19.24 | 138 | 302 | mixedlm |
| S2 | stationary | onset 0 | 4.52 | 8.41 | 9.55 | 75 | 450 | mixedlm |
| S2 | stationary | onset 4 | 7.64 | 8.85 | 11.69 | 273 | 456 | mixedlm |
| S2 | stationary | onset 8 | 7.76 | 8.22 | 11.30 | 258 | 426 | mixedlm |
| S2 | stationary | onset 12 | 6.68 | 8.53 | 10.83 | 268 | 468 | mixedlm |
| S2 | patch_foraging | onset 0 | 6.69 | 9.62 | 11.72 | 75 | 450 | mixedlm |
| S2 | patch_foraging | onset 4 | 10.38 | 11.05 | 15.16 | 266 | 449 | mixedlm |
| S2 | patch_foraging | onset 8 | 10.02 | 13.36 | 16.70 | 269 | 470 | mixedlm |
| S2 | patch_foraging | onset 12 | 8.10 | 15.69 | 17.66 | 262 | 431 | mixedlm |

## Type-I error check: detection rate at delta = 0 (should be ~0.05)

- Study 1 task-level: 0.046–0.068
- Study 1 per-hint: 0.040–0.064
- Study 2: 0.037–0.067
- simulated fits that failed: 0
