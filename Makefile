# make all: data -> stats -> tables -> figures 1-5 (CPU, ~10 min). make ablations / make test, see README.
PY ?= python
T := analysis/tables

.PHONY: all data stats tables figures ablations test

all: data stats tables figures

data:
	$(PY) behavioral/hint_time/data/make_clean.py
	$(PY) behavioral/instructed_bandits/data/make_clean.py
	$(PY) centaur/data/make_clean.py
	$(PY) centaur/data/make_clean_hint_time.py

stats:
	$(PY) behavioral/instructed_bandits/analysis/hint_effects.py --refit
	$(PY) behavioral/hint_time/analysis/hint_effects.py --refit
	$(PY) -c 'import sys; sys.path[:0] = ["analysis", "."]; import result2_hint_effects as R; R.refit_model_effects()'

tables:
	cd $(T) && $(PY) hint_effects.py && $(PY) model_diff.py && $(PY) trajectory.py && $(PY) trajectory_corr.py && $(PY) hint_corr.py && $(PY) timing.py

figures:
	$(PY) analysis/result1_learning.py
	$(PY) analysis/result3_finegrained.py
	$(PY) analysis/result4_hint_timing.py
	$(PY) analysis/result2_hint_effects.py
	$(PY) analysis/result5_model_comparison.py

ablations:
	$(PY) analysis/bayes_ablations.py

test:
	pytest -q bayesian
