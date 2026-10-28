from __future__ import annotations

import argparse
import sys
import time
import warnings
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for d in (
    REPO / "bayesian/fit",
    REPO / "bayesian",
    REPO / "analysis",
    REPO,
):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))

import fit_variants as FV  # noqa: E402
import hint_functions as HF  # noqa: E402
import simulate_bayesian as SB  # noqa: E402
from constants import HINT_LABELS, TYPE_ORDER  # noqa: E402
from distributions import RandomNumberGenerator  # noqa: E402

OUT_DIR = REPO / "bayesian/data/ablations"
FITS = REPO / "bayesian/fit/variants"
_ORIG_SB_BELIEF = SB.default_hierarchical_belief


def configure(variant, true_type):
    space, _ = FV.VARIANTS[variant]
    FV.configure_space(space, true_type=true_type)
    # simulate_bayesian imported default_hierarchical_belief by name; patch its copy the same way.
    SB.default_hierarchical_belief = (
        FV.F.default_hierarchical_belief if space != "full" else _ORIG_SB_BELIEF
    )


def hint_fn_for(variant, task_type, hint_text):
    space, hints_on = FV.VARIANTS[variant]
    if not hints_on or hint_text == "No hint.":
        return None
    return HF.HINT_TEMPLATE_TO_FUNCTION[task_type][hint_text]


def gamma_for(variant, params_csv=None):
    if not FV.VARIANTS[variant][1]:
        return 0.0
    csv = params_csv or FITS / "params.csv"
    p = pd.read_csv(csv)
    g = p[p["variant"] == variant]["gamma"].astype(float)
    assert len(g) == 5, f"{variant}: expected 5 folds in {csv}, found {len(g)}"
    return float(g.mean())


def _study1_condition(args):
    variant, btype, text, hs, n_envs, seed = args
    warnings.filterwarnings("ignore")
    configure(variant, true_type=btype)
    fn = hint_fn_for(variant, btype, text)
    # identical seeds to simulate_bayesian.simulate_all: one stream per task, drawn in TYPE_ORDER
    rng = RandomNumberGenerator(seed)
    seeds = {}
    for t in TYPE_ORDER:
        es = [rng.randint(0, 2**32) for _ in range(n_envs)]
        as_ = [rng.randint(0, 2**32) for _ in range(n_envs)]
        seeds[t] = (es, as_)
    es_list, as_list = seeds[btype]
    n_trials = SB.N_TRIALS_BY_TYPE.get(btype, 20)
    cat = ",".join(HINT_LABELS.get(text, [])) if text != "No hint." else "none"
    rows = []
    for i, (es, aseed) in enumerate(zip(es_list, as_list)):
        bandit = SB.BANDITS[btype](es % 1000)
        for td in SB.run_game(bandit, btype, fn, hs, n_trials, aseed, es, temp=None):
            td.update(type=btype, ogHint=text, hint_cat=cat, sim_id=i)
            rows.append(td)
    return rows


def run(variant, n_envs, n_jobs, seed, hs, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [
        (variant, t, text, hs, n_envs, seed)
        for t in TYPE_ORDER
        for text in ["No hint."] + list(HF.HINT_TEMPLATE_TO_FUNCTION.get(t, {}))
    ]
    worker, out = _study1_condition, out_dir / f"{variant}_sim_trials.parquet"
    print(
        f"[{variant}] {len(jobs)} conditions x {n_envs} envs, gamma={hs}, n_jobs={n_jobs}",
        flush=True,
    )
    t0 = time.time()
    rows = []
    if n_jobs == 1:
        for j in jobs:
            rows += worker(j)
    else:
        with Pool(n_jobs) as pool:
            for k, r in enumerate(pool.imap_unordered(worker, jobs), 1):
                rows += r
                if k % 8 == 0:
                    print(
                        f"  {k}/{len(jobs)} conditions {(time.time() - t0) / 60:.1f} min",
                        flush=True,
                    )
    df = pd.DataFrame(rows)
    df.attrs = {}
    df.to_parquet(out, index=False)
    print(
        f"[{variant}] wrote {len(df)} rows -> {out} ({(time.time() - t0) / 60:.1f} min)",
        flush=True,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", required=True, choices=list(FV.VARIANTS))
    p.add_argument("--n-envs", type=int, default=100)
    p.add_argument("--n-jobs", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--gamma", type=float, default=None, help="override the fitted hint strength"
    )
    p.add_argument("--params-csv", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    a = p.parse_args()
    hs = a.gamma if a.gamma is not None else gamma_for(a.variant, a.params_csv)
    run(a.variant, a.n_envs, a.n_jobs, a.seed, hs, a.out_dir)


if __name__ == "__main__":
    main()
