import argparse, dataclasses as dc, json, sys
from pathlib import Path
import numpy as np, pandas as pd
from tqdm import tqdm

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "bayesian"))
sys.path.insert(0, str(REPO / "analysis"))  # for exp_paths

from exp_paths import SIM_DIR, FIT_JSON
from constants import TYPE_ORDER, HINT_LABELS, IDEAL_PRIORS
from hint_functions import HINT_TEMPLATE_TO_FUNCTION
from bandits import (
    GaussianMAB,
    DriftingMAB,
    TimeDelayedMAB,
    PatchForagingMAB,
    TwoContextMAB,
    ThreeContextMAB,
    BernoulliMAB,
    LinearContextualMAB,
)
from hierarchical import default_hierarchical_belief
from distributions import RandomNumberGenerator

MODEL_COMP = FIT_JSON  # from exp_paths (env BAYES_FIT_JSON)
OUT = (
    SIM_DIR / "bayesian_sim_trials.parquet"
)  # SIM_DIR from exp_paths (env BAYES_SIM_DIR)

N_ARMS = 5
ARM_MEANS = (10, 20, 40, 60, 80)
ARM_STDEVS = (5, 5, 5, 5, 5)
N_TRIALS_BY_TYPE = {
    "stationary": 20,
    "drifting": 20,
    "time_delayed": 20,
    "patch_foraging": 20,
    "linear_contextual": 20,
    "two_context": 40,
    "three_context": 40,
    "bernoulli": 40,
}

BANDITS = {
    "stationary": lambda s: GaussianMAB(ARM_MEANS, ARM_STDEVS),
    "drifting": lambda s: DriftingMAB(ARM_MEANS, ARM_STDEVS, drift_epsilon=2),
    "time_delayed": lambda s: TimeDelayedMAB(ARM_MEANS, ARM_STDEVS, delay_k=3),
    "patch_foraging": lambda s: PatchForagingMAB(
        ARM_MEANS, ARM_STDEVS, change_rate=0.2
    ),
    "two_context": lambda s: TwoContextMAB(ARM_MEANS, ARM_STDEVS),
    "three_context": lambda s: ThreeContextMAB(ARM_MEANS, ARM_STDEVS, random_seed=s),
    "bernoulli": lambda s: BernoulliMAB(ARM_MEANS, ARM_STDEVS),
    "linear_contextual": lambda s: LinearContextualMAB(
        arm_means=(0,) * N_ARMS,
        arm_stdevs=(1,) * N_ARMS,
        theta=np.array([-2.0, 0.0, 5.0]),
        n_features=3,
        noise_std=1.0,
    ),
}

MODEL_NAME = "Bayesian + hints (fitted hs)"


def _softmax(x):
    x = np.where(np.isfinite(x), x, -np.inf)
    m = np.max(x)
    if not np.isfinite(m):
        return np.ones_like(x) / len(x)
    p = np.exp(x - m)
    return p / (p.sum() + 1e-300)


def load_fitted_params(model_comp=None, model_name=MODEL_NAME, use_temp=False):
    model_comp = model_comp or MODEL_COMP
    with open(model_comp) as f:
        data = json.load(f)
    for m in data["models"]:
        if m["model"] == model_name:
            hs = m["fitted_hint_strength"]["mean"]
            temp = (
                m["fitted_temp"]["mean"] if (use_temp and "fitted_temp" in m) else None
            )
            return temp, hs
    raise ValueError(f"Model '{model_name}' not found in {model_comp}")


def _build_belief(bandit):
    arm_stdevs = np.asarray(bandit.arm_stdevs, dtype=float)
    belief = default_hierarchical_belief(
        n_arms=bandit.n_arms, arm_stdevs=arm_stdevs, **IDEAL_PRIORS
    )
    if hasattr(bandit, "n_features"):
        lc = belief.type_specific_priors.setdefault("linear_contextual", {})
        lc["n_features"], lc["noise_std"] = bandit.n_features, bandit.noise_std
    return belief


def _build_priors(belief, btype, t, bandit):
    priors = {
        k: dict(v) if isinstance(v, dict) else v
        for k, v in belief.type_specific_priors.items()
    }
    if hasattr(bandit, "_context_history") and btype in (
        "two_context",
        "three_context",
    ):
        p = dict(priors.get(btype, {}))
        p["context_history"] = dict(bandit._context_history)
        if t in bandit._context_history:
            p["current_context"] = bandit._context_history[t]
        priors[btype] = p
    if hasattr(bandit, "_trial_features") and btype == "linear_contextual":
        p = dict(priors.get("linear_contextual", {}))
        p["trial_features"] = dict(bandit._trial_features)
        if t in bandit._trial_features:
            p["current_features"] = bandit._trial_features[t]
            p["n_features"] = bandit._trial_features[t].shape[1]
        priors["linear_contextual"] = p
    return priors


def _means_at_sample(bandit, t, pre_means):
    if pre_means is not None:
        return pre_means
    if hasattr(bandit, "expected_reward_at_trial"):
        return np.array(bandit.expected_reward_at_trial(t))
    if hasattr(bandit, "_means_for_context"):
        return bandit._means_for_context(bandit._context_history.get(t, 0))
    if hasattr(bandit, "_current_means"):
        return bandit._current_means.copy()  # drifting: post-drift
    return np.asarray(bandit.arm_means, dtype=float)


def run_game(
    bandit,
    btype,
    hint_fn,
    hs,
    n_trials,
    agent_seed,
    env_seed,
    ws=500,
    hint_start=0,
    temp=None,
):
    arng, erng = RandomNumberGenerator(agent_seed), RandomNumberGenerator(env_seed)
    belief0 = _build_belief(bandit)
    n = bandit.n_arms
    obs = np.full((n_trials, n), np.nan)
    acts = np.zeros(n_trials, dtype=int)
    rows = []
    for t in range(n_trials):
        if hasattr(bandit, "generate_context"):
            bandit.generate_context(t, erng)
        if hasattr(bandit, "generate_trial_features"):
            bandit.generate_trial_features(t, erng)
        belief = dc.replace(
            belief0, type_specific_priors=_build_priors(belief0, btype, t, bandit)
        )
        if t > 0:
            belief = belief.update(acts[:t], obs[:t], np.arange(t))

        samples = belief.sample_type_and_params(rng=arng, n=ws)
        arm_means = np.array(
            [belief.type_specs[i].expected_rewards_from_params(p) for i, p in samples]
        )
        best = np.argmax(arm_means, axis=1)

        if hint_fn is not None and t >= hint_start:
            t_ids = np.array([i for i, _ in samples])
            ctx = (
                bandit._context_history.get(t)
                if hasattr(bandit, "_context_history")
                else None
            )
            logw = (
                np.asarray(
                    hint_fn(
                        t_ids,
                        arm_means.T,
                        [p for _, p in samples],
                        actions=acts[:t],
                        obs=obs[:t],
                        context=ctx,
                    ),
                    dtype=float,
                )
                * hs
            )
            w = _softmax(logw)
            ap = np.array([w[best == a].sum() for a in range(n)])
        else:
            ap = np.bincount(best, minlength=n).astype(float)
        ap = ap / (ap.sum() + 1e-300)
        # temp=None -> sample directly from the hint-weighted Thompson policy (normative)
        probs = ap if temp is None else _softmax(np.log(np.clip(ap, 1e-12, 1.0)) / temp)
        a = int(arng.np.choice(n, p=probs))
        acts[t] = a

        # Capture patch means BEFORE sample_rewards (depletion happens inside)
        pre_means = (
            bandit._current_rewards.copy()
            if hasattr(bandit, "_current_rewards")
            else None
        )

        if hasattr(bandit, "sample_rewards"):
            obs[t, :] = bandit.sample_rewards(t=t, action=a, rng=erng)
        else:
            obs[t, a] = bandit.reward_dists.sample(rng=erng, n=1)[0, a]

        means = _means_at_sample(bandit, t, pre_means)
        opt = int(np.argmax(means))
        rows.append(
            dict(
                trial_num=t,
                choice=a,
                optimal=opt,
                correct=int(a == opt),
                regret=float(means[opt] - means[a]),
            )
        )
    return rows


def simulate_all(hs, n_envs, seed, temp=None):
    rng = RandomNumberGenerator(seed)
    rows = []
    for btype in TYPE_ORDER:
        hints = HINT_TEMPLATE_TO_FUNCTION.get(btype, {})
        n_trials = N_TRIALS_BY_TYPE.get(btype, 20)
        es_list = [rng.randint(0, 2**32) for _ in range(n_envs)]
        as_list = [rng.randint(0, 2**32) for _ in range(n_envs)]
        conditions = [("No hint.", None)] + list(hints.items())
        for text, fn in conditions:
            cat = ",".join(HINT_LABELS.get(text, [])) if fn is not None else "none"
            desc = f"{btype}/{getattr(fn, '__name__', 'none')}"
            for i, (es, aseed) in enumerate(
                tqdm(list(zip(es_list, as_list)), desc=desc, leave=False)
            ):
                bandit = BANDITS[btype](es % 1000)
                for td in run_game(
                    bandit, btype, fn, hs, n_trials, aseed, es, temp=temp
                ):
                    td.update(type=btype, ogHint=text, hint_cat=cat, sim_id=i)
                    rows.append(td)
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-envs", type=int, default=100)
    p.add_argument("--resim", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--gamma",
        type=float,
        default=None,
        help="fixed hint strength (Exp 3); if omitted, load gamma from the fit",
    )
    p.add_argument(
        "--model-comp",
        type=Path,
        default=None,
        help="model_comparison.json to load gamma/T from (default: BAYES_FIT_JSON)",
    )
    p.add_argument(
        "--model-name",
        type=str,
        default=MODEL_NAME,
        help='which fitted model to read (e.g. "Bayesian + hints (fitted T + hs)")',
    )
    p.add_argument(
        "--use-temp",
        action="store_true",
        help="apply the fitted softmax temperature in simulation (Exp 1)",
    )
    p.add_argument(
        "--out", type=Path, default=None, help="output parquet (default: BAYES_SIM_DIR)"
    )
    args = p.parse_args()

    out = args.out or OUT
    if out.exists() and not args.resim:
        print(f"Cached: {out}")
        return

    if args.gamma is not None:
        temp, hs = None, args.gamma
    else:
        temp, hs = load_fitted_params(args.model_comp, args.model_name, args.use_temp)
    print(f"Params: hs={hs:.2f}, temp={temp}")
    print(f"Simulating ({args.n_envs} envs/condition)...")
    df = simulate_all(hs, args.n_envs, args.seed, temp=temp)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    print(f"Saved {len(df)} rows to {out}")
    print(f"  Types: {sorted(df['type'].unique())}")
    print(f"  Conditions: {df.groupby(['type', 'ogHint']).ngroups}")


if __name__ == "__main__":
    main()
