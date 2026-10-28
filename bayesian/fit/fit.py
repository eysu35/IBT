import argparse, dataclasses, hashlib, json, pickle, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import minimize_scalar, minimize
from sklearn.model_selection import KFold
from tqdm import tqdm

FIT_DIR = Path(__file__).resolve().parent
REPO = FIT_DIR.parent.parent
for d in (FIT_DIR.parent, REPO):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))

from constants import normalize_type, IDEAL_PRIORS
from hierarchical import default_hierarchical_belief
from hint_functions import HINT_TEMPLATE_TO_FUNCTION
from distributions import RandomNumberGenerator

N_ARMS = 5
DELAY_K = 3  # time_delayed: trial t's reward = action at t-DELAY_K (matches experiment)
DATA_DIR = REPO / "behavioral" / "instructed_bandits" / "data"
T_B, HS_B = (0.1, 20.0), (0.1, 20.0)


@dataclasses.dataclass
class TrialCache:
    human_choice: int
    ts_counts: np.ndarray  # (n_arms,) hint-free action support
    best_actions: np.ndarray  # (n_samples,)
    raw_hint_logw: np.ndarray  # (n_samples,) or None
    game_type: str
    user_id: int


def _softmax(x):
    x = np.where(np.isfinite(x), x, -np.inf)
    m = np.max(x)
    if not np.isfinite(m):
        return np.ones_like(x) / len(x)
    p = np.exp(x - m)
    return p / (p.sum() + 1e-300)


def action_probs(tc, hs=None):
    if tc.raw_hint_logw is not None and hs is not None:
        w = _softmax(tc.raw_hint_logw * hs)
        ap = np.array([w[tc.best_actions == a].sum() for a in range(N_ARMS)])
    else:
        ap = tc.ts_counts
    return ap / (ap.sum() + 1e-300)


def apply_temp(p, T):
    # T=None -> no temperature (the Thompson policy is already a distribution).
    if T is None:
        return p
    return _softmax(np.log(np.clip(p, 1e-12, 1.0)) / T)


def trial_ll(tc, T=None, hs=None):
    return float(
        np.log(
            np.clip(apply_temp(action_probs(tc, hs), T)[tc.human_choice], 1e-12, 1.0)
        )
    )


def total_ll(caches, T=None, hs=None):
    return sum(trial_ll(tc, T, hs) for tc in caches)


def load_games(d):
    df = pd.read_parquet(d / "behavioral_clean.parquet").dropna(
        subset=["choice", "type"]
    )
    df["type"] = df["type"].apply(normalize_type)
    df["choice"] = (
        pd.to_numeric(df["choice"], errors="coerce").fillna(0).astype(int).clip(lower=0)
    )
    df = df.sort_values(["user_id", "game_id", "trial"])
    games = []
    for (uid, _), g in df.groupby(["user_id", "game_id"]):
        gt = g["type"].iloc[0]
        game = {
            "user_id": int(uid),
            "type": gt,
            "hint": str(g["ogHint"].iloc[0]),
            "choices": g["choice"].tolist(),
            "rewards": g["reward"].tolist()
            if "reward" in g.columns
            else [0.0] * len(g),
        }
        if gt in ("two_context", "three_context") and "contexts" in g.columns:
            game["contexts"] = [int(c) if pd.notna(c) else 0 for c in g["contexts"]]
        if gt == "linear_contextual" and "allArmFts" in g.columns:
            fts = g["allArmFts"].tolist()
            if all(f is not None for f in fts):
                game["allArmFts"] = fts
        games.append(game)
    return games


def _build_priors(defaults, gt, t, contexts, fts):
    priors = {k: dict(v) if isinstance(v, dict) else v for k, v in defaults.items()}
    if gt in ("two_context", "three_context") and contexts is not None:
        p = dict(priors.get(gt, {}))
        p["context_history"] = {i: int(contexts[i]) for i in range(t)}
        if t < len(contexts):
            p["current_context"] = int(contexts[t])
        priors[gt] = p
    if gt == "linear_contextual" and fts is not None:
        p = dict(priors.get("linear_contextual", {}))
        p["noise_std"] = 0.01  # actual experiment: reward = theta @ features, no noise
        tf = {i: np.stack(fts[i]).astype(float) for i in range(t) if fts[i] is not None}
        p["trial_features"] = tf
        cur = fts[t] if t < len(fts) and fts[t] is not None else None
        ref = (
            np.stack(cur).astype(float)
            if cur is not None
            else (next(iter(tf.values())) if tf else None)
        )
        if cur is not None:
            p["current_features"] = ref
        if ref is not None:
            p["n_features"] = ref.shape[1] if ref.ndim == 2 else ref.shape[0]
        priors["linear_contextual"] = p
    return priors


def precompute_games(games, n_samples, seed):
    warnings.filterwarnings("ignore", message="divide by zero", category=RuntimeWarning)
    belief0 = default_hierarchical_belief(n_arms=N_ARMS, **IDEAL_PRIORS)
    rng = RandomNumberGenerator(seed)
    caches = []
    for game in tqdm(games, desc="Precomputing games"):
        gt, T = game["type"], len(game["choices"])
        hint_fn = HINT_TEMPLATE_TO_FUNCTION.get(gt, {}).get(game["hint"])
        contexts, fts = game.get("contexts"), game.get("allArmFts")
        actions = np.clip(np.array(game["choices"], dtype=int), 0, N_ARMS - 1)
        if gt == "time_delayed":
            # Bandit returns 0 for non-delivered arms; delayed reward at trial t is for action t-DELAY_K.
            obs = np.zeros((T, N_ARMS))
            for i in range(DELAY_K, T):
                r = game["rewards"][i]
                if np.isfinite(r):
                    obs[i, actions[i - DELAY_K]] = float(r)
        else:
            obs = np.full((T, N_ARMS), np.nan)
            for i, r in enumerate(game["rewards"]):
                if np.isfinite(r):
                    obs[i, actions[i]] = float(r)
        for t in range(T):
            b = dataclasses.replace(
                belief0,
                type_specific_priors=_build_priors(
                    belief0.type_specific_priors, gt, t, contexts, fts
                ),
            )
            if t > 0:
                b = b.update(actions[:t], obs[:t], np.arange(t))
            samples = b.sample_type_and_params(rng=rng, n=n_samples)
            arm_means = np.array(
                [b.type_specs[i].expected_rewards_from_params(p) for i, p in samples]
            )
            best = np.argmax(arm_means, axis=1)
            logw = None
            if hint_fn is not None:
                t_ids = np.array([i for i, _ in samples])
                logw = np.asarray(
                    hint_fn(
                        t_ids,
                        arm_means.T,
                        [p for _, p in samples],
                        actions=actions[:t],
                        obs=obs[:t],
                        context=int(contexts[t])
                        if contexts is not None and t < len(contexts)
                        else None,
                    ),
                    dtype=float,
                )
            caches.append(
                TrialCache(
                    int(actions[t]),
                    np.bincount(best, minlength=N_ARMS).astype(float),
                    best,
                    logw,
                    gt,
                    game["user_id"],
                )
            )
    return caches


# ---------- Models ----------


class RandomModel:
    name, n_params = "Random (uniform)", 0

    def fit(self, c):
        return {}

    def eval_ll(self, c, p):
        return len(c) * np.log(0.2)

    def eval_acc(self, c, p):
        return 0.2


class BayesModel:
    def __init__(self, fit_temp=False):
        self.fit_temp = fit_temp
        self.name = "Bayesian (fitted T)" if fit_temp else "Bayesian"
        self.n_params = 1 if fit_temp else 0

    def fit(self, c):
        if not self.fit_temp:
            return {}
        T = minimize_scalar(lambda t: -total_ll(c, T=t), bounds=T_B, method="bounded").x
        return {"temp": float(T)}

    def eval_ll(self, c, p):
        return total_ll(c, T=p.get("temp"))

    def eval_acc(self, c, p):
        T = p.get("temp")
        return sum(
            tc.human_choice == np.argmax(apply_temp(action_probs(tc), T)) for tc in c
        ) / max(len(c), 1)


class HintModel:
    def __init__(self, fit_temp=False):
        self.fit_temp = fit_temp
        self.name = (
            "Bayesian + hints (fitted T + hs)"
            if fit_temp
            else "Bayesian + hints (fitted hs)"
        )
        self.n_params = 2 if fit_temp else 1

    def fit(self, c):
        if self.fit_temp:
            r = minimize(
                lambda x: -total_ll(c, T=x[0], hs=x[1]),
                [5.0, 5.0],
                method="L-BFGS-B",
                bounds=[T_B, HS_B],
            )
            return {"temp": float(r.x[0]), "hint_strength": float(r.x[1])}
        hs = minimize_scalar(
            lambda h: -total_ll(c, hs=h), bounds=HS_B, method="bounded"
        ).x
        return {"hint_strength": float(hs)}

    def eval_ll(self, c, p):
        return total_ll(c, T=p.get("temp"), hs=p["hint_strength"])

    def eval_acc(self, c, p):
        T = p.get("temp")
        return sum(
            tc.human_choice
            == np.argmax(apply_temp(action_probs(tc, p["hint_strength"]), T))
            for tc in c
        ) / max(len(c), 1)


# ---------- CV + output ----------


def _mean_se(vals):
    vals = [v for v in vals if np.isfinite(v)]
    if not vals:
        return np.nan, 0.0
    if len(vals) == 1:
        return float(vals[0]), 0.0
    return float(np.mean(vals)), float(np.std(vals, ddof=1) / np.sqrt(len(vals)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--behavioral-dir", type=Path, default=DATA_DIR)
    p.add_argument(
        "--output", type=Path, default=FIT_DIR / "data" / "model_comparison.json"
    )
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--weighted-samples", type=int, default=500)
    p.add_argument(
        "--fit-temp",
        action="store_true",
        help="also fit a softmax temperature (Exp 1); omit for gamma-only (Exp 2)",
    )
    args = p.parse_args()

    print("Loading games...")
    games = load_games(args.behavioral_dir)
    n_games, n_trials = len(games), sum(len(g["choices"]) for g in games)
    n_users = len({g["user_id"] for g in games})
    print(f"  {n_games} games, {n_trials} trials, {n_users} participants")

    _src = "".join(
        (FIT_DIR.parent / f).read_text()
        for f in ("hint_functions.py", "hierarchical.py")
    )
    _key = hashlib.md5(
        (_src + repr(sorted(IDEAL_PRIORS.items()))).encode()
    ).hexdigest()[:8]
    cache_path = (
        FIT_DIR / "data" / f"cache_ws{args.weighted_samples}_s{args.seed}_{_key}.pkl"
    )
    if cache_path.exists():
        print(f"Loading cache: {cache_path}")
        with open(cache_path, "rb") as f:
            caches = pickle.load(f)
    else:
        caches = precompute_games(games, args.weighted_samples, args.seed)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(caches, f)
        print(f"Cached: {cache_path}")
    print(f"  {len(caches)} trial caches")

    models = [RandomModel(), BayesModel(args.fit_temp), HintModel(args.fit_temp)]
    users = np.array(sorted({tc.user_id for tc in caches}))
    kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    results = {
        m.name: {"ll": [], "acc": [], "params": [], "n_test": []} for m in models
    }

    print(f"\n{args.n_folds}-fold CV...")
    for fold, (tr_idx, te_idx) in enumerate(kf.split(users)):
        tr, te = set(users[tr_idx]), set(users[te_idx])
        train = [tc for tc in caches if tc.user_id in tr]
        test = [tc for tc in caches if tc.user_id in te]
        for m in models:
            print(f"  Fold {fold + 1}/{args.n_folds}: {m.name}")
            params = m.fit(train)
            results[m.name]["ll"].append(m.eval_ll(test, params))
            results[m.name]["acc"].append(m.eval_acc(test, params))
            results[m.name]["params"].append(params)
            results[m.name]["n_test"].append(len(test))

    print("\n" + "=" * 90)
    print(
        f"{'Model':<45} {'k':>3} {'N':>7} {'MeanLL':>10} {'SE':>8} {'Acc':>6} {'R²':>7}"
    )
    print("-" * 90)
    output = {
        "n_folds": args.n_folds,
        "seed": args.seed,
        "n_participants": n_users,
        "n_games": n_games,
        "n_trials": n_trials,
        "weighted_samples": args.weighted_samples,
        "models": [],
    }

    for m in models:
        fr = results[m.name]
        tot_ll, tot_n = sum(fr["ll"]), sum(fr["n_test"])
        pfm = [ll / n for ll, n in zip(fr["ll"], fr["n_test"]) if n > 0]
        se_ll = float(np.std(pfm, ddof=1) / np.sqrt(len(pfm))) if len(pfm) > 1 else 0.0
        acc = float(np.mean(fr["acc"]))
        mean_ll = tot_ll / tot_n if tot_n > 0 else np.nan
        r2 = 1 - tot_ll / (tot_n * np.log(0.2)) if tot_n > 0 else np.nan
        print(
            f"{m.name:<45} {m.n_params:>3} {tot_n:>7} {mean_ll:>10.4f} {se_ll:>8.4f} {acc:>6.3f} {r2:>7.4f}"
        )

        entry = {
            "model": m.name,
            "n_params": m.n_params,
            "n_trials": tot_n,
            "total_ll": tot_ll,
            "mean_ll": mean_ll,
            "se_ll": se_ll,
            "accuracy": acc,
            "pseudo_r2": r2,
            "aic": -2 * tot_ll + 2 * m.n_params,
            "bic": -2 * tot_ll + m.n_params * np.log(tot_n) if tot_n > 0 else np.nan,
        }

        parts = []
        for key in ("temp", "hint_strength"):
            if key in fr["params"][0]:
                mn, se = _mean_se([p[key] for p in fr["params"]])
                entry[f"fitted_{key}"] = {"mean": mn, "se": se}
                short = {"temp": "T", "hint_strength": "hs"}[key]
                parts.append(f"{short}={mn:.4f}±{se:.4f}")
        if parts:
            print(f"    {m.name}: {', '.join(parts)}")

        output["models"].append(entry)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, default=float)
    print(f"\nSaved {args.output}")


if __name__ == "__main__":
    main()
