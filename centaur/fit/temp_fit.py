from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp
from tqdm import tqdm
from sklearn.model_selection import KFold

FIT_DIR = Path(__file__).resolve().parent
CENTAUR_DIR = FIT_DIR.parent
REPO_ROOT = CENTAUR_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from constants import normalize_type, TYPE_ORDER

DATA_DIR = FIT_DIR / "data"
DEFAULT_SIM_DIR = CENTAUR_DIR / "data"
BEHAVIORAL_DIR = REPO_ROOT / "behavioral" / "instructed_bandits" / "data"
REF_TEMP = 0.6
MODEL_KEYS = ["centaur", "llama", "centaur_IBT"]
NICE = {"centaur": "Centaur", "llama": "Llama", "centaur_IBT": "CentaurIBT"}


# --- Data loading ---


def _to_rank(choice, arms):
    arms = np.asarray(arms, dtype=float).ravel()
    return int(np.where(np.argsort(-arms) == choice)[0][0])


def _probs_to_rank(probs, arms):
    return np.asarray(probs, dtype=float).ravel()[
        np.argsort(-np.asarray(arms, dtype=float).ravel())
    ]


def _hint(row):
    h = row.get("ogHints", row.get("ogHint", row.get("hint", "")))
    return str(h[0]) if isinstance(h, (list, np.ndarray)) and len(h) else str(h)


def load_human_trials(behavioral_dir):
    df = pd.read_parquet(behavioral_dir / "behavioral_clean.parquet")
    df["hint"] = df["ogHint"].astype(str)
    df["choice"] = df["choice"].astype(int)
    if "arms" in df.columns:
        df["choice"] = df.apply(
            lambda r: (
                _to_rank(r["choice"], r["arms"])
                if r["arms"] is not None and len(r["arms"]) > 0
                else r["choice"]
            ),
            axis=1,
        )
    return df[
        [
            "user_id",
            "game_id",
            "type",
            "hint",
            "trial",
            "choice",
            *[c for c in ["arms", "regret"] if c in df.columns],
        ]
    ].copy()


def _norm_rank(p, arms):
    p = np.clip(np.asarray(p, dtype=float).ravel(), 1e-10, 1.0)
    p /= p.sum()
    return _probs_to_rank(p, arms) if arms is not None else p


def load_model_mean_probs(sim_dir, model):
    rows = []
    for path in Path(sim_dir).glob("**/*_clean.parquet"):
        if not path.stem.lower().startswith(model.lower()):
            continue
        df = pd.read_parquet(path)
        df["type"] = df["type"].apply(normalize_type)
        tcol = "trial_num" if "trial_num" in df.columns else "trial"
        has_arms = "arms" in df.columns
        for _, row in df.iterrows():
            arr = np.asarray(row["probs"], dtype=float)
            arms = (
                np.asarray(row["arms"], dtype=float).ravel()
                if has_arms and row["arms"] is not None
                else None
            )
            hint = _hint(row)
            if arr.ndim == 1:
                rows.append(
                    {
                        "type": row["type"],
                        "hint": hint,
                        "trial": int(row[tcol]),
                        "probs": _norm_rank(arr, arms),
                    }
                )
            else:
                for t in range(arr.shape[0]):
                    a_t = arms[t] if arms is not None and arms.ndim == 2 else arms
                    rows.append(
                        {
                            "type": row["type"],
                            "hint": hint,
                            "trial": t,
                            "probs": _norm_rank(arr[t].ravel(), a_t),
                        }
                    )
    if not rows:
        return pd.DataFrame(columns=["type", "hint", "trial", "mean_probs"])
    return (
        pd.DataFrame(rows)
        .groupby(["type", "hint", "trial"])["probs"]
        .apply(lambda x: np.array(x.tolist()).mean(axis=0))
        .reset_index()
        .rename(columns={"probs": "mean_probs"})
    )


def load_model_mean_regret(sim_dir, model):
    rows = []
    for path in Path(sim_dir).glob("**/*_clean.parquet"):
        if not path.stem.lower().startswith(model.lower()):
            continue
        df = pd.read_parquet(path)
        df["type"] = df["type"].apply(normalize_type)
        tcol = "trial_num" if "trial_num" in df.columns else "trial"
        for _, row in df.iterrows():
            reg = float(row["regret"]) * (100.0 if row["type"] == "bernoulli" else 1.0)
            rows.append(
                {
                    "type": row["type"],
                    "hint": _hint(row),
                    "trial": int(row[tcol]),
                    "regret": reg,
                }
            )
    if not rows:
        return pd.DataFrame(columns=["type", "hint", "trial", "mean_regret"])
    return (
        pd.DataFrame(rows)
        .groupby(["type", "hint", "trial"])["regret"]
        .mean()
        .reset_index()
        .rename(columns={"regret": "mean_regret"})
    )


# --- Softmax & fitting ---


def softmax_log_prob(probs, choice, temp):
    log_p = np.log(np.clip(np.asarray(probs, dtype=float).ravel(), 1e-12, 1.0)) * (
        REF_TEMP / max(temp, 1e-6)
    )
    return log_p[choice] - logsumexp(log_p)


def total_ll(choices, probs_list, temp):
    return sum(softmax_log_prob(p, int(a), temp) for a, p in zip(choices, probs_list))


def fit_temp(choices, probs_list, bounds=(0.01, 2.0)):
    return float(
        minimize_scalar(
            lambda t: -total_ll(choices, probs_list, t), bounds=bounds, method="bounded"
        ).x
    )


# --- Models ---


class RandomModel:
    name = "Random (uniform)"
    n_params = 0
    probs_col = None

    def fit(self, df):
        return {}

    def eval_ll(self, df, p):
        return len(df) * np.log(0.2)

    def eval_accuracy(self, df, p):
        return 0.2


class TempModel:
    def __init__(self, key, mode="fixed"):
        self.key, self.mode = key, mode
        self.probs_col = f"mean_probs_{key}"
        labels = {
            "fixed": f"(T={REF_TEMP})",
            "single": "(fitted T)",
            "per_type": "(T per type)",
        }
        self.name = f"{NICE[key]} {labels[mode]}"
        self.n_params = 0 if mode == "fixed" else (1 if mode == "single" else 0)

    def fit(self, df):
        pl = list(df[self.probs_col].values)
        c = df["choice"].values
        if self.mode == "fixed":
            return {"temp": REF_TEMP}
        if self.mode == "single":
            return {"temp": fit_temp(c, pl)}
        temps = {}
        for typ, g in df.groupby("type"):
            temps[typ] = fit_temp(g["choice"].values, list(g[self.probs_col].values))
        self.n_params = len(temps)
        return {"temps": temps}

    def _get_temp(self, params, typ=None):
        if "temps" in params:
            return params["temps"].get(typ, REF_TEMP)
        return params.get("temp", REF_TEMP)

    def eval_ll(self, df, p):
        if self.mode != "per_type":
            return total_ll(
                df["choice"].values, list(df[self.probs_col].values), self._get_temp(p)
            )
        return sum(
            total_ll(
                g["choice"].values,
                list(g[self.probs_col].values),
                self._get_temp(p, typ),
            )
            for typ, g in df.groupby("type")
        )

    def eval_accuracy(self, df, p):
        correct = 0
        for _, r in df.iterrows():
            t = self._get_temp(p, r["type"])
            log_p = np.log(
                np.clip(np.asarray(r[self.probs_col], dtype=float).ravel(), 1e-12, 1.0)
            ) * (REF_TEMP / max(t, 1e-6))
            correct += int(r["choice"]) == np.argmax(log_p)
        return correct / len(df)


# --- K-fold CV ---


def run_kfold(human, models, n_folds, seed):
    users = human["user_id"].unique()
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fr = {
        m.name: {"ll": [], "accuracy": [], "params": [], "n_test": []} for m in models
    }

    for _, (train_idx, test_idx) in enumerate(
        tqdm(kf.split(users), total=n_folds, desc="Folds")
    ):
        train_df = human[human["user_id"].isin(set(users[train_idx]))]
        test_df = human[human["user_id"].isin(set(users[test_idx]))]
        for m in models:
            if m.probs_col:
                tv = train_df[train_df[m.probs_col].notna()]
                ev = test_df[test_df[m.probs_col].notna()]
            else:
                tv, ev = train_df, test_df
            if tv.empty or ev.empty:
                fr[m.name]["ll"].append(np.nan)
                fr[m.name]["accuracy"].append(np.nan)
                fr[m.name]["params"].append({})
                fr[m.name]["n_test"].append(0)
                continue
            params = m.fit(tv)
            fr[m.name]["ll"].append(m.eval_ll(ev, params))
            fr[m.name]["accuracy"].append(m.eval_accuracy(ev, params))
            fr[m.name]["params"].append(params)
            fr[m.name]["n_test"].append(len(ev))
    return fr


def regret_comparison(human, model_regret):
    hr = (
        human.groupby(["type", "hint", "trial"])["regret"]
        .mean()
        .reset_index(name="human_regret")
    )
    merged = hr.merge(model_regret, on=["type", "hint", "trial"], how="inner")
    if merged.empty:
        return {"correlation": np.nan, "rmse": np.nan, "n_points": 0}
    corr = float(merged["human_regret"].corr(merged["mean_regret"]))
    rmse = float(
        np.sqrt(((merged["human_regret"] - merged["mean_regret"]) ** 2).mean())
    )
    return {"correlation": corr, "rmse": rmse, "n_points": len(merged)}


def _mean_se(vals):
    vals = [v for v in vals if not np.isnan(v)]
    if not vals:
        return np.nan, 0.0
    m = float(np.mean(vals))
    s = float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
    return m, s


# --- Main ---


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sim-dir", type=Path, default=DEFAULT_SIM_DIR)
    p.add_argument("--behavioral-dir", type=Path, default=BEHAVIORAL_DIR)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--output", type=Path, default=FIT_DIR / "data" / "model_comparison.json"
    )
    args = p.parse_args()

    print("Loading human trials...")
    human = load_human_trials(args.behavioral_dir)
    n_total, n_users = len(human), human["user_id"].nunique()
    print(f"  {n_total} trials, {n_users} participants")

    for key in MODEL_KEYS:
        probs = load_model_mean_probs(args.sim_dir, key)
        human = human.merge(
            probs.rename(columns={"mean_probs": f"mean_probs_{key}"}),
            on=["type", "hint", "trial"],
            how="left",
        )

    has_any = human[[f"mean_probs_{k}" for k in MODEL_KEYS]].notna().any(axis=1)
    human = human[has_any].copy()
    print(f"  Matched {len(human)}/{n_total} trials")

    models = [RandomModel()]
    for key in MODEL_KEYS:
        models += [TempModel(key, m) for m in ("fixed", "single", "per_type")]

    print(f"\nRunning {args.n_folds}-fold CV...")
    fold_results = run_kfold(human, models, args.n_folds, args.seed)

    # Aggregate & print
    print("\n" + "=" * 80)
    results_table = []
    for m in models:
        fr = fold_results[m.name]
        valid = [
            (ll, n)
            for ll, n in zip(fr["ll"], fr["n_test"])
            if not np.isnan(ll) and n > 0
        ]
        if not valid:
            results_table.append(
                {
                    "model": m.name,
                    "n_params": m.n_params,
                    "n_matched": 0,
                    "total_ll": np.nan,
                    "mean_ll": np.nan,
                    "se_ll": np.nan,
                    "accuracy": np.nan,
                    "pseudo_r2": np.nan,
                    "aic": np.nan,
                    "bic": np.nan,
                }
            )
            continue
        lls, ns = zip(*valid)
        tot_ll, tot_n = sum(lls), sum(ns)
        pfm = [ll / n for ll, n in valid]
        se = np.std(pfm, ddof=1) / np.sqrt(len(pfm)) if len(pfm) > 1 else 0.0
        ll_rand = tot_n * np.log(0.2)
        accs = [x for x in fr["accuracy"] if not np.isnan(x)]
        results_table.append(
            {
                "model": m.name,
                "n_params": m.n_params,
                "n_matched": tot_n,
                "total_ll": tot_ll,
                "mean_ll": tot_ll / tot_n,
                "se_ll": se,
                "accuracy": np.mean(accs) if accs else np.nan,
                "pseudo_r2": 1 - tot_ll / ll_rand if ll_rand else np.nan,
                "aic": -2 * tot_ll + 2 * m.n_params,
                "bic": -2 * tot_ll + m.n_params * np.log(tot_n) if tot_n else np.nan,
            }
        )

    print(
        f"{'Model':<30} {'k':>3} {'N':>7} {'MeanLL':>10} {'SE':>8} {'Acc':>6} {'R²':>7} {'AIC':>12}"
    )
    print("-" * 90)
    for r in results_table:
        print(
            f"{r['model']:<30} {r['n_params']:>3} {r['n_matched']:>7} {r['mean_ll']:>10.4f} "
            f"{r['se_ll']:>8.4f} {r['accuracy']:>6.3f} {r['pseudo_r2']:>7.4f} {r['aic']:>12.1f}"
        )

    # Per-type temps
    print("\nPER-TYPE TEMPERATURES:")
    for m in models:
        if not isinstance(m, TempModel):
            continue
        if m.mode == "single":
            mn, se = _mean_se(
                [p.get("temp", np.nan) for p in fold_results[m.name]["params"] if p]
            )
            if not np.isnan(mn):
                print(f"  {m.name}: T = {mn:.4f} +/- {se:.4f}")
        elif m.mode == "per_type":
            all_t = [p.get("temps", {}) for p in fold_results[m.name]["params"] if p]
            keys = set().union(*all_t) if all_t else set()
            if keys:
                print(f"  {m.name}:")
                for typ in TYPE_ORDER:
                    if typ not in keys:
                        continue
                    mn, se = _mean_se([t.get(typ, np.nan) for t in all_t])
                    if not np.isnan(mn):
                        print(f"    {typ:<25} T = {mn:.4f} +/- {se:.4f}")

    # Regret
    print("\nREGRET:")
    regret_results = {}
    for key in MODEL_KEYS:
        mr = load_model_mean_regret(args.sim_dir, key)
        if mr.empty:
            continue
        rc = regret_comparison(human, mr)
        regret_results[key] = rc
        print(
            f"  {key}: corr={rc['correlation']:.4f}, RMSE={rc['rmse']:.4f} ({rc['n_points']} pts)"
        )

    # Save JSON
    output = {
        "n_folds": args.n_folds,
        "seed": args.seed,
        "n_participants": int(n_users),
        "n_trials_total": int(n_total),
        "n_trials_matched": int(len(human)),
        "ref_temp": REF_TEMP,
        "models": [],
        "regret_comparison": regret_results,
    }
    for r in results_table:
        entry = {
            k: (
                None
                if isinstance(v, float) and np.isnan(v)
                else int(v)
                if isinstance(v, (int, np.integer))
                else float(v)
                if isinstance(v, (float, np.floating))
                else v
            )
            for k, v in r.items()
        }
        m_obj = next(m for m in models if m.name == r["model"])
        if isinstance(m_obj, TempModel) and m_obj.mode == "per_type":
            all_t = [
                p.get("temps", {}) for p in fold_results[m_obj.name]["params"] if p
            ]
            entry["per_type_temps"] = {}
            for typ in sorted(set().union(*all_t)):
                mn, se = _mean_se([t.get(typ, np.nan) for t in all_t])
                if not np.isnan(mn):
                    entry["per_type_temps"][typ] = {"mean": mn, "se": se}
        elif isinstance(m_obj, TempModel) and m_obj.mode == "single":
            mn, se = _mean_se(
                [p.get("temp", np.nan) for p in fold_results[m_obj.name]["params"] if p]
            )
            if not np.isnan(mn):
                entry["fitted_temp"] = {"mean": mn, "se": se}
        output["models"].append(entry)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved {args.output}")


if __name__ == "__main__":
    main()
