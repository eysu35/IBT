from __future__ import annotations

import argparse, dataclasses, json, pickle, subprocess, sys, time, warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for d in (REPO / "bayesian" / "fit", REPO / "bayesian", REPO):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))

import fit as F  # bayesian/fit/fit.py
import hint_functions as HF
import hierarchical as H
from constants import TYPE_ORDER

DATA_DIR = F.DATA_DIR  # behavioral/instructed_bandits/data
SPACES = ("full", "stationary", "oracle")
VARIANTS = {
    # variant: (hypothesis space, hints on?)
    "full": ("full", True),
    "no_hint": ("full", False),
    "no_hier": ("stationary", True),
    "neither": ("stationary", False),
    "no_hier_oracle": ("oracle", True),
}


def git_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------- data + remap
def rank_perm(og_arms):
    og = np.asarray(og_arms, dtype=float)
    return np.argsort(np.argsort(og, kind="stable"), kind="stable")


def load_games_ranked(d=DATA_DIR):
    games = F.load_games(d)
    df = pd.read_parquet(d / "behavioral_clean.parquet").dropna(
        subset=["choice", "type"]
    )
    df["type"] = df["type"].apply(F.normalize_type)
    df = df.sort_values(["user_id", "game_id", "trial"])
    meta = []
    for (uid, gid), g in df.groupby(["user_id", "game_id"]):
        gt = g["type"].iloc[0]
        og, og1 = None, None
        if gt in ("two_context", "three_context"):
            # arms do not drift in the context tasks; `arms` is the per-context layout
            c0, c1 = g[g["contexts"] == 0], g[g["contexts"] == 1]
            og = np.asarray(c0["arms"].iloc[0] if len(c0) else g["arms"].iloc[0], float)
            og1 = np.asarray(c1["arms"].iloc[0], float) if len(c1) else None
        elif gt != "linear_contextual":
            og = np.asarray(g["ogArms"].iloc[0], float)
        meta.append((int(uid), float(gid), gt, og, og1, g["trial"].tolist()))
    assert len(meta) == len(games)
    n_remapped = 0
    for game, (uid, gid, gt, og, og1, trials) in zip(games, meta):
        assert game["user_id"] == uid and game["type"] == gt
        game["game_id"] = gid
        game["trials"] = trials
        if og is not None:
            perm = rank_perm(og)
            if not np.array_equal(perm, np.arange(len(perm))):
                n_remapped += 1
            game["choices"] = [int(perm[c]) for c in game["choices"]]
            # context tasks: context 1 must be the rank-inverted context-0 layout (as in bandits.TwoContextMAB)
            if gt in ("two_context", "three_context") and og1 is not None:
                assert np.array_equal(rank_perm(og1), (len(perm) - 1) - perm), (
                    f"{gt} game {uid}/{gid}: context 1 is not rank-inverted"
                )
    return games, n_remapped


# ---------------------------------------------------------------- hypothesis space
def _one_hot_belief_factory(type_name):
    idx = TYPE_ORDER.index(type_name)

    def make(**kw):
        b = H.default_hierarchical_belief(**kw)
        tp = np.zeros(len(b.type_probs))
        tp[idx] = 1.0
        return dataclasses.replace(b, type_probs=tp)

    return make


def _all_true_mask(t_ids, *type_indices):
    return np.ones(len(t_ids), dtype=bool)


_ORIG_BELIEF = F.default_hierarchical_belief
_ORIG_MASK = HF._type_mask


def configure_space(space, true_type=None):
    F.default_hierarchical_belief = _ORIG_BELIEF
    HF._type_mask = _ORIG_MASK
    if space == "full":
        return "type prior uniform over 8 types; hint functions unpatched"
    if space == "stationary":
        F.default_hierarchical_belief = _one_hot_belief_factory("stationary")
        HF._type_mask = _all_true_mask
        return "type prior one-hot on stationary; hint_functions._type_mask -> all True"
    if space == "oracle":
        assert true_type is not None
        F.default_hierarchical_belief = _one_hot_belief_factory(true_type)
        return (
            f"type prior one-hot on true type ({true_type}); hint functions unpatched"
        )
    raise ValueError(space)


# ---------------------------------------------------------------- precompute
def _compact(tc):
    return F.TrialCache(
        tc.human_choice,
        tc.ts_counts.astype(np.float32),
        tc.best_actions.astype(np.int8),
        None if tc.raw_hint_logw is None else tc.raw_hint_logw.astype(np.float32),
        tc.game_type,
        tc.user_id,
    )


def _worker(args):
    space, gi, game, n_samples, seed = args
    warnings.filterwarnings("ignore")
    configure_space(space, true_type=game["type"])
    shown_hint = game["hint"]
    # per-game seed so the result does not depend on the worker assignment
    per_seed = int((seed * 1_000_003 + gi) % (2**31 - 1))
    caches = F.precompute_games([game], n_samples, per_seed)
    rows = [
        dict(
            user_id=game["user_id"],
            game_id=game["game_id"],
            trial=int(t),
            task_type=game["type"],
            hint_id=shown_hint,
            space=space,
        )
        for t in game["trials"]
    ]
    assert len(rows) == len(caches)
    return gi, [_compact(c) for c in caches], rows


def stage_precompute(space, n_samples, seed, n_jobs, limit=None, out_dir=None):
    out_dir = Path(out_dir) if out_dir else HERE / "variants"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_pkl = out_dir / f"cache_{space}.pkl"
    out_idx = out_dir / f"cache_{space}_index.parquet"
    games, n_remapped = load_games_ranked()
    if limit:
        games = games[:limit]
    print(
        f"[{space}] {len(games)} games, {sum(len(g['choices']) for g in games)} trials, "
        f"{n_remapped} games remapped to rank order; n_jobs={n_jobs}",
        flush=True,
    )
    jobs = [(space, gi, g, n_samples, seed) for gi, g in enumerate(games)]
    t0 = time.time()
    results = []
    if n_jobs == 1:
        for j in jobs:
            results.append(_worker(j))
            if len(results) % 100 == 0:
                print(
                    f"  {len(results)}/{len(jobs)} games {(time.time() - t0) / 60:.1f} min",
                    flush=True,
                )
    else:
        with Pool(n_jobs) as pool:
            for r in pool.imap_unordered(_worker, jobs, chunksize=4):
                results.append(r)
                if len(results) % 200 == 0:
                    print(
                        f"  {len(results)}/{len(jobs)} games {(time.time() - t0) / 60:.1f} min",
                        flush=True,
                    )
    results.sort(key=lambda r: r[0])
    caches = [c for _, cs, _ in results for c in cs]
    index = pd.DataFrame([row for _, _, rows in results for row in rows])
    with open(out_pkl, "wb") as f:
        pickle.dump(caches, f, protocol=pickle.HIGHEST_PROTOCOL)
    index.to_parquet(out_idx, index=False)
    meta = dict(
        space=space,
        n_samples=n_samples,
        seed=seed,
        n_games=len(games),
        n_trials=len(caches),
        n_games_remapped=n_remapped,
        git=git_hash(),
        minutes=(time.time() - t0) / 60,
        patch=configure_space(space, true_type="stationary"),
    )
    (out_dir / f"cache_{space}_meta.json").write_text(json.dumps(meta, indent=2))
    print(
        f"[{space}] done: {len(caches)} trials in {meta['minutes']:.1f} min -> {out_pkl}",
        flush=True,
    )


# ---------------------------------------------------------------- fit
def folds_table(users, n_folds, seed):
    users = np.array(sorted(users))
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    rows = []
    for fold, (_, te) in enumerate(kf.split(users)):
        rows += [dict(user_id=int(u), fold=fold) for u in users[te]]
    return pd.DataFrame(rows)


def _trial_rows(caches, idx, hs, T, model, fold):
    out = []
    for tc, (_, r) in zip(caches, idx.iterrows()):
        p = F.apply_temp(F.action_probs(tc, hs), T)
        ll = float(np.log(np.clip(p[tc.human_choice], 1e-12, 1.0)))
        out.append(
            dict(
                model=model,
                fold=fold,
                participant=r.user_id,
                game=r.game_id,
                trial=r.trial,
                task_type=r.task_type,
                hint_id=r.hint_id,
                log_lik=ll,
                correct=bool(np.argmax(p) == tc.human_choice),
                p_choice=float(p[tc.human_choice]),
            )
        )
    return out


def cv_variant(variant, caches, index, folds, fit_temp=True):
    space, hints_on = VARIANTS[variant]
    model = F.HintModel(fit_temp) if hints_on else F.BayesModel(fit_temp)
    user_fold = dict(zip(folds.user_id, folds.fold))
    fold_of = index.user_id.map(user_fold).to_numpy()
    params_rows, trial_rows = [], []
    for fold in sorted(folds.fold.unique()):
        tr = [c for c, f in zip(caches, fold_of) if f != fold]
        te_mask = fold_of == fold
        te = [c for c, m in zip(caches, te_mask) if m]
        te_idx = index[te_mask]
        t0 = time.time()
        p = model.fit(tr)
        T, hs = p.get("temp"), p.get("hint_strength")
        train_ll = F.total_ll(tr, T=T, hs=hs)
        test_ll = F.total_ll(te, T=T, hs=hs)
        params_rows.append(
            dict(
                variant=variant,
                model=model.name,
                fold=fold,
                gamma=hs,
                T=T,
                train_ll=train_ll,
                train_ll_per_trial=train_ll / len(tr),
                test_ll=test_ll,
                test_ll_per_trial=test_ll / len(te),
                n_train=len(tr),
                n_test=len(te),
                T_at_bound=T is not None
                and (
                    np.isclose(T, F.T_B[0], atol=1e-3)
                    or np.isclose(T, F.T_B[1], atol=1e-3)
                ),
                gamma_at_bound=hs is not None
                and (
                    np.isclose(hs, F.HS_B[0], atol=1e-3)
                    or np.isclose(hs, F.HS_B[1], atol=1e-3)
                ),
                seconds=time.time() - t0,
            )
        )
        trial_rows += _trial_rows(te, te_idx, hs, T, variant, fold)
        print(
            f"  {variant:15s} fold {fold}: T={T} gamma={hs} test LL/trial={test_ll / len(te):.4f} ({time.time() - t0:.0f}s)",
            flush=True,
        )
    return params_rows, trial_rows


def stage_fit(n_folds, seed, variants, out_dir=None):
    out_dir = Path(out_dir) if out_dir else HERE / "variants"
    caches, indexes = {}, {}
    for space in sorted({VARIANTS[v][0] for v in variants}):
        pkl, idx = (
            out_dir / f"cache_{space}.pkl",
            out_dir / f"cache_{space}_index.parquet",
        )
        assert pkl.exists() and idx.exists(), (
            f"missing {pkl}; run precompute --space {space}"
        )
        with open(pkl, "rb") as f:
            caches[space] = pickle.load(f)
        indexes[space] = pd.read_parquet(idx)
        assert len(caches[space]) == len(indexes[space])
    ref = next(iter(indexes.values()))[["user_id", "game_id", "trial"]]
    for sp, ix in indexes.items():
        assert ix[["user_id", "game_id", "trial"]].equals(ref), (
            f"trial set of {sp} differs from the others"
        )
    folds = folds_table(ref.user_id.unique(), n_folds, seed)
    folds.to_csv(out_dir / "folds.csv", index=False)
    all_params, all_trials = [], []
    # Random baseline on the same trials
    user_fold = dict(zip(folds.user_id, folds.fold))
    for fold in sorted(folds.fold.unique()):
        m = ref[ref.user_id.map(user_fold) == fold]
        all_params.append(
            dict(
                variant="random",
                model="Random (uniform)",
                fold=fold,
                gamma=None,
                T=None,
                train_ll=np.nan,
                train_ll_per_trial=np.log(0.2),
                test_ll=len(m) * np.log(0.2),
                test_ll_per_trial=np.log(0.2),
                n_train=len(ref) - len(m),
                n_test=len(m),
                T_at_bound=False,
                gamma_at_bound=False,
                seconds=0.0,
            )
        )
        ixm = next(iter(indexes.values()))[ref.user_id.map(user_fold) == fold]
        all_trials += [
            dict(
                model="random",
                fold=fold,
                participant=r.user_id,
                game=r.game_id,
                trial=r.trial,
                task_type=r.task_type,
                hint_id=r.hint_id,
                log_lik=np.log(0.2),
                correct=False,
                p_choice=0.2,
            )
            for _, r in ixm.iterrows()
        ]
    for v in variants:
        space, _ = VARIANTS[v]
        configure_space(
            space, true_type="stationary"
        )  # only affects precompute; harmless here
        pr, tr = cv_variant(v, caches[space], indexes[space], folds)
        all_params += pr
        all_trials += tr
    params = pd.DataFrame(all_params)
    params.to_csv(out_dir / "params.csv", index=False)
    trials = pd.DataFrame(all_trials)
    trials.to_parquet(out_dir / "heldout_trials.parquet", index=False)
    summ = (
        trials.groupby("model")
        .agg(
            mean_ll=("log_lik", "mean"),
            accuracy=("correct", "mean"),
            n=("log_lik", "size"),
        )
        .join(
            trials.groupby(["model", "fold"])
            .log_lik.mean()
            .groupby("model")
            .agg(se_ll=lambda s: s.std(ddof=1) / np.sqrt(len(s)))
        )
    )
    summ = summ.reset_index()
    print("\n" + summ.to_string(index=False))
    (out_dir / "fit_summary.json").write_text(
        json.dumps(
            dict(
                git=git_hash(),
                seed=seed,
                n_folds=n_folds,
                variants=variants,
                bounds=dict(T=F.T_B, gamma=F.HS_B),
                at_bound=params[params.T_at_bound | params.gamma_at_bound][
                    ["variant", "fold", "T", "gamma"]
                ].to_dict("records"),
                summary=summ.to_dict("records"),
            ),
            indent=2,
            default=float,
        )
    )
    print(
        f"wrote {out_dir / 'params.csv'}, {out_dir / 'heldout_trials.parquet'}, {out_dir / 'folds.csv'}"
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["precompute", "fit"])
    p.add_argument("--space", choices=SPACES)
    p.add_argument("--n-samples", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--n-jobs", type=int, default=1)
    p.add_argument("--limit", type=int, default=None, help="first N games (smoke test)")
    p.add_argument(
        "--variants",
        nargs="+",
        default=["full", "no_hint", "no_hier", "neither", "no_hier_oracle"],
    )
    p.add_argument(
        "--out-dir",
        default=None,
        help="defaults to this directory; use a scratch dir for smoke tests",
    )
    a = p.parse_args()
    warnings.filterwarnings("ignore")
    if a.stage == "precompute":
        assert a.space, "--space required"
        stage_precompute(a.space, a.n_samples, a.seed, a.n_jobs, a.limit, a.out_dir)
    else:
        stage_fit(a.n_folds, a.seed, a.variants, a.out_dir)


if __name__ == "__main__":
    main()
