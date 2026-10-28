from __future__ import annotations
import json, subprocess, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
N_DRAWS = 8000
TABLES = HERE
for d in (REPO, REPO / "analysis"):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))
warnings.filterwarnings("ignore")

from constants import TYPE_ORDER  # noqa: E402, F401  (re-exported as C.TYPE_ORDER)
import result2_hint_effects as R2  # noqa: E402  (Fig. 4 loaders)
import result4_hint_timing as R4  # noqa: E402  (Fig. 5 loaders)
import result3_finegrained as R3  # noqa: E402, F401  (Fig. 3 metric functions, used as C.R3)
import freq_lmm as F  # noqa: E402  (OLS/HC3 contrasts for the model simulations)

AGENTS = ["people", "bayesian", "centaur"]
AGENT_LABEL = dict(
    people="People",
    bayesian="Bayesian",
    centaur="Centaur",
    llama="LLaMA",
    centaur_ibt="CentaurIBT",
)
SEED = 20261009
N_BOOT = 10_000
NO_HINT = "No hint."
TYPE_LABELS = R2.TYPE_LABELS
HUMAN_EXP1_CSV = R2.HUMAN_CSV
HUMAN_EXP2_CSV = R4.HUMAN_CSV
HINT_LABELS_CSV = TABLES / "hint_labels.csv"


def git_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip()
    except Exception:
        return "unknown"


def write_table(df, name, extra_meta=None):
    TABLES.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLES / name, index=False)
    meta = dict(
        table=name,
        git=git_hash(),
        seed=SEED,
        n_boot=N_BOOT,
        rows=len(df),
        columns=list(df.columns),
    )
    meta.update(extra_meta or {})
    (TABLES / (Path(name).stem + "_meta.json")).write_text(
        json.dumps(meta, indent=2, default=str)
    )
    print(f"wrote {TABLES / name} ({len(df)} rows)")


# ------------------------------------------------------------------ hint labels
def hint_labels():
    hl = pd.read_csv(HINT_LABELS_CSV)
    hl["exp2_text"] = hl["exp2_text"].fillna("")
    return hl


def hint_id_map():
    hl = hint_labels()
    m = {(r.task_type, r.full_text): r.hint_id for r in hl.itertuples()}
    for r in hl.itertuples():
        if r.exp2_text:
            m[(r.task_type, r.exp2_text)] = r.hint_id
    return m


def hint_order(task_type):
    hl = hint_labels()
    return hl[
        hl.task_type == task_type
    ].full_text.tolist()  # No hint first, then the 5 hints


def games_people_exp1():
    df = F.normalize_regret(
        pd.read_parquet(
            R2._REPO_ROOT
            / "behavioral/instructed_bandits/data/behavioral_clean.parquet"
        )
    )
    g = (
        df.groupby(["user_id", "game_id", "type", "ogHint"])
        .agg(avg_regret=("regret", "mean"))
        .reset_index()
    )
    g["neg_regret"] = -g["avg_regret"]
    return g.rename(columns={"user_id": "subject", "game_id": "game"})


def games_bayes_exp1():
    g = R2._load_bayes_games()
    return g.rename(columns={"sim_id": "game"}).assign(subject=lambda d: d["game"])


def games_centaur_exp1():
    g = R2._load_centaur_games()
    return g.rename(columns={"id": "game"}).assign(subject=lambda d: d["game"])


GAMES_EXP1 = dict(
    people=games_people_exp1, bayesian=games_bayes_exp1, centaur=games_centaur_exp1
)


# ------------------------------------------------------------------ Exp 2 games
def games_people_exp2():
    tr = trials_people_exp2()
    g = (
        tr[tr["trial"] >= tr["hint_timing"]]
        .groupby(["subject", "game", "type", "ogHint", "hint_timing"])
        .agg(avg_regret=("regret", "mean"))
        .reset_index()
    )
    g["neg_regret"] = -g["avg_regret"]
    return g


def trials_people_exp2():
    df = F.normalize_regret(
        pd.read_parquet(
            R4._REPO_ROOT / "behavioral/hint_time/data/behavioral_clean.parquet"
        )
    )
    df["hint_timing"] = df["hint_timing"].astype(int)
    return df.rename(columns={"user_id": "subject", "game_id": "game"})[
        ["subject", "game", "type", "ogHint", "hint_timing", "trial", "regret"]
    ]


def trials_bayes_exp2():
    df = F.normalize_regret(pd.read_parquet(R4.BAYES_TIMING))
    df["ogHint"] = df["ogHint"].map(R4._norm)
    df["hint_timing"] = df["hint_timing"].astype(int)
    df["game"] = (
        df["type"]
        + ":"
        + df["ogHint"]
        + ":"
        + df["hint_timing"].astype(str)
        + ":"
        + df["sim_id"].astype(str)
    )
    return df.rename(columns={"trial_num": "trial"})[
        ["game", "type", "ogHint", "hint_timing", "trial", "regret"]
    ]


def trials_centaur_exp2():
    df = F.normalize_regret(pd.read_parquet(R4.CENTAUR_HT))
    df["ogHint"] = df["ogHints"].map(R4._norm)
    df["hint_timing"] = df["hint_timing"].astype(int)
    df["game"] = df["id"].astype(str) + ":" + df["hint_timing"].astype(str)
    return df.rename(columns={"trial_num": "trial"})[
        ["game", "type", "ogHint", "hint_timing", "trial", "regret"]
    ]


TRIALS_EXP2 = dict(
    people=trials_people_exp2, bayesian=trials_bayes_exp2, centaur=trials_centaur_exp2
)


def games_from_trials(tr, post_onset=False):
    t = tr if not post_onset else tr[tr["trial"] >= tr["hint_timing"]]
    g = (
        t.groupby(["game", "type", "ogHint", "hint_timing"])
        .agg(avg_regret=("regret", "mean"), n_trials=("regret", "size"))
        .reset_index()
    )
    g["neg_regret"] = -g["avg_regret"]
    return g


def games_post_onset(tr, onset):
    onset = int(onset)
    keys = ["game", "type", "ogHint", "hint_timing"]
    g = (
        tr[tr["trial"] >= onset]
        .groupby(keys)["regret"]
        .agg(avg_regret="mean", n_post="size")
        .reset_index()
    )
    if onset > 0:
        g = g.merge(
            tr[tr["trial"] < onset]
            .groupby(keys)["regret"]
            .mean()
            .rename("pre_regret")
            .reset_index(),
            on=keys,
            how="left",
        )
    else:
        g["pre_regret"] = np.nan
    g["neg_regret"] = -g["avg_regret"]
    return g


def exp2_contrast(x, nh, onset, label=""):
    two = pd.concat([x.assign(cond="hint"), nh.assign(cond=NO_HINT)], ignore_index=True)
    cov = ["pre_regret"] if int(onset) > 0 else None
    return F.fit_hint_contrasts(
        two, hint_col="cond", subject_col=None, ref=NO_HINT, label=label, covariates=cov
    ).iloc[0]


# ------------------------------------------------------------------ human effects (draws)
def _se(r, stem):
    col = "se" if stem == "delta" else "no_hint_se"
    v = getattr(r, col, np.nan)
    if v is not None and np.isfinite(v):
        return float(v)
    lo, hi = getattr(r, f"{stem}_lo"), getattr(r, f"{stem}_hi")
    return float((hi - lo) / (2 * 1.96))


def normal_diff(est_a, se_a, est_b, se_b):
    d = float(est_a - est_b)
    se = float(np.sqrt(se_a**2 + se_b**2))
    return d, d - 1.96 * se, d + 1.96 * se, se


def human_exp1_posterior():
    csv = pd.read_csv(HUMAN_EXP1_CSV)
    out = dict(source=None, nohint={}, delta={})
    rng = np.random.default_rng(SEED + 1)
    for r in csv.itertuples():
        if r.ogHint == NO_HINT:
            out["nohint"][r.type] = rng.normal(
                r.no_hint_mean, _se(r, "no_hint"), N_DRAWS
            )
        else:
            out["delta"][(r.type, r.ogHint)] = rng.normal(
                r.delta, _se(r, "delta"), N_DRAWS
            )
    out["source"] = f"normal draws from {HUMAN_EXP1_CSV} (frequentist mixed model)"
    return out


def human_exp2_posterior():
    csv = pd.read_csv(HUMAN_EXP2_CSV)
    out = dict(source=None, nohint={}, delta={})
    rng = np.random.default_rng(SEED + 2)
    for r in (
        csv.itertuples()
    ):  # the Exp 2 CSV has no no-hint rows; the baseline columns sit on every hint row
        if (
            r.type not in out["nohint"] and int(r.hint_timing) == 0
        ):  # per-cell intercept at onset 0
            out["nohint"][r.type] = rng.normal(
                r.no_hint_mean, _se(r, "no_hint"), N_DRAWS
            )
        if r.ogHint != NO_HINT:
            out["delta"][(r.type, r.ogHint, int(r.hint_timing))] = rng.normal(
                r.delta, _se(r, "delta"), N_DRAWS
            )
    out["source"] = f"normal draws from {HUMAN_EXP2_CSV} (frequentist mixed model)"
    return out


# ------------------------------------------------------------------ bootstrap
def rng_for(*key):
    import hashlib

    h = int(hashlib.md5("|".join(str(k) for k in key).encode()).hexdigest()[:8], 16)
    return np.random.default_rng([SEED, h])


def boot_mean(x, rng, n_boot=N_BOOT):
    x = np.asarray(x, float)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    return x[idx].mean(axis=1)


def ci(d):
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def pair_draws(a, b, rng):
    a = np.asarray(a)
    b = np.asarray(b)
    return a[rng.integers(0, len(a), size=len(b))], b
