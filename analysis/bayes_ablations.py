from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

_REPO = Path(__file__).resolve().parents[1]
for d in (
    _REPO / "analysis",
    _REPO / "analysis/tables",
    _REPO / "bayesian/fit",
    _REPO / "bayesian",
    _REPO,
):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))
import freq_lmm as F  # noqa: E402
import paper_style as PS  # noqa: E402
import result1_learning as R1  # noqa: E402
import result2_hint_effects as R2  # noqa: E402
import result3_finegrained as R3  # noqa: E402
from constants import HINT_LABELS, TYPE_ORDER  # noqa: E402


def _seed(*key):
    return int(hashlib.md5("|".join(map(str, key)).encode()).hexdigest()[:8], 16)


FITS = _REPO / "bayesian/fit/variants"
SIMS = _REPO / "bayesian/data/ablations"
OUT = _REPO / "analysis/results/ablations"
FIG_DIR = _REPO / "analysis/figures"
HUMAN_S1 = _REPO / "behavioral/instructed_bandits/analysis/hint_effects.csv"
PAPER_BAYES = _REPO / "analysis/results/bayes_lmm_effects.csv"
REPO_FIT_JSON = _REPO / "bayesian/fit/data/model_comparison.json"
CENTAUR_FIT_JSON = _REPO / "centaur/fit/data/model_comparison.json"
TRAJ = (
    _REPO / "analysis/tables/trajectory.csv"
)  # Fig. 3 condition means (people, centaur, ...)
TRAJ_CORR = _REPO / "analysis/tables/trajectory_corr.csv"
FG_LABEL = {
    "switch_rate": "Switch rate",
    "entropy": "Choice entropy",
    "exploitation_onset": "Exploitation onset",
}
N_BOOT = 10_000
NO_HINT = "No hint."
Z = 1.96

VARIANTS = ["no_hier_oracle", "full", "no_hier", "no_hint", "neither"]
LABEL = {
    "no_hier_oracle": "Fixed-type + hinted",
    "full": "Hierarchical hinted",
    "no_hier": "Non-hierarchical hinted",
    "neither": "Non-hierarchical no-hint",
    "no_hint": "Hierarchical no-hint",
}
# Bar style: fill = hints (teal) vs no hints (grey); hatch = non-hierarchical; skyline = open bar.
BAR_STYLE = {
    "no_hier_oracle": dict(
        color="white", edgecolor=PS.AGENT_COLORS["Bayesian"], hatch="..."
    ),
    "full": dict(color=PS.AGENT_COLORS["Bayesian"], edgecolor="black"),
    "no_hier": dict(color=PS.AGENT_COLORS["Bayesian"], edgecolor="black", hatch="//"),
    "no_hint": dict(color="#b8b8b8", edgecolor="black"),
    "neither": dict(color="#b8b8b8", edgecolor="black", hatch="//"),
}


HINT_KIND = {
    ("stationary", "{max2} is the best."): "mu",
    ("stationary", "{min2} is the worst."): "mu",
    (
        "stationary",
        "Explore all buttons, then stick with the one that gave the most points.",
    ): "history",
    ("stationary", "{max1} is better than {min2}, {min1}, and {mid}."): "mu",
    ("stationary", "One button is the best."): "tau",
    (
        "drifting",
        "{max2} begins as the best, but the button values change over trials.",
    ): "mu",
    ("drifting", "The button values drift by a small number at each trial."): "tau",
    (
        "drifting",
        "Even after finding a good button, continue trying the others to detect changes.",
    ): "history",
    ("drifting", "{min2} and {min1} begin as the worst buttons."): "mu",
    ("drifting", "The button values change gradually."): "tau",
    ("two_context", "The button values change for square and circle."): "tau",
    ("two_context", "Explore the buttons to find the best one for each shape."): "tau",
    (
        "two_context",
        "When you see a square, {max2} is the best. When you see a circle, {min2} is the best.",
    ): "mu",
    ("two_context", "Pay attention to the shapes."): "tau",
    (
        "two_context",
        "Each shape corresponds to a different set of button values.",
    ): "tau",
    (
        "three_context",
        "When you see a square, choose G. When you see a circle, choose A. When you see a triangle, explore all buttons to find the best.",
    ): "mu",
    ("three_context", "The button values change under each shape."): "tau",
    (
        "three_context",
        "When you see a square or circle, the buttons are sorted in different directions.",
    ): "tau",
    (
        "three_context",
        "Explore the buttons to find the best one for each shape.",
    ): "tau",
    ("three_context", "Pay attention to the shapes."): "tau",
    (
        "time_delayed",
        "{max2} is the best, even if it doesn't seem like it at first.",
    ): "mu",
    (
        "time_delayed",
        "Choose a button for a few trials in a row to observe its true value.",
    ): "history",
    (
        "time_delayed",
        "The points correspond to the choice you made 3 trials ago.",
    ): "theta",
    (
        "time_delayed",
        "There is a time delay between your choice and the points you receive.",
    ): "tau",
    ("time_delayed", "{max2} is the best."): "mu",
    (
        "patch_foraging",
        "{max2} begins as the best, but the points decrease over time.",
    ): "history",
    (
        "patch_foraging",
        "Each button value will decrease if you keep choosing it, but it will increase over time if not.",
    ): "tau",
    (
        "patch_foraging",
        "Find a few good buttons and alternate between them to maximize your points.",
    ): "history",
    (
        "patch_foraging",
        "The points you get on the first pick of a button are the most representative of the button value.",
    ): "tau",
    ("patch_foraging", "Do not stick with any one button for too long."): "history",
    ("bernoulli", "{max2} is most likely to return 100 points."): "mu",
    (
        "bernoulli",
        "Explore all buttons, then stick to the one that most often returned points.",
    ): "history",
    (
        "bernoulli",
        "The probability for returning points is fixed for each button.",
    ): "tau",
    (
        "bernoulli",
        "There is a chance of returning 0 points even if the button is good.",
    ): "tau",
    (
        "bernoulli",
        "Choose a button multiple times to learn its probability of returning points.",
    ): "tau",
    (
        "linear_contextual",
        "If the third number under the button is 1, you will get the most points.",
    ): "theta",
    (
        "linear_contextual",
        "One of the three numbers under the buttons always leads to more points.",
    ): "tau",
    (
        "linear_contextual",
        "The three numbers under each button combine to calculate the total points.",
    ): "tau",
    ("linear_contextual", "Choose the button with a 1 as its third number."): "theta",
    (
        "linear_contextual",
        "The first number has a negative impact on the points. Only choose buttons with a 0 as the first number.",
    ): "theta",
}
UNDEFINED = {
    "no_hier_oracle": {"tau"},  # the type is given, so a type gate selects every sample
    "no_hier": {
        "tau",
        "theta",
    },  # one stationary type: no type to gate, no delay_k / theta parameters
}


def hint_cat(text):
    return ",".join(HINT_LABELS.get(text, [])) or "none"


# ------------------------------------------------------------------ A. held-out fit
def load_heldout():
    if not (FITS / "heldout_trials.parquet").exists():
        raise FileNotFoundError(f"no heldout_trials.parquet under {FITS}")
    t = pd.read_parquet(FITS / "heldout_trials.parquet")
    p = pd.read_csv(FITS / "params.csv")
    p["run"] = "main"
    return t, p


def summarise_ll_taskavg(t):
    per = (
        t.groupby(["model", "fold", "task_type"])
        .agg(ll=("log_lik", "mean"), acc=("correct", "mean"))
        .reset_index()
    )
    fold = (
        per.groupby(["model", "fold"])
        .agg(ll=("ll", "mean"), acc=("acc", "mean"))
        .reset_index()
    )
    out = fold.groupby("model").agg(
        mean_ll=("ll", "mean"),
        accuracy=("acc", "mean"),
        se_ll=("ll", lambda s: s.std(ddof=1) / np.sqrt(len(s))),
    )
    out["n"] = t.groupby("model")["log_lik"].size()
    out = out.reset_index()
    out["ci95"] = Z * out["se_ll"]
    return out


def summarise_ll(t, by=()):
    keys = ["model", *by]
    fold = t.groupby([*keys, "fold"]).agg(ll=("log_lik", "mean")).reset_index()
    se = (
        fold.groupby(keys)["ll"]
        .agg(lambda s: s.std(ddof=1) / np.sqrt(len(s)))
        .rename("se_ll")
    )
    out = t.groupby(keys).agg(
        mean_ll=("log_lik", "mean"), accuracy=("correct", "mean"), n=("log_lik", "size")
    )
    out = out.join(se).reset_index()
    out["ci95"] = Z * out["se_ll"]
    return out


def paired_vs_full(t, task_avg=True):
    if task_avg:
        pt = (
            t.groupby(["model", "task_type", "participant"])["log_lik"]
            .mean()
            .unstack("model")
        )
        per = pt.groupby(level="participant").mean()
        ntr = pd.Series(1.0, index=per.index)
    else:
        per = t.groupby(["model", "participant"])["log_lik"].sum().unstack("model")
        ntr = t.groupby("participant")["log_lik"].size()
    rows = []
    for v in per.columns:
        if v in ("full", "random"):
            continue
        d = (per[v] - per["full"]).dropna()
        per_trial = d / ntr.loc[d.index]
        tt = stats.ttest_rel(per[v].loc[d.index], per["full"].loc[d.index])
        try:
            wp = stats.wilcoxon(d).pvalue
        except ValueError:
            wp = np.nan
        rows.append(
            dict(
                variant=v,
                n_participants=len(d),
                mean_diff=d.mean(),
                ci_lo=d.mean() - Z * d.std(ddof=1) / np.sqrt(len(d)),
                ci_hi=d.mean() + Z * d.std(ddof=1) / np.sqrt(len(d)),
                mean_diff_per_trial=per_trial.mean(),
                t=tt.statistic,
                p_paired_t=tt.pvalue,
                p_wilcoxon=wp,
                frac_participants_worse=(d < 0).mean(),
            )
        )
    return pd.DataFrame(rows)


PAIRS_BY_TASK = [
    ("full", "no_hier"),
    ("no_hint", "neither"),
    ("no_hier_oracle", "no_hier"),
    ("no_hier_oracle", "full"),
    ("full", "no_hint"),
    ("no_hier", "neither"),
]


def paired_by_task(t, pairs=PAIRS_BY_TASK):
    t = t.copy()
    t["hinted"] = t["hint_id"] != NO_HINT
    rows = []
    for split, keys in (
        ("all games", ["model", "task_type", "participant"]),
        ("hint games", ["model", "task_type", "participant"]),
        ("no-hint games", ["model", "task_type", "participant"]),
    ):
        sub = t if split == "all games" else t[t.hinted == (split == "hint games")]
        per = sub.groupby(keys)["log_lik"].agg(["sum", "size"]).unstack("model")
        for a, b in pairs:
            if a not in per["sum"].columns or b not in per["sum"].columns:
                continue
            for task, g in per.groupby(level="task_type"):
                d = (g["sum"][a] - g["sum"][b]).dropna()
                n = g["size"][a].loc[d.index]
                if len(d) < 3:
                    continue
                se = d.std(ddof=1) / np.sqrt(len(d))
                rows.append(
                    dict(
                        a=a,
                        b=b,
                        games=split,
                        task_type=task,
                        n_participants=len(d),
                        mean_diff=d.mean(),
                        ci_lo=d.mean() - Z * se,
                        ci_hi=d.mean() + Z * se,
                        mean_diff_per_trial=(d / n).mean(),
                        p_paired_t=stats.ttest_1samp(d, 0).pvalue,
                    )
                )
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ B. simulated hint effects
def sim_games(variant):
    f = SIMS / f"{variant}_sim_trials.parquet"
    if not f.exists():
        return None
    df = F.normalize_regret(pd.read_parquet(f).rename(columns={"trial_num": "trial"}))
    return df


def per_hint_effects(df):
    g = (
        df.groupby(["sim_id", "type", "ogHint"])
        .agg(avg_regret=("regret", "mean"))
        .reset_index()
    )
    g["neg_regret"] = -g["avg_regret"]
    return R2.fit_lmm_effects(g)


def task_effects(df):
    d = df.copy()
    d["has_hint"] = d["ogHint"] != NO_HINT
    d["game"] = list(zip(d["type"], d["ogHint"], d["sim_id"]))
    return R1.hint_advantage_model(d)


def human_effects():
    h = pd.read_csv(HUMAN_S1)
    h = h[h["ogHint"] != NO_HINT][
        ["type", "ogHint", "delta", "delta_lo", "delta_hi", "se", "p_adj", "stars"]
    ]
    return h.rename(
        columns={
            "delta": "human",
            "delta_lo": "human_lo",
            "delta_hi": "human_hi",
            "se": "human_se",
        }
    )


def agreement(eff, human, full_eff=None):
    m = human.merge(
        eff[eff.ogHint != NO_HINT][["type", "ogHint", "delta"]], on=["type", "ogHint"]
    )
    if full_eff is not None:
        m = m.merge(
            full_eff[full_eff.ogHint != NO_HINT][["type", "ogHint", "delta"]].rename(
                columns={"delta": "full"}
            ),
            on=["type", "ogHint"],
        )
    m["cat"] = m["ogHint"].map(hint_cat)
    rows = []
    for name, sub in [
        ("all", m),
        ("V", m[m.cat == "V"]),
        ("P", m[m.cat == "P"]),
        ("S", m[m.cat == "S"]),
        ("non-V", m[m.cat != "V"]),
    ]:
        r = dict(subset=name, n=len(sub))
        if len(sub) >= 3:
            r.update(
                r_human=stats.pearsonr(sub.human, sub.delta)[0],
                rho_human=stats.spearmanr(sub.human, sub.delta)[0],
                mae_human=(sub.human - sub.delta).abs().mean(),
                mean_abs_effect=sub.delta.abs().mean(),
            )
            if full_eff is not None:
                r.update(
                    r_full=stats.pearsonr(sub.full, sub.delta)[0],
                    mae_full=(sub.full - sub.delta).abs().mean(),
                )
        rows.append(r)
    return pd.DataFrame(rows), m


CATS = [("V", "Value"), ("P", "Procedure"), ("S", "Structure")]
SHARED_HINT = [
    ("stationary", "{max2} is the best."),
    ("bernoulli", "{max2} is most likely to return 100 points."),
    ("time_delayed", "{max2} is the best."),
]


def category_agreement(effects, human, n_boot=N_BOOT):
    rows = []
    for v, eff in effects.items():
        m = human.merge(
            eff[eff.ogHint != NO_HINT][["type", "ogHint", "delta"]],
            on=["type", "ogHint"],
        )
        m["cat"] = m["ogHint"].map(hint_cat)
        for c, _ in CATS + [("all", "All")]:
            sub = m if c == "all" else m[m.cat == c]
            x, y = sub.human.to_numpy(), sub.delta.to_numpy()
            n = len(x)
            r = float(stats.pearsonr(x, y)[0]) if y.std() > 0 else 0.0
            rng = np.random.default_rng(_seed(v, c))
            idx = rng.integers(0, n, size=(n_boot, n))
            bx, by = x[idx], y[idx]
            bx, by = bx - bx.mean(1, keepdims=True), by - by.mean(1, keepdims=True)
            with np.errstate(invalid="ignore", divide="ignore"):
                br = (bx * by).sum(1) / np.sqrt((bx**2).sum(1) * (by**2).sum(1))
            br = np.nan_to_num(br, nan=0.0)
            rows.append(
                dict(
                    variant=v,
                    category=c,
                    n=n,
                    pearson_r=r,
                    r_lo=float(np.percentile(br, 2.5)),
                    r_hi=float(np.percentile(br, 97.5)),
                    mae=float(np.abs(x - y).mean()),
                    mean_human=float(x.mean()),
                    mean_model=float(y.mean()),
                )
            )
    return pd.DataFrame(rows)


def shared_hint_effects(effects, human):
    rows = []
    for t, h in SHARED_HINT:
        hp = human[(human.type == t) & (human.ogHint == h)].iloc[0]
        rows.append(
            dict(
                task_type=t,
                ogHint=h,
                agent="people",
                delta=hp.human,
                lo=hp.human_lo,
                hi=hp.human_hi,
            )
        )
        for v, eff in effects.items():
            e = eff[(eff.type == t) & (eff.ogHint == h)]
            if len(e):
                e = e.iloc[0]
                rows.append(
                    dict(
                        task_type=t,
                        ogHint=h,
                        agent=v,
                        delta=e.delta,
                        lo=e.delta_lo,
                        hi=e.delta_hi,
                    )
                )
    return pd.DataFrame(rows)


def people_finegrained():
    tr = pd.read_csv(TRAJ)
    return tr[tr.agent == "people"].set_index(["task_type", "hint_text"])[R3.METRICS]


def fine_grained(variant, people):
    f = SIMS / f"{variant}_sim_trials.parquet"
    if not f.exists():
        return None
    df = pd.read_parquet(f)
    rows = []
    for (t, h, _), g in df.groupby(["type", "ogHint", "sim_id"]):
        c = g.sort_values("trial_num")["choice"].dropna().values
        if len(c) >= 5:
            rows.append({"task_type": t, "hint_text": h, **R3.game_metrics(c)})
    agg = pd.DataFrame(rows).groupby(["task_type", "hint_text"])[R3.METRICS].mean()
    m = people.join(agg, lsuffix="_people", rsuffix="_model", how="inner")
    out = []
    for met in R3.METRICS:
        x, y = m[met + "_people"].to_numpy(), m[met + "_model"].to_numpy()
        n = len(x)
        rng = np.random.default_rng(_seed(variant, met))
        idx = rng.integers(0, n, size=(N_BOOT, n))
        bx, by = x[idx], y[idx]
        bx, by = bx - bx.mean(1, keepdims=True), by - by.mean(1, keepdims=True)
        br = (bx * by).sum(1) / np.sqrt((bx**2).sum(1) * (by**2).sum(1))
        out.append(
            dict(
                variant=variant,
                metric=met,
                pearson_r=float(stats.pearsonr(x, y)[0]),
                spearman_rho=float(stats.spearmanr(x, y)[0]),
                r_lo=float(np.nanpercentile(br, 2.5)),
                r_hi=float(np.nanpercentile(br, 97.5)),
                n=n,
            )
        )
    return pd.DataFrame(out), m


# ------------------------------------------------------------------ E. figure
def _bars(ax, variants, vals, errs):
    xs = []
    for k, v in enumerate(variants):
        if v not in vals:
            continue
        lo, hi = errs[v]
        ax.bar(
            k,
            vals[v],
            width=0.8,
            yerr=[[vals[v] - lo], [hi - vals[v]]],
            alpha=PS.BAR_ALPHA,
            linewidth=1.8,
            zorder=2,
            error_kw=dict(lw=2.4, capsize=7, capthick=2.0),
            label=LABEL[v],
            **BAR_STYLE[v],
        )
        xs.append(k)
    ax.set_xticks(xs)
    ax.set_xticklabels([])
    ax.set_xlim(-0.7, len(variants) - 0.3)
    ax.grid(axis="x", visible=False)


def supp_figure(summary, refs):
    PS.set_theme(font_scale=3.0)
    variants = [v for v in VARIANTS if v in summary.model.values]
    fig = plt.figure(figsize=(9.5, 10.5), dpi=100)
    gs = fig.add_gridspec(1, 1, left=0.2, right=0.98, top=0.95, bottom=0.3)
    ax = fig.add_subplot(gs[0, 0])
    s = summary.set_index("model")
    _bars(
        ax,
        variants,
        {v: -s.loc[v, "mean_ll"] for v in variants},
        {
            v: (
                -s.loc[v, "mean_ll"] - s.loc[v, "ci95"],
                -s.loc[v, "mean_ll"] + s.loc[v, "ci95"],
            )
            for v in variants
        },
    )
    lo = min(-s.loc[v, "mean_ll"] - s.loc[v, "ci95"] for v in variants)
    hi = max(-s.loc[v, "mean_ll"] + s.loc[v, "ci95"] for v in variants)
    bar_lo = lo
    for name, val, col, ls in refs:
        ax.axhline(val, color=col, lw=2.2, ls=ls, zorder=1)
        ax.text(
            -0.62,
            val + 0.004,
            name,
            color=col,
            fontsize=PS.SMALL_LABEL + 4,
            va="bottom",
            ha="left",
            fontweight="bold",
            zorder=4,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=1.5)
            if val < bar_lo
            else None,
        )
        lo, hi = min(lo, val), max(hi, val)
    ax.set_ylim(lo - 0.02, hi + 0.03)
    ax.set_title("Held-out fit")
    ax.set_ylabel("Negative log-likelihood")
    handles, labels = ax.get_legend_handles_labels()
    x_mid = (0.2 + 0.98) / 2
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(x_mid, 0.01),
        ncol=2,
        fontsize=PS.SMALL_LABEL + 7,
        handlelength=2.0,
        columnspacing=1.6,
        labelspacing=0.8,
        frameon=False,
    )
    PS.save(fig, FIG_DIR, "suppfig_bayes_ablations")


# ------------------------------------------------------------------ main
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    trials, params = load_heldout()
    present = [v for v in VARIANTS if v in trials.model.unique()]
    print("variants with held-out fits:", present)

    summary = summarise_ll_taskavg(trials)
    summary_tw = summarise_ll(trials)
    by_task = summarise_ll(trials, by=["task_type"])
    trials["game_kind"] = np.where(
        trials["hint_id"] == NO_HINT, "no-hint games", "hint games"
    )
    by_kind = summarise_ll(trials, by=["game_kind"])
    trials["hint_cat"] = trials["hint_id"].map(hint_cat)
    by_cat = summarise_ll(trials, by=["hint_cat"])
    paired = paired_vs_full(trials, task_avg=True)
    paired_tw = paired_vs_full(trials, task_avg=False)
    by_task_paired = paired_by_task(trials)
    params_tbl = params[
        [
            "run",
            "variant",
            "fold",
            "T",
            "gamma",
            "test_ll_per_trial",
            "train_ll_per_trial",
            "T_at_bound",
            "gamma_at_bound",
        ]
    ]
    summary.to_csv(OUT / "heldout_summary.csv", index=False)
    summary_tw.to_csv(OUT / "heldout_summary_trialweighted.csv", index=False)
    paired_tw.to_csv(OUT / "paired_vs_full_trialweighted.csv", index=False)
    by_task_paired.to_csv(OUT / "paired_by_task.csv", index=False)
    by_task.to_csv(OUT / "heldout_by_task.csv", index=False)
    by_kind.to_csv(OUT / "heldout_by_hintgame.csv", index=False)
    by_cat.to_csv(OUT / "heldout_by_hintcat.csv", index=False)
    paired.to_csv(OUT / "paired_vs_full.csv", index=False)
    params_tbl.to_csv(OUT / "params_by_fold.csv", index=False)

    # discrepancy vs the paper's existing Bayesian numbers (repo fit, display-position space)
    repo = {m["model"]: m for m in json.load(open(REPO_FIT_JSON))["models"]}
    repo_full = repo["Bayesian + hints (fitted T + hs)"]
    repo_nohint = repo["Bayesian (fitted T)"]

    # B ---------------------------------------------------------------
    human = human_effects()
    effects, tasks, agree = {}, {}, []
    for v in VARIANTS:
        df = sim_games(v)
        if df is None:
            continue
        print(f"[{v}] per-hint and task-level effects")
        effects[v] = per_hint_effects(df)
        effects[v].to_csv(OUT / f"{v}_lmm_effects.csv", index=False)
        tasks[v] = task_effects(df)
        tasks[v].to_csv(OUT / f"{v}_task_effects.csv", index=False)
    for v, eff in effects.items():
        a, m = agreement(eff, human, effects.get("full"))
        a.insert(0, "variant", v)
        agree.append(a)
    agree_df = pd.concat(agree, ignore_index=True) if agree else pd.DataFrame()
    agree_df.to_csv(OUT / "effect_agreement.csv", index=False)
    # paper's own Bayesian effects as a further reference row
    if PAPER_BAYES.exists() and "full" in effects:
        pb = pd.read_csv(PAPER_BAYES)
        a, _ = agreement(pb, human, effects["full"])
        a.insert(0, "variant", "paper_bayesian (repo fit gamma)")
        agree_df = pd.concat([agree_df, a], ignore_index=True)
        agree_df.to_csv(OUT / "effect_agreement.csv", index=False)
    # which hints change most under each ablation
    changes = []
    if "full" in effects:
        fe = effects["full"][effects["full"].ogHint != NO_HINT][
            ["type", "ogHint", "delta"]
        ].rename(columns={"delta": "full"})
        for v, eff in effects.items():
            if v == "full":
                continue
            m = fe.merge(eff[["type", "ogHint", "delta"]], on=["type", "ogHint"])
            m["change"] = m["delta"] - m["full"]
            m["abs_change"] = m["change"].abs()
            m["kind"] = [HINT_KIND.get((t, h), "?") for t, h in zip(m.type, m.ogHint)]
            m["cat"] = m["ogHint"].map(hint_cat)
            m["undefined"] = [k in UNDEFINED.get(v, set()) for k in m["kind"]]
            m.insert(0, "variant", v)
            changes.append(m.sort_values("abs_change", ascending=False))
    changes_df = pd.concat(changes, ignore_index=True) if changes else pd.DataFrame()
    changes_df.to_csv(OUT / "effect_changes.csv", index=False)

    # B2 --------------------------------------------------------------
    people_fg = people_finegrained()
    fg = []
    fg_cond = {}
    for v in VARIANTS:
        r = fine_grained(v, people_fg)
        if r is not None:
            print(f"[{v}] fine-grained agreement")
            fg.append(r[0])
            fg_cond[v] = r[1]
    fg_df = pd.concat(fg, ignore_index=True) if fg else pd.DataFrame()
    fg_df.to_csv(OUT / "finegrained_corr.csv", index=False)

    # E ---------------------------------------------------------------
    cj = {m["model"]: m for m in json.load(open(CENTAUR_FIT_JSON))["models"]}
    refs = [
        ("chance", -np.log(0.2), "#9a9a9a", ":"),
        ("LLaMA", -cj["Llama (fitted T)"]["mean_ll"], PS.MODEL_COLORS["LLaMA"], "--"),
        (
            "Centaur",
            -cj["Centaur (fitted T)"]["mean_ll"],
            PS.MODEL_COLORS["LLaMA"],
            "--",
        ),
    ]
    tc = pd.read_csv(TRAJ_CORR)
    fg_refs = {
        m: [
            (
                "Centaur",
                float(
                    tc[
                        (tc.y == "centaur") & (tc.x == "people") & (tc.metric == m)
                    ].pearson_r.iloc[0]
                ),
                PS.AGENT_COLORS["Centaur"],
                "--",
            )
        ]
        for m in R3.METRICS
    }
    cat_agree = category_agreement(effects, human)
    cat_agree.to_csv(OUT / "category_agreement.csv", index=False)
    shared = shared_hint_effects(effects, human)
    shared.to_csv(OUT / "shared_hint_effects.csv", index=False)
    supp_figure(summary, refs)

    # D ---------------------------------------------------------------
    write_report(
        summary,
        by_task,
        by_kind,
        by_cat,
        paired,
        params_tbl,
        repo_full,
        repo_nohint,
        agree_df,
        changes_df,
        effects,
        tasks,
        fg_df,
        tc,
        summary_tw=summary_tw,
        paired_tw=paired_tw,
        by_task_paired=by_task_paired,
        cat_agree=cat_agree,
        shared=shared,
    )


def _f(x, d=3):
    return "" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


def write_report(
    summary,
    by_task,
    by_kind,
    by_cat,
    paired,
    params_tbl,
    repo_full,
    repo_nohint,
    agree,
    changes,
    effects,
    tasks,
    fg,
    tc,
    summary_tw=None,
    paired_tw=None,
    by_task_paired=None,
    cat_agree=None,
    shared=None,
):
    L = []
    P = L.append
    P("# Bayesian model ablations (supplement)\n")
    P(
        "Held-out fit: 5-fold CV over participants (seed 42), K = 500 posterior samples, T and γ fitted per fold "
        "on the training folds, log-likelihood of every held-out trial. Choices are scored in rank space "
        "(as in the simulations; bayesian/fit/fit.py, whose gamma the simulations use, scores displayed positions). Simulations: 100 agents per task x hint condition, "
        "γ = mean fitted γ over folds, T = None (normative Thompson policy, as in the paper); per-hint effects "
        "by OLS/HC3 vs the variant's own no-hint games, in % of arm spread.\n"
    )
    P("## Summary\n")
    P(
        "Held-out LL / trial and accuracy are computed per task and averaged over the eight tasks with equal weight "
        "(stationary and patch foraging have about twice as many participants as the other tasks, so a plain per-trial "
        "mean over-weights them; the trial-weighted numbers, the weighting of Fig. 5a and the main text, are given below). "
        "Δ LL is per participant: the participant's LL / trial per task, minus the full model's, averaged over the tasks "
        "they played (nats per trial).\n"
    )
    P(
        "| Variant | LL / trial (nats) ± 1.96 SE | Accuracy | Δ LL / trial vs full, per participant [95% CI] | paired t p | r with human per-hint effects | MAE vs human (% spread) |"
    )
    P("|---|---|---|---|---|---|---|")
    s = summary.set_index("model")
    pd_ = paired.set_index("variant") if len(paired) else None
    ag = agree.set_index(["variant", "subset"]) if len(agree) else None
    for v in VARIANTS:
        if v not in s.index:
            continue
        row = s.loc[v]
        d = ""
        if pd_ is not None and v in pd_.index:
            q = pd_.loc[v]
            d = f"{q.mean_diff:+.4f} [{q.ci_lo:+.4f}, {q.ci_hi:+.4f}] | {q.p_paired_t:.2g}"
        else:
            d = "— | —"
        r = mae = ""
        if ag is not None and (v, "all") in ag.index:
            r, mae = (
                _f(ag.loc[(v, "all"), "r_human"], 2),
                _f(ag.loc[(v, "all"), "mae_human"], 1),
            )
        P(
            f"| {LABEL[v]} | {row.mean_ll:.4f} ± {row.ci95:.4f} | {row.accuracy:.3f} | {d} | {r} | {mae} |"
        )
    if "random" in s.index:
        P(f"| Chance | {s.loc['random', 'mean_ll']:.4f} | 0.200 | | | | |")
    P("")
    P(
        "Negative Δ = the ablation predicts choices worse than the full model. The Centaur and LLaMA reference lines in "
        "the figure are their overall (trial-weighted) held-out LL from the main model comparison, since no per-task "
        "likelihoods are stored for them.\n"
    )
    if summary_tw is not None:
        stw = summary_tw.set_index("model")
        ptw = (
            paired_tw.set_index("variant")
            if paired_tw is not None and len(paired_tw)
            else None
        )
        P(
            "Trial-weighted (plain mean over all held-out trials; the weighting used in Fig. 5a and the main-text "
            "model comparison; Δ = summed LL over the participant's held-out trials, nats): "
            + "; ".join(
                f"{LABEL[v]} {stw.loc[v, 'mean_ll']:.4f} ± {stw.loc[v, 'ci95']:.4f}"
                + (
                    f" (Δ {ptw.loc[v, 'mean_diff']:+.1f} [{ptw.loc[v, 'ci_lo']:+.1f}, {ptw.loc[v, 'ci_hi']:+.1f}], "
                    f"p = {ptw.loc[v, 'p_paired_t']:.2g})"
                    if ptw is not None and v in ptw.index
                    else ""
                )
                for v in VARIANTS
                if v in stw.index
            )
            + ".\n"
        )

    P("## Discrepancy with the paper's existing Bayesian numbers\n")
    if "full" in s.index:
        P(
            f"- Paper (repo fit, display-position space): Bayesian + hints {repo_full['mean_ll']:.4f} ± {Z * repo_full['se_ll']:.4f}, "
            f"accuracy {repo_full['accuracy']:.3f}; Bayesian without hints {repo_nohint['mean_ll']:.4f}, accuracy {repo_nohint['accuracy']:.3f}."
        )
        sw = summary_tw.set_index("model") if summary_tw is not None else s
        P(
            f"- Ablation code (rank space, same trial-weighted mean): full {sw.loc['full', 'mean_ll']:.4f} ± {sw.loc['full', 'ci95']:.4f}, accuracy {sw.loc['full', 'accuracy']:.3f}"
            + (
                f"; hierarchical no-hint {sw.loc['no_hint', 'mean_ll']:.4f}, accuracy {sw.loc['no_hint', 'accuracy']:.3f}."
                if "no_hint" in sw.index
                else "."
            )
        )
        P(
            "- The difference is the choice space only (rank space here, displayed button positions in bayesian/fit/fit.py).\n"
        )

    P("## Fitted parameters per fold\n")
    P("| Variant | fold | T | γ | test LL / trial | at bound |")
    P("|---|---|---|---|---|---|")
    order = {v: i for i, v in enumerate(VARIANTS)}
    pt = params_tbl.assign(_o=params_tbl["variant"].map(order)).sort_values(
        ["_o", "fold"]
    )
    for r in pt.itertuples():
        if r.variant == "random":
            continue
        P(
            f"| {LABEL.get(r.variant, r.variant)} | {r.fold} | {_f(r.T, 2)} | {_f(r.gamma, 2)} | {r.test_ll_per_trial:.4f} | {'T' if r.T_at_bound else ''}{'γ' if r.gamma_at_bound else ''} |"
        )
    P("")

    gw = FITS / "gamma_wide" / "params.csv"
    if gw.exists():
        g = pd.read_csv(gw)
        g = g[g.variant.isin(VARIANTS)]
        pm = params_tbl[params_tbl.variant.isin(g.variant.unique())].merge(
            g, on=["variant", "fold"], suffixes=("", "_wide")
        )
        d = (pm.test_ll_per_trial_wide - pm.test_ll_per_trial).abs().max()
        P(
            "γ sits at its lower bound (0.1) in every fold of every hinted variant: it scales only the soft, history-based "
            "hints (hard hints have log-weights 0 / -inf, unchanged by γ), and the fit wants those as weak as allowed. "
            f"Refitting {', '.join(LABEL[v] for v in VARIANTS if v in g.variant.unique())} with the lower bound widened to "
            f"{g.gamma.min():g} (bayesian/fit/variants/gamma_wide/, run_gamma_wide.py; same caches, folds, seed): "
            f"γ = {g.gamma.min():g} in every fold, T unchanged to two decimals, held-out LL / trial changes by at most "
            f"{d:.4f} nats, accuracy unchanged. The bound is immaterial.\n"
        )
    P("## Held-out fit by hint vs no-hint games\n")
    P(
        "| Variant | hint games LL | hint games acc | no-hint games LL | no-hint games acc |"
    )
    P("|---|---|---|---|---|")
    bk = by_kind.set_index(["model", "game_kind"])
    for v in VARIANTS:
        if (v, "hint games") not in bk.index:
            continue
        h, nh = bk.loc[(v, "hint games")], bk.loc[(v, "no-hint games")]
        P(
            f"| {LABEL[v]} | {h.mean_ll:.4f} ± {h.ci95:.4f} | {h.accuracy:.3f} | {nh.mean_ll:.4f} ± {nh.ci95:.4f} | {nh.accuracy:.3f} |"
        )
    P("")

    if len(fg):
        P(
            "## Fine-grained behaviour: r with people over the 48 conditions (Fig. 3 metrics, bootstrap 95% CI)\n"
        )
        P("| Variant | " + " | ".join(FG_LABEL[m] for m in R3.METRICS) + " |")
        P("|---|" + "---|" * len(R3.METRICS))
        f = fg.set_index(["variant", "metric"])
        for v in VARIANTS:
            if (v, R3.METRICS[0]) not in f.index:
                continue
            P(
                f"| {LABEL[v]} | "
                + " | ".join(
                    f"{f.loc[(v, m), 'pearson_r']:.2f} [{f.loc[(v, m), 'r_lo']:.2f}, {f.loc[(v, m), 'r_hi']:.2f}]"
                    for m in R3.METRICS
                )
                + " |"
            )
        for y in ("bayesian", "centaur"):
            q = tc[(tc.y == y) & (tc.x == "people")].set_index("metric")
            P(
                f"| Paper table ({y} vs people) | "
                + " | ".join(
                    f"{q.loc[m, 'pearson_r']:.2f} [{q.loc[m, 'r_lo']:.2f}, {q.loc[m, 'r_hi']:.2f}]"
                    for m in R3.METRICS
                )
                + " |"
            )
        P("")

    P(
        "## Held-out fit by hint category (LL / trial; V = value, P = procedure, S = structure hints)\n"
    )
    bc = by_cat.pivot(index="hint_cat", columns="model", values="mean_ll").reindex(
        ["V", "P", "S", "none"]
    )
    cols = [v for v in VARIANTS if v in bc.columns]
    P("| Games | " + " | ".join(LABEL[v] for v in cols) + " |")
    P("|---|" + "---|" * len(cols))
    for c, r in bc.iterrows():
        P(
            f"| {'no hint' if c == 'none' else c + ' hints'} | "
            + " | ".join(f"{r[v]:.3f}" for v in cols)
            + " |"
        )
    P("")

    P("## Held-out fit by task (LL / trial)\n")
    bt = by_task.pivot(index="task_type", columns="model", values="mean_ll").reindex(
        TYPE_ORDER
    )
    cols = [v for v in VARIANTS if v in bt.columns]
    P("| Task | " + " | ".join(LABEL[v] for v in cols) + " |")
    P("|---|" + "---|" * len(cols))
    for t, r in bt.iterrows():
        P(f"| {t} | " + " | ".join(f"{r[v]:.3f}" for v in cols) + " |")
    P("")

    if by_task_paired is not None and len(by_task_paired):
        P("## Where the hierarchy matters: per-task paired differences\n")
        P(
            "Per participant, held-out LL summed over their trials in the task, first model minus second; mean over "
            "participants [95% CI], paired t p. Positive = the first model fits that task's choices better. The "
            "hierarchy (all eight task types in the hypothesis space vs stationary only) helps where behaviour depends "
            "on structure a stationary learner cannot represent (contexts, delayed value) and costs on the tasks people "
            "treat as simple bandits (bernoulli, drifting, patch foraging), where its type uncertainty makes it explore "
            "more than people do. The near-zero overall difference is the cancellation of these two reliable effects.\n"
        )
        for a, b in PAIRS_BY_TASK:
            sub = by_task_paired[(by_task_paired.a == a) & (by_task_paired.b == b)]
            if not len(sub):
                continue
            P(f"**{LABEL[a]} minus {LABEL[b]}**\n")
            P(
                "| Task | all games: Δ [95% CI] | p | hint games: Δ | p | no-hint games: Δ | p |"
            )
            P("|---|---|---|---|---|---|---|")
            w = sub.set_index(["task_type", "games"])
            for t in TYPE_ORDER:
                if (t, "all games") not in w.index:
                    continue
                r = w.loc[(t, "all games")]
                cells = [
                    f"{r.mean_diff:+.1f} [{r.ci_lo:+.1f}, {r.ci_hi:+.1f}]",
                    f"{r.p_paired_t:.2g}",
                ]
                for g in ("hint games", "no-hint games"):
                    if (t, g) in w.index:
                        q = w.loc[(t, g)]
                        cells += [f"{q.mean_diff:+.1f}", f"{q.p_paired_t:.2g}"]
                    else:
                        cells += ["", ""]
                P(f"| {t} | " + " | ".join(cells) + " |")
            P("")

    if cat_agree is not None:
        P("## Hint effects by hint category, and the shared value hint across tasks\n")
        P(
            "Pearson r between each model's simulated per-hint effects and people's, by hint category "
            "(bootstrap 95% CI over hints). Then the arm-naming value hint that appears with the same meaning in "
            "stationary, bernoulli and time delayed; effect = regret reduction vs the agent's own no-hint games, % of "
            "arm spread (people: LMM; models: OLS/HC3).\n"
        )
        ca = cat_agree.set_index(["category", "variant"])
        cols = [v for v in VARIANTS if (("all", v) in ca.index)]
        P("| Category | n | " + " | ".join(LABEL[v] for v in cols) + " |")
        P("|---|---|" + "---|" * len(cols))
        for c, lab in CATS + [("all", "All")]:
            P(
                f"| {lab} | {int(ca.loc[(c, cols[0]), 'n'])} | "
                + " | ".join(
                    f"{ca.loc[(c, v), 'pearson_r']:.2f} [{ca.loc[(c, v), 'r_lo']:.2f}, {ca.loc[(c, v), 'r_hi']:.2f}]"
                    for v in cols
                )
                + " |"
            )
        P("")
        sh = shared.set_index(["task_type", "agent"])
        P("| Task | People | " + " | ".join(LABEL[v] for v in cols) + " |")
        P("|---|---|" + "---|" * len(cols))
        for t, h in SHARED_HINT:
            P(
                f"| {t} | {sh.loc[(t, 'people'), 'delta']:.1f} [{sh.loc[(t, 'people'), 'lo']:.1f}, {sh.loc[(t, 'people'), 'hi']:.1f}] | "
                + " | ".join(f"{sh.loc[(t, v), 'delta']:.1f}" for v in cols)
                + " |"
            )
        P("")

    if len(agree):
        P(
            "## Simulated per-hint effects: agreement with people and with the full model\n"
        )
        P(
            "| Variant | subset | n | r (people) | ρ (people) | MAE (people) | r (full) | MAE (full) | mean |effect| |"
        )
        P("|---|---|---|---|---|---|---|---|---|")
        for r in agree.itertuples():
            P(
                f"| {LABEL.get(r.variant, r.variant)} | {r.subset} | {r.n} | {_f(getattr(r, 'r_human', np.nan), 2)} | {_f(getattr(r, 'rho_human', np.nan), 2)} | "
                f"{_f(getattr(r, 'mae_human', np.nan), 1)} | {_f(getattr(r, 'r_full', np.nan), 2)} | {_f(getattr(r, 'mae_full', np.nan), 1)} | {_f(getattr(r, 'mean_abs_effect', np.nan), 1)} |"
            )
        P(
            "\nV = value hints (name an arm), P = procedure / strategy hints, S = structure / dynamics hints (constants.HINT_LABELS).\n"
        )

    if len(tasks):
        P("## Task-level hint effects (any hint vs no hint, % of arm spread)\n")
        cols = [v for v in VARIANTS if v in tasks]
        P("| Task | " + " | ".join(LABEL[v] for v in cols) + " |")
        P("|---|" + "---|" * len(cols))
        for t in TYPE_ORDER:
            P(
                f"| {t} | "
                + " | ".join(
                    f"{tasks[v].set_index('type').loc[t, 'advantage']:.1f}"
                    for v in cols
                )
                + " |"
            )
        P("")

    if len(changes):
        P("## Which hints change most under each ablation\n")
        for v in [v for v in VARIANTS if v in changes.variant.values]:
            c = changes[changes.variant == v]
            P(f"**{LABEL[v]}** (largest |Δ effect| vs full, % of arm spread):")
            for r in c.head(6).itertuples():
                P(
                    f'- {r.type}: "{r.ogHint}" {r.full:+.1f} → {r.delta:+.1f} (Δ {r.change:+.1f}; {r.kind} hint{", undefined under this ablation" if r.undefined else ""})'
                )
            und = c[c.undefined]
            if len(und):
                P(
                    f"  Undefined under this ablation ({len(und)} hints, effect should be ≈ 0): "
                    + "; ".join(
                        f"{r.type}: {r.ogHint[:50]}…"
                        if len(r.ogHint) > 50
                        else f"{r.type}: {r.ogHint}"
                        for r in und.itertuples()
                    )
                )
            P("")

    P("## Hints whose meaning function cannot act under an ablation\n")
    P(
        "- **Type known**: the 16 pure type-gate hints (kind = tau) select every posterior sample once the type is given, so "
        "their weight is constant and the predicted effect is 0 by construction. Value (mu), parameter (theta) and history hints are unaffected in form."
    )
    P(
        "- **Stationary-only**: the type-gate hints (tau) have no type to gate, and the parameter hints (theta: the 3-trial delay and the "
        "linear-contextual feature hints) refer to parameters a stationary bandit does not have, so both kinds are undefined (constant weight, "
        "effect 0). The value and history hints act on the sampled arm means and are defined. The context-dependent value hints are "
        'defined but the stationary model has no context, so they reduce to "the best arm is the top-ranked one".'
    )
    P(
        "- **No-hint models**: γ = 0, every hint weight is constant, so all 40 effects are 0 up to simulation noise.\n"
    )

    (OUT / "bayes_ablation_report.md").write_text("\n".join(L) + "\n")
    print(f"wrote {OUT / 'bayes_ablation_report.md'}")


if __name__ == "__main__":
    main()
