import sys
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # analysis/, for exp_paths
from exp_paths import SIM_DIR, FIG_DIR, OUT_DIR
from constants import TYPE_ORDER
import paper_style as PS
import freq_lmm as F

HUMAN_PARQUET = (
    _REPO_ROOT / "behavioral/instructed_bandits/data/behavioral_clean.parquet"
)
BAYES_PARQUET = SIM_DIR / "bayesian_sim_trials.parquet"
CENTAUR_PARQUET = _REPO_ROOT / "centaur/data/centaur_clean.parquet"
FIGURES_DIR = FIG_DIR

TYPE_LABELS = PS.TYPE_LABELS
AGENTS = ["Human", "Bayesian", "Centaur"]

TASK_EFFECTS_CSV = OUT_DIR / "task_effects.csv"


# --- loaders: trial-level frames with columns type, trial, regret, has_hint, game (+ user_id) ---


def load_human():
    df = F.normalize_regret(pd.read_parquet(HUMAN_PARQUET))
    df["has_hint"] = df["ogHint"] != "No hint."
    df["game"] = list(zip(df["user_id"], df["gameId"]))
    return df


def load_bayesian():
    df = F.normalize_regret(
        pd.read_parquet(BAYES_PARQUET).rename(columns={"trial_num": "trial"})
    )
    df["has_hint"] = df["ogHint"] != "No hint."
    df["game"] = list(zip(df["type"], df["ogHint"], df["sim_id"]))
    return df


def load_centaur():
    df = pd.read_parquet(CENTAUR_PARQUET).rename(
        columns={"trial_num": "trial", "ogHints": "ogHint"}
    )
    df["has_hint"] = df["ogHint"] != "No hint."
    df["game"] = df["id"]
    bern = df["type"] == "bernoulli"
    df.loc[bern, "regret"] = df.loc[bern, "regret"] * 100
    return F.normalize_regret(df)


# --- panel a ---


def panel_a_learning_curves(frames, gs):
    inner = gridspec.GridSpecFromSubplotSpec(
        2, 4, subplot_spec=gs, hspace=0.42, wspace=0.22
    )
    axes = []
    for i, btype in enumerate(TYPE_ORDER):
        ax = plt.subplot(inner[i // 4, i % 4])
        axes.append(ax)
        for agent in AGENTS:
            df = frames[agent]
            sub = df[(df["type"] == btype) & (~df["has_hint"])]
            g = sub.groupby("trial")["regret"]
            mean, sem = g.mean(), g.sem()
            ci = PS.Z95 * sem
            color = PS.AGENT_COLORS[agent]
            ax.plot(
                mean.index,
                mean.values,
                color=color,
                lw=PS.LW_CURVE,
                alpha=0.85,
                label=agent,
                zorder=3 if agent == "Human" else 2,
            )
            ax.fill_between(
                mean.index,
                mean - ci,
                mean + ci,
                color=color,
                alpha=PS.BAND_ALPHA_CURVE,
                lw=0,
            )
        ax.set_title(TYPE_LABELS[btype])
        ax.margins(x=0, y=PS.AXIS_PAD)
        if i // 4 == 1:
            ax.set_xlabel("Trial")
        else:
            ax.set_xlabel("")
            ax.set_xticklabels([])
        if i % 4 == 0:
            ax.set_ylabel("Regret (%)")
        else:
            ax.set_ylabel("")
    return axes


# --- panel b ---


def _game_means(df):
    return (
        df.groupby("game")
        .agg(
            type=("type", "first"),
            has_hint=("has_hint", "first"),
            avg_regret=("regret", "mean"),
        )
        .reset_index()
    )


def hint_advantage_human(df):
    game_df = (
        df.groupby(["user_id", "gameId", "type", "has_hint"])
        .agg(avg_regret=("regret", "mean"))
        .reset_index()
    )
    game_df["neg_regret"] = -game_df["avg_regret"]
    game_df["any_hint"] = game_df["has_hint"].astype(int)
    rows = []
    for btype in TYPE_ORDER:
        sub = game_df[game_df["type"] == btype]
        r = F.fit_binary_contrast(sub, "any_hint", subject_col="user_id", label=btype)
        rows.append(
            dict(
                type=btype,
                advantage=r["delta"],
                lo=r["delta_lo"],
                hi=r["delta_hi"],
                se=r["se"],
                z=r["z"],
                p=r["p"],
                n_games=r["n_games"],
                n_subjects=r["n_subjects"],
                sd_subject=r["sd_subject"],
                sd_resid=r["sd_resid"],
                method=r["method"],
            )
        )
    out = F.add_adjusted_p(pd.DataFrame(rows))
    out["stars"] = out["p_adj"].map(F.stars)
    return out.set_index("type").reindex(TYPE_ORDER).reset_index()


def hint_advantage_model(df):
    game_df = _game_means(df)
    game_df["neg_regret"] = -game_df["avg_regret"]
    game_df["any_hint"] = game_df["has_hint"].astype(int)
    rows = []
    for btype in TYPE_ORDER:
        sub = game_df[game_df["type"] == btype]
        r = F.fit_binary_contrast(sub, "any_hint", subject_col=None, label=btype)
        rows.append(
            dict(
                type=btype,
                advantage=r["delta"],
                lo=r["delta_lo"],
                hi=r["delta_hi"],
                se=r["se"],
                z=r["z"],
                p=r["p"],
                n_games=r["n_games"],
                n_subjects=np.nan,
                sd_subject=np.nan,
                sd_resid=r["sd_resid"],
                method=r["method"],
            )
        )
    out = pd.DataFrame(rows)
    out["p_adj"] = np.nan
    out["stars"] = ""
    return out.set_index("type").reindex(TYPE_ORDER).reset_index()


def panel_b_advantage(adv, gs):
    ax = plt.subplot(gs)
    y = np.arange(len(TYPE_ORDER))
    h = 0.3
    offsets = {"Human": -h, "Bayesian": 0.0, "Centaur": +h}
    star_rows = []
    for agent in AGENTS:
        a = adv[agent]
        vals = a["advantage"].values
        ax.barh(
            y + offsets[agent],
            vals,
            height=h,
            color=PS.AGENT_COLORS[agent],
            alpha=PS.BAR_ALPHA,
            edgecolor="black",
            linewidth=1.5,
            label=agent,
        )
        ax.errorbar(
            vals,
            y + offsets[agent],
            xerr=[vals - a["lo"].values, a["hi"].values - vals],
            fmt="none",
            color="black",
            capsize=5,
            lw=2.5,
            capthick=2.0,
        )
        if agent == "Human":
            for yi, (_, row) in zip(y, a.iterrows()):
                if row["stars"]:
                    star_rows.append((yi + offsets[agent], row["hi"], row["stars"]))
    ax.axvline(0, **PS.ZERO_LINE)
    ax.set_yticks(y)
    ax.set_yticklabels([TYPE_LABELS[t] for t in TYPE_ORDER])
    ax.invert_yaxis()
    ax.set_xlabel("Regret reduction (%)")
    PS.tight_x(
        ax,
        min(min(a["lo"].min() for a in adv.values()), 0),
        max(a["hi"].max() for a in adv.values()),
    )
    PS.place_stars(ax, star_rows)
    return ax


def main():
    PS.set_theme()

    print("Loading data …")
    frames = {
        "Human": load_human(),
        "Bayesian": load_bayesian(),
        "Centaur": load_centaur(),
    }
    for k, v in frames.items():
        print(f"  {k}: {len(v)} trials, {v['game'].nunique()} games")

    print("Fitting per-type hint advantage …")
    adv = {
        "Human": hint_advantage_human(frames["Human"]),
        "Bayesian": hint_advantage_model(frames["Bayesian"]),
        "Centaur": hint_advantage_model(frames["Centaur"]),
    }
    table = pd.concat([a.assign(agent=k) for k, a in adv.items()], ignore_index=True)
    table = table[["agent"] + [c for c in table.columns if c != "agent"]]
    table.to_csv(TASK_EFFECTS_CSV, index=False)
    print(f"Saved: {TASK_EFFECTS_CSV}")
    print(
        adv["Human"][
            ["type", "advantage", "lo", "hi", "p", "p_adj", "stars", "method"]
        ].to_string(index=False)
    )

    print("Plotting …")
    fig = plt.figure(figsize=(30, 9.5), dpi=100)
    gs = gridspec.GridSpec(
        1,
        2,
        width_ratios=[2.5, 1],
        wspace=0.34,
        left=0.04,
        right=0.995,
        top=0.93,
        bottom=0.19,
    )
    axes_a = panel_a_learning_curves(frames, gs[0, 0])
    ax_b = panel_b_advantage(adv, gs[0, 1])

    PS.panel_letter(fig, axes_a[0], "a")
    PS.panel_letter(fig, ax_b, "b")
    handles = [
        mlines.Line2D([], [], color=PS.AGENT_COLORS[a], linewidth=PS.LW_CURVE, label=a)
        for a in ("Human", "Bayesian", "Centaur")
    ]
    PS.bottom_legends(fig, [handles], cx=0.5, y0=0.01)
    PS.save(fig, FIGURES_DIR, "result1_learning")


if __name__ == "__main__":
    main()
