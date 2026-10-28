import argparse
import sys
from pathlib import Path

import matplotlib.gridspec as gridspec
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

HUMAN_CSV = _REPO_ROOT / "behavioral/instructed_bandits/analysis/hint_effects.csv"
CENTAUR_PARQUET = _REPO_ROOT / "centaur/data/centaur_clean.parquet"
BAYES_PARQUET = SIM_DIR / "bayesian_sim_trials.parquet"
FIGURES_DIR = FIG_DIR

CENTAUR_EFFECTS_CSV = OUT_DIR / "centaur_lmm_effects.csv"
BAYES_EFFECTS_CSV = OUT_DIR / "bayes_lmm_effects.csv"

TYPE_LABELS = PS.TYPE_LABELS
NO_HINT = "No hint."

_THREE_CTX_CANONICAL = (
    "When you see a square, choose G. When you see a circle, choose A. "
    "When you see a triangle, explore all buttons to find the best."
)
_THREE_CTX_PREFIX = "When you see a square, choose"


def _normalize_centaur_hint(h):
    if h.startswith(_THREE_CTX_PREFIX) and "triangle" in h:
        return _THREE_CTX_CANONICAL
    return h


def _load_centaur_games():
    df = pd.read_parquet(CENTAUR_PARQUET)
    df["ogHint"] = df["ogHints"].map(_normalize_centaur_hint)
    df.loc[df["type"] == "bernoulli", "regret"] *= 100
    df = F.normalize_regret(df)
    g = (
        df.groupby(["id", "type", "ogHint"])
        .agg(avg_regret=("regret", "mean"))
        .reset_index()
    )
    g["neg_regret"] = -g["avg_regret"]
    return g


def _load_bayes_games():
    df = F.normalize_regret(pd.read_parquet(BAYES_PARQUET))
    g = (
        df.groupby(["sim_id", "type", "ogHint"])
        .agg(avg_regret=("regret", "mean"))
        .reset_index()
    )
    g["neg_regret"] = -g["avg_regret"]
    return g


def fit_lmm_effects(game_df, subject_col=None):
    rows = []
    for btype in TYPE_ORDER:
        sub = game_df[game_df["type"] == btype]
        print(f"  {btype}: {len(sub)} games", flush=True)
        eff = F.fit_hint_contrasts(
            sub, hint_col="ogHint", subject_col=subject_col, ref=NO_HINT, label=btype
        )
        base = eff.attrs["baseline"]
        rows.append(
            dict(
                type=btype,
                ogHint=NO_HINT,
                delta=0.0,
                delta_lo=0.0,
                delta_hi=0.0,
                se=np.nan,
                z=np.nan,
                p=np.nan,
                n_games=base["n_games"],
                no_hint_mean=base["mean"],
                no_hint_se=base["se"],
                no_hint_lo=base["lo"],
                no_hint_hi=base["hi"],
                method=base["method"],
            )
        )
        for r in eff.itertuples():
            rows.append(
                dict(
                    type=btype,
                    ogHint=r.ogHint,
                    delta=r.delta,
                    delta_lo=r.delta_lo,
                    delta_hi=r.delta_hi,
                    se=r.se,
                    z=r.z,
                    p=r.p,
                    n_games=r.n_games,
                    no_hint_mean=base["mean"],
                    no_hint_se=base["se"],
                    no_hint_lo=base["lo"],
                    no_hint_hi=base["hi"],
                    method=r.method,
                )
            )
    return pd.DataFrame(rows)


NO_HINT_ROW_LABEL = "No hint\n(model vs. people)"


def plot_panel(ax, btype, effects, stars):
    sub = effects[effects["task_type"] == btype]
    people = sub[sub["agent"] == "people"]
    hints = (
        people[people["hint_text"] != NO_HINT]
        .sort_values("rr_own_base")[
            "hint_text"
        ]  # ascending → largest effect at the top
        .tolist()
    )

    star_rows = []
    for yi, hint in enumerate(hints):
        for agent, key in PS.AGENT_KEY.items():
            cell = sub[(sub["agent"] == key) & (sub["hint_text"] == hint)]
            if cell.empty:
                continue
            r = cell.iloc[0]
            PS.draw_point(
                ax,
                agent,
                r["rr_own_base"],
                r["rr_own_lo"],
                r["rr_own_hi"],
                yi + PS.AGENT_STYLE[agent]["dy"],
            )
            if agent == "Human":
                s = stars.get((btype, hint))
                if isinstance(s, str) and s:
                    # right of the widest interval in the row, so the stars never sit on a model bar
                    row_hi = sub[sub["hint_text"] == hint]["rr_own_hi"].max()
                    star_rows.append((yi + PS.AGENT_STYLE[agent]["dy"], row_hi, s))

    h0 = people.iloc[0]
    PS.no_hint_band(ax, h0["human_nohint_lo"], h0["human_nohint_hi"])
    y_nh = -1
    PS.draw_point(
        ax, "Human", 0.0, 0.0, 0.0, y_nh + PS.AGENT_STYLE["Human"]["dy"], interval=False
    )
    for agent in ("Bayesian", "Centaur"):
        nh = sub[(sub["agent"] == PS.AGENT_KEY[agent]) & (sub["hint_text"] == NO_HINT)]
        if not nh.empty:
            r = nh.iloc[0]
            PS.draw_point(
                ax,
                agent,
                r["rr_human_base"],
                r["rr_human_lo"],
                r["rr_human_hi"],
                y_nh + PS.AGENT_STYLE[agent]["dy"],
            )
    ax.axhline(-0.5, color="gray", lw=1.0, ls=":", alpha=0.7, zorder=1)
    ax.set_yticks([y_nh] + list(range(len(hints))))
    ax.set_yticklabels(
        [NO_HINT_ROW_LABEL] + [PS.wrap(PS.hint_label(h), 38) for h in hints]
    )
    ax.tick_params(axis="y", labelsize=PS.SMALL_LABEL)
    ax.set_ylim(y_nh - 0.6, len(hints) - 0.4)
    ax.set_title(TYPE_LABELS[btype], pad=14)
    ax.set_xlabel("Regret reduction (%)")
    ax.margins(x=PS.AXIS_PAD)
    PS.place_stars(ax, star_rows)


def refit_model_effects():
    print("Fitting Centaur per-task OLS/HC3 contrasts …")
    fit_lmm_effects(_load_centaur_games()).to_csv(CENTAUR_EFFECTS_CSV, index=False)
    print(f"Saved: {CENTAUR_EFFECTS_CSV}")
    print("Fitting Bayesian per-task OLS/HC3 contrasts …")
    fit_lmm_effects(_load_bayes_games()).to_csv(BAYES_EFFECTS_CSV, index=False)
    print(f"Saved: {BAYES_EFFECTS_CSV}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--refit",
        action="store_true",
        help="refit the cached model LMM effects before plotting "
        "(then rerun the table pipeline to refresh hint_effects.csv)",
    )
    args = parser.parse_args()
    if args.refit:
        refit_model_effects()

    PS.set_theme(font_scale=2.5)

    print("Loading tables …")
    effects = PS.read_table("hint_effects.csv")
    human = pd.read_csv(HUMAN_CSV)
    stars = {
        (t, h): s for t, h, s in zip(human["type"], human["ogHint"], human["stars"])
    }

    print("Plotting …")
    fig = plt.figure(figsize=(31, 31), dpi=100)
    gs = gridspec.GridSpec(
        4, 2, hspace=0.34, wspace=0.9, left=0.21, right=0.79, top=0.97, bottom=0.085
    )
    for ax_i, btype in enumerate(TYPE_ORDER):
        ax = plt.subplot(gs[ax_i // 2, ax_i % 2])
        plot_panel(ax, btype, effects, stars)

    handles = PS.agent_handles(lines=False)
    PS.bottom_legends(fig, [handles, [PS.band_handle()]], y0=0.026, dy=0.026)
    PS.save(fig, FIGURES_DIR, "result2_hint_effects")


if __name__ == "__main__":
    main()
