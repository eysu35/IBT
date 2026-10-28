import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "analysis"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import freq_lmm as F

DATA_PATH = _REPO_ROOT / "behavioral/hint_time/data/behavioral_clean.parquet"
OUT_DIR = Path(__file__).resolve().parent

TYPE_ORDER = ["stationary", "patch_foraging"]
TIMING_ORDER = ["0", "4", "8", "12"]
NO_HINT = "No hint."
PRE = "pre_regret"
CAT_COLORS = {"V": "#d62728", "P": "#1f77b4", "S": "#2ca02c", "none": "#7f7f7f"}
HINT_CATS = {
    "{max2} is the best.": "V",
    "{max1} and {max2} are better than the others.": "V",
    "Avoid {min1} and {min2}.": "V",
    "{max2} begins as the best, but the points decrease over time.": "V",
    "Explore all buttons, then stick with the one that gave the most points.": "P",
    "Do not stick with any one button for too long.": "P",
    "Find a few good buttons and alternate between them to maximize your points.": "P",
    "One button is the best.": "S",
    "Each button value will decrease if you keep choosing it, but it will increase over time if not.": "S",
    "The points you get on the first pick of a button are the most representative of the button value.": "S",
    "No hint.": "none",
}


def load_trials():
    return F.normalize_regret(pd.read_parquet(DATA_PATH))


def games_post_onset(
    trials, onset, keys=("user_id", "game_id", "type", "ogHint", "hint_timing")
):
    onset = int(onset)
    keys = list(keys)
    post = (
        trials[trials["trial"] >= onset]
        .groupby(keys)["regret"]
        .agg(avg_regret="mean", n_post="size")
    )
    g = post.reset_index()
    if onset > 0:
        pre = trials[trials["trial"] < onset].groupby(keys)["regret"].mean().rename(PRE)
        g = g.merge(pre.reset_index(), on=keys, how="left")
    else:
        g[PRE] = np.nan
    g["neg_regret"] = -g["avg_regret"]
    return g


def cell_games(trials, btype, timing):
    t = trials[trials["type"] == btype]
    t = t[(t["hint_timing"] == timing) | (t["ogHint"] == NO_HINT)]
    g = games_post_onset(t, timing)
    g["hint_cat"] = g["ogHint"].map(HINT_CATS).fillna("")
    return g


def load_cells():
    trials = load_trials()
    return {
        (t, on): cell_games(trials, t, on) for t in TYPE_ORDER for on in TIMING_ORDER
    }


def _covariates(timing):
    return [PRE] if int(timing) > 0 else None


def _base_cols(base):
    return dict(
        no_hint_mean=base["mean"],
        no_hint_se=base["se"],
        no_hint_lo=base["lo"],
        no_hint_hi=base["hi"],
    )


def extract_hint_effects(cells):
    rows = []
    for btype in TYPE_ORDER:
        for timing in TIMING_ORDER:
            sub = cells[(btype, timing)]
            label = f"{btype} onset {timing}"
            eff = F.fit_hint_contrasts(
                sub,
                hint_col="ogHint",
                subject_col="user_id",
                ref=NO_HINT,
                label=label,
                covariates=_covariates(timing),
            )
            base = eff.attrs["baseline"]
            pre_coef = eff.attrs["covariates"].get(PRE, np.nan)
            for r in eff.itertuples():
                rows.append(
                    dict(
                        type=btype,
                        ogHint=r.ogHint,
                        hint_timing=timing,
                        hint_cat=HINT_CATS.get(r.ogHint, ""),
                        delta=r.delta,
                        delta_lo=r.delta_lo,
                        delta_hi=r.delta_hi,
                        se=r.se,
                        z=r.z,
                        p=r.p,
                        p_adj=np.nan,
                        **_base_cols(base),
                        n=r.n_games,
                        n_games=r.n_games,
                        n_subjects=r.n_subjects,
                        n_games_nohint=base["n_games"],
                        n_post_trials=20 - int(timing),
                        pre_coef=pre_coef,
                        sd_subject=r.sd_subject,
                        sd_resid=r.sd_resid,
                        method=r.method,
                        fit_note=r.fit_note,
                    )
                )
    df = pd.DataFrame(rows)
    df = F.add_adjusted_p(df)  # FDR family: all 40 contrasts
    df["hint_color"] = df["hint_cat"].map(lambda c: CAT_COLORS.get(c, "#888"))
    df["stars"] = df["p_adj"].map(F.stars)
    return df


def _fit_all(cells):
    hint_effects = extract_hint_effects(cells)
    hint_effects.to_csv(OUT_DIR / "hint_effects.csv", index=False)
    print(
        f"significant (p < .05): {int((hint_effects['p'] < 0.05).sum())}/{len(hint_effects)}; "
        f"after FDR: {int((hint_effects['p_adj'] < 0.05).sum())}/{len(hint_effects)}; methods: "
        f"{hint_effects['method'].value_counts().to_dict()}"
    )
    print(
        hint_effects[
            [
                "type",
                "hint_timing",
                "ogHint",
                "delta",
                "delta_lo",
                "delta_hi",
                "p",
                "p_adj",
                "stars",
                "pre_coef",
            ]
        ]
        .round(3)
        .to_string(index=False)
    )
    return hint_effects


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refit", action="store_true")
    args = parser.parse_args()

    hint_csv = OUT_DIR / "hint_effects.csv"
    if hint_csv.exists() and not args.refit:
        print(f"Exists (use --refit to regenerate): {hint_csv}")
        return
    _fit_all(load_cells())
    print("Done.")


if __name__ == "__main__":
    main()
