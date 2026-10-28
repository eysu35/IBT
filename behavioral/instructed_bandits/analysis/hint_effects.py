import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "analysis"))
from constants import TYPE_ORDER, HINT_LABELS
import freq_lmm as F

DATA_PATH = _REPO_ROOT / "behavioral/instructed_bandits/data/behavioral_clean.parquet"
OUT_DIR = Path(__file__).resolve().parent

NO_HINT = "No hint."
REFERENCE = f"stationary:{NO_HINT}"
CAT_COLORS = {"V": "#d62728", "P": "#1f77b4", "S": "#2ca02c", "none": "#7f7f7f"}


def _hint_cat(hint):
    labels = HINT_LABELS.get(hint, [])
    return labels[0] if labels else "none"


def load_game_data():
    df = F.normalize_regret(pd.read_parquet(DATA_PATH))
    g = (
        df.groupby(["user_id", "game_id", "type", "ogHint"])
        .agg(avg_regret=("regret", "mean"))
        .reset_index()
    )
    g["neg_regret"] = -g["avg_regret"]
    g["type_hint"] = g["type"] + ":" + g["ogHint"]
    return g


def _no_hint_row(btype, base):
    return dict(
        type=btype,
        ogHint=NO_HINT,
        delta=0.0,
        delta_lo=0.0,
        delta_hi=0.0,
        se=np.nan,
        z=np.nan,
        p=np.nan,
        p_adj=np.nan,
        n_games=base["n_games"],
        n_subjects=base["n_subjects"],
        no_hint_mean=base["mean"],
        no_hint_se=base["se"],
        no_hint_lo=base["lo"],
        no_hint_hi=base["hi"],
        hint_cat="none",
        hint_color=CAT_COLORS["none"],
        stars="",
        sd_subject=base["sd_subject"],
        sd_resid=base["sd_resid"],
        method=base["method"],
        fit_note=base["fit_note"],
    )


def fit_per_task(game_df):
    rows = []
    for btype in TYPE_ORDER:
        sub = game_df[game_df["type"] == btype]
        print(
            f"  {btype}: {len(sub)} games, {sub['user_id'].nunique()} subjects",
            flush=True,
        )
        eff = F.fit_hint_contrasts(
            sub, hint_col="ogHint", subject_col="user_id", ref=NO_HINT, label=btype
        )
        base = eff.attrs["baseline"]
        rows.append(_no_hint_row(btype, base))
        for r in eff.itertuples():
            cat = _hint_cat(r.ogHint)
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
                    p_adj=np.nan,
                    n_games=r.n_games,
                    n_subjects=r.n_subjects,
                    no_hint_mean=base["mean"],
                    no_hint_se=base["se"],
                    no_hint_lo=base["lo"],
                    no_hint_hi=base["hi"],
                    hint_cat=cat,
                    hint_color=CAT_COLORS.get(cat, "#888"),
                    stars="",
                    sd_subject=r.sd_subject,
                    sd_resid=r.sd_resid,
                    method=r.method,
                    fit_note=r.fit_note,
                )
            )
    df = pd.DataFrame(rows)
    df = F.add_adjusted_p(df)  # FDR family: all 40 hint contrasts
    df["stars"] = df["p_adj"].map(F.stars)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refit", action="store_true")
    args = parser.parse_args()

    out_csv = OUT_DIR / "hint_effects.csv"
    if out_csv.exists() and not args.refit:
        print(f"Exists (use --refit to regenerate): {out_csv}")
        return

    print("Loading human game data …")
    game_df = load_game_data()
    print("Fitting per-task mixed models …")
    effects = fit_per_task(game_df)
    effects.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}  ({len(effects)} rows)")

    hints = effects[effects["ogHint"] != NO_HINT]
    print(
        f"significant (p < .05): {int((hints['p'] < 0.05).sum())}/{len(hints)}; "
        f"after FDR: {int((hints['p_adj'] < 0.05).sum())}/{len(hints)}"
    )
    print(
        effects[
            [
                "type",
                "ogHint",
                "delta",
                "delta_lo",
                "delta_hi",
                "p",
                "p_adj",
                "stars",
                "method",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
