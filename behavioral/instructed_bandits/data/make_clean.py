from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from constants import normalize_type  # noqa: E402

RAW = SCRIPT_DIR / "raw" / "study1_participants.json"
OUT_PARQUET = SCRIPT_DIR / "behavioral_clean.parquet"
OUT_STATS = SCRIPT_DIR / "behavioral_clean_stats.json"
ARMFTS_RECONSTRUCTION = SCRIPT_DIR / "reconstruct_armfts_out.json"

BUG_HINT = "Find a few best buttons and alternate between them to maximize your points."
HINT_REMAP = {
    "The points you get on the first pick of a button is the most representative of the button value.": "The points you get on the first pick of a button are the most representative of the button value.",
    "Don't stick with any one button for too long.": "Do not stick with any one button for too long.",
}
NO_HINT = "No hint."


def compute_performance(grp):
    grp = grp.copy()
    if grp["type"].iloc[0] == "linear_contextual" and "armFts" in grp.columns:
        grp["regret"] = 5.0 - grp["reward"].astype(float)
        grp["optimal"] = grp["armFts"].apply(
            lambda f: int(f[2] == 1) if hasattr(f, "__len__") and len(f) > 2 else np.nan
        )
        return grp
    regrets, optimal = [], []
    for _, row in grp.iterrows():
        ar = row["arms"]
        if ar is None or (hasattr(ar, "__len__") and len(ar) == 0):
            regrets.append(np.nan)
            optimal.append(np.nan)
            continue
        ar = np.asarray(ar).ravel()
        c = int(row["choice"])
        if 0 <= c < len(ar):
            regrets.append(float(np.max(ar) - ar[c]))
            optimal.append(1 if c == int(np.argmax(ar)) else 0)
        else:
            regrets.append(np.nan)
            optimal.append(np.nan)
    grp["regret"] = regrets
    grp["optimal"] = optimal
    return grp


def expand_trial_rows(df):
    is_list = df["choice"].apply(
        lambda c: (
            c is not None
            and hasattr(c, "__len__")
            and not isinstance(c, str)
            and len(c) > 0
        )
    )
    if not is_list.any():
        return df
    sample = df.loc[is_list].iloc[0]
    cols = [
        c for c in ("choice", "reward", "rt", "armFts", "contexts") if c in df.columns
    ]
    cols = [
        c
        for c in cols
        if hasattr(sample[c], "__len__")
        and not isinstance(sample[c], str)
        and len(sample[c]) > 0
    ]
    return df.explode(cols, ignore_index=True) if cols else df


def load_flat(sessions):
    rows = []
    for user in sessions:
        df = pd.DataFrame(user["studyData"])
        df["user_id"] = user["participant"]
        df["batch"] = user["batch"]
        if "gameId" not in df.columns:
            df["gameId"] = (df.get("id", 0) == 0).cumsum()
        df["gameId"] = df["gameId"].ffill()
        rows.append(df)
    out = pd.concat(rows, ignore_index=True)
    out["hintUnderstanding"] = out.groupby(["user_id", "gameId"], group_keys=False)[
        "hintUnderstanding"
    ].apply(lambda s: s.bfill().ffill())
    out = out.dropna(subset=["type", "choice"])
    out = expand_trial_rows(out)
    out["_id_num"] = pd.to_numeric(out["id"], errors="coerce")
    out = out.sort_values(["user_id", "gameId", "_id_num"], kind="stable").drop(
        columns="_id_num"
    )
    out["trial"] = out.groupby(["user_id", "gameId"]).cumcount()
    out["type"] = out["type"].apply(normalize_type)
    out = out[out["ogHint"] != BUG_HINT]
    for col in ("ogHint", "hint"):
        out[col] = out[col].replace(HINT_REMAP)
    out["game_id"] = out["gameId"]
    out = out.reset_index(drop=True)
    perf = out.groupby(["user_id", "game_id"], group_keys=False).apply(
        compute_performance, include_groups=False
    )
    out["regret"] = perf["regret"]
    out["optimal"] = perf["optimal"]
    return out


def add_lincon_armfts(flat):
    recs = json.load(open(ARMFTS_RECONSTRUCTION))
    rdf = pd.DataFrame(recs)
    rdf["game_id"] = rdf["game_idx"] + 1
    flat = flat.merge(
        rdf[["user_id", "game_id", "trial", "allArmFts"]],
        on=["user_id", "game_id", "trial"],
        how="left",
    )
    print(
        f"  {int(flat['allArmFts'].notna().sum())} linear_contextual trials with reconstructed all-arm features"
    )
    return flat


def stats(flat):
    games = flat.drop_duplicates(subset=["user_id", "game_id"])
    return {
        "total_trials": int(len(flat)),
        "total_games": int(len(games)),
        "total_users": int(flat["user_id"].nunique()),
        "users_per_batch": {
            int(k): int(v)
            for k, v in games.groupby("batch")["user_id"].nunique().items()
        },
        "games_per_type": {
            str(k): int(v) for k, v in games["type"].value_counts().sort_index().items()
        },
        "games_per_hint": {
            str(k): int(v)
            for k, v in games["ogHint"].value_counts().sort_index().items()
        },
        "missing": {
            "hintUnderstanding_na_count": int(
                flat.loc[flat["ogHint"] != NO_HINT, "hintUnderstanding"].isna().sum()
            ),
            "optimal_na_count": int(flat["optimal"].isna().sum()),
            "choice_na_count": int(flat["choice"].isna().sum()),
        },
    }


def main():
    flat = add_lincon_armfts(load_flat(json.load(open(RAW))))
    flat.to_parquet(OUT_PARQUET, index=False)
    s = stats(flat)
    json.dump(s, open(OUT_STATS, "w"), indent=2)
    print(
        f"  {s['total_users']} participants, {s['total_games']} games, {s['total_trials']} trials -> {OUT_PARQUET}"
    )


if __name__ == "__main__":
    main()
