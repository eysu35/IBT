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

RAW = SCRIPT_DIR / "raw" / "study2_participants.json"
OUT_PARQUET = SCRIPT_DIR / "behavioral_clean.parquet"
OUT_STATS = SCRIPT_DIR / "behavioral_clean_stats.json"

COHORT_TO_VERSION = {"onset0": "old", "timed": "new"}
NO_HINT = "No hint."
HINT_REMAP = {
    "The points you get on the first pick of a button is the most representative of the button value.": "The points you get on the first pick of a button are the most representative of the button value."
}


def compute_performance(grp):
    grp = grp.copy()
    regrets, optimal = [], []
    for _, row in grp.iterrows():
        ar = row["arms"]
        c = int(row["choice"])
        if ar is None or len(ar) == 0 or not (0 <= c < len(ar)):
            regrets.append(np.nan)
            optimal.append(np.nan)
            continue
        ar = np.asarray(ar).ravel()
        regrets.append(float(np.max(ar) - ar[c]))
        optimal.append(1 if c == int(np.argmax(ar)) else 0)
    grp["regret"] = regrets
    grp["optimal"] = optimal
    return grp


def load_flat(sessions):
    rows = []
    for user in sessions:
        df = pd.DataFrame(user["studyData"])
        df["user_id"] = user["participant"]
        df["data_version"] = COHORT_TO_VERSION[user["cohort"]]
        rows.append(df)
    out = pd.concat(rows, ignore_index=True)
    out["hintUnderstanding"] = out.groupby(["user_id", "gameId"], group_keys=False)[
        "hintUnderstanding"
    ].apply(lambda s: s.bfill().ffill())
    out = out[out["id"] != "nextGame"]
    out = out.dropna(subset=["type", "choice", "ogHint"])
    out["type"] = out["type"].apply(normalize_type)
    for col in ("ogHint", "hint"):
        out[col] = out[col].replace(HINT_REMAP)
    out["game_id"] = out["gameId"]
    out["_id_num"] = pd.to_numeric(out["id"], errors="coerce")
    out = (
        out.sort_values(["user_id", "gameId", "_id_num"])
        .reset_index(drop=True)
        .drop(columns=["_id_num"])
    )
    out["trial"] = out.groupby(["user_id", "game_id"]).cumcount()

    # on the first build every hint was shown at trial 0, whatever the logged hintTiming
    onset0_hint = (out["data_version"] == "old") & (out["ogHint"] != NO_HINT)
    out.loc[onset0_hint, "hintTiming"] = 0
    out["hint_timing"] = "0"
    has_hint = (out["ogHint"] != NO_HINT) & out["hintTiming"].notna()
    out.loc[has_hint, "hint_timing"] = (
        out.loc[has_hint, "hintTiming"].astype(int).astype(str)
    )

    perf = out.groupby(["user_id", "game_id"], group_keys=False).apply(
        compute_performance, include_groups=False
    )
    out["regret"] = perf["regret"].values
    out["optimal"] = perf["optimal"].values
    out["id"] = out["id"].astype(str)
    return out


def main():
    flat = load_flat(json.load(open(RAW)))
    flat.to_parquet(OUT_PARQUET, index=False)
    games = flat.drop_duplicates(subset=["user_id", "game_id"])
    s = {
        "total_trials": int(len(flat)),
        "total_games": int(len(games)),
        "total_users": int(flat["user_id"].nunique()),
        "games_per_type": {
            k: int(v) for k, v in games["type"].value_counts().sort_index().items()
        },
        "games_per_hint": {
            k: int(v) for k, v in games["ogHint"].value_counts().sort_index().items()
        },
        "hint_timing_counts": {
            k: int(v)
            for k, v in games["hint_timing"].value_counts().sort_index().items()
        },
        "data_version_counts": {
            k: int(v)
            for k, v in games["data_version"].value_counts().sort_index().items()
        },
    }
    json.dump(s, open(OUT_STATS, "w"), indent=2)
    print(
        f"  {s['total_users']} participants, {s['total_games']} games, {s['total_trials']} trials -> {OUT_PARQUET}"
    )


if __name__ == "__main__":
    main()
