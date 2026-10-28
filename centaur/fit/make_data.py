from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd

FIT_DIR = Path(__file__).resolve().parent
REPO_ROOT = FIT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT)) if str(REPO_ROOT) not in sys.path else None
from constants import normalize_type

SRC = REPO_ROOT / "behavioral" / "instructed_bandits" / "data"
OUT = FIT_DIR / "data"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--behavioral-dir", type=Path, default=SRC)
    p.add_argument("--output-dir", type=Path, default=OUT)
    p.add_argument("--test-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    path = args.behavioral_dir / "behavioral_clean.parquet"
    assert path.exists(), f"Not found: {path}. Run make_clean.py first."
    df = pd.read_parquet(path).dropna(subset=["choice", "type"])
    df["hint"] = df["ogHint"].astype(str)
    df["choice"] = (
        pd.to_numeric(df["choice"], errors="coerce").fillna(0).astype(int).clip(lower=0)
    )

    # Participant split
    users = df["user_id"].unique()
    rng = np.random.default_rng(args.seed)
    rng.shuffle(users)
    n_test = max(1, int(len(users) * args.test_frac))
    test_users, train_users = set(users[:n_test]), set(users[n_test:])

    # Trial parquets
    cols = ["user_id", "game_id", "type", "hint", "trial", "choice"]
    train_trials = df[df["user_id"].isin(train_users)][cols]
    test_trials = df[df["user_id"].isin(test_users)][cols]

    # Game JSONs
    df = df.sort_values(["user_id", "game_id", "trial"])
    games = []
    for (uid, gid), grp in df.groupby(["user_id", "game_id"]):
        choices = grp["choice"].tolist()
        if not choices:
            continue
        rewards = (
            grp["reward"].tolist() if "reward" in grp.columns else [0.0] * len(choices)
        )
        gtype = normalize_type(grp["type"].iloc[0])
        game = {
            "user_id": int(uid),
            "type": gtype,
            "hint": str(grp["ogHint"].iloc[0]),
            "choices": choices,
            "rewards": rewards,
        }
        if gtype in ("two_context", "three_context") and "contexts" in grp.columns:
            game["contexts"] = [int(c) if pd.notna(c) else 0 for c in grp["contexts"]]
        if gtype == "linear_contextual":
            if "allArmFts" in grp.columns and grp["allArmFts"].notna().all():
                game["allArmFts"] = [
                    [list(int(x) for x in arm) for arm in trial]
                    for trial in grp["allArmFts"]
                ]
            if "armFts" in grp.columns:
                game["armFts"] = [
                    list(int(x) for x in f) for f in grp["armFts"] if f is not None
                ]
        games.append(game)

    train_games = [g for g in games if g["user_id"] in train_users]
    test_games = [g for g in games if g["user_id"] in test_users]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_trials.to_parquet(args.output_dir / "train_trials.parquet", index=False)
    test_trials.to_parquet(args.output_dir / "test_trials.parquet", index=False)
    for name, data in [("train_games", train_games), ("test_games", test_games)]:
        with open(args.output_dir / f"{name}.json", "w") as f:
            json.dump(data, f, indent=2)
    with open(args.output_dir / "split_info.json", "w") as f:
        json.dump(
            {
                "test_frac": args.test_frac,
                "seed": args.seed,
                "n_train": len(train_users),
                "n_test": len(test_users),
                "test_user_ids": sorted(int(x) for x in test_users),
                "train_user_ids": sorted(int(x) for x in train_users),
            },
            f,
            indent=2,
        )
    print(
        f"Wrote {len(train_users)} train ({len(train_games)} games) / "
        f"{len(test_users)} test ({len(test_games)} games) to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
