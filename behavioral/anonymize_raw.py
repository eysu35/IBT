from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
OUT_STUDY1 = HERE / "instructed_bandits/data/raw/study1_participants.json"
OUT_STUDY2 = HERE / "hint_time/data/raw/study2_participants.json"

TIMED_BUILD_START = datetime(2026, 2, 26, tzinfo=timezone.utc).timestamp()
STUDY1_FREEZE = datetime(2026, 2, 24, 1, 0, tzinfo=timezone.utc).timestamp()
STUDY1_TYPES = ("stationary", "patch_foraging", "patchForaging")
STUDY1_ONLY_HINTS = {
    "Avoid {min1} and {min2}.",
    "{max1} and {max2} are better than the others.",
}
KEEP_CONDITIONS = ("instructionsVersion", "experimentVersion")


def age(dob, start):
    if not dob or not start:
        return None
    born = datetime.strptime(dob[:10], "%Y-%m-%d").date()
    at = datetime.fromtimestamp(start, timezone.utc).date()
    a = at.year - born.year - ((at.month, at.day) < (born.month, born.day))
    return a if 18 <= a <= 100 else None


def trial_entries(user):
    return [
        e for e in user.get("studyData", []) if isinstance(e, dict) and "phase" not in e
    ]


def quiz_entries(user):
    return [
        e for e in user.get("studyData", []) if isinstance(e, dict) and "phase" in e
    ]


def has_hint_timing(user):
    return any(e.get("hintTiming") is not None for e in trial_entries(user))


def session(user, index, **extra):
    start = (user.get("starttime") or {}).get("_seconds")
    demo = user.get("demographicForm") or {}
    out = {
        "participant": index,
        "consented": bool(user.get("consented", False)),
        "done": bool(user.get("done", False)),
        "withdrawn": bool(user.get("withdrawn", False)),
        "conditions": {
            k: v
            for k, v in (user.get("conditions") or {}).items()
            if k in KEEP_CONDITIONS
        },
        "demographics": {
            "gender": demo.get("gender"),
            "age": age(demo.get("dob"), start),
        },
    }
    out.update(extra)
    out["quiz"] = quiz_entries(user)
    return out


def study1_sessions(study1_raw, onset_raw):
    out = []
    for i, u in enumerate(study1_raw):
        out.append(session(u, i, batch=1, studyData=trial_entries(u)))
    k = len(out)
    for u in onset_raw:
        start = (u.get("starttime") or {}).get("_seconds")
        end = (u.get("endtime") or {}).get("_seconds")
        if not (
            start
            and end
            and start < TIMED_BUILD_START
            and end < STUDY1_FREEZE
            and has_hint_timing(u)
        ):
            continue
        keep = {
            e.get("gameId")
            for e in trial_entries(u)
            if e.get("type") in STUDY1_TYPES
            and e.get("ogHint") not in STUDY1_ONLY_HINTS
        }
        games = [e for e in trial_entries(u) if e.get("gameId") in keep]
        for e in games:
            e.pop("hintTiming", None)
        out.append(session(u, k, batch=2, studyData=games))
        k += 1
    return out


def study2_sessions(onset_raw, games_parquet):
    keep = pd.read_parquet(games_parquet).drop_duplicates(["user_id", "game_id"])
    keep_games = {(int(r.user_id), int(r.game_id)) for r in keep.itertuples()}
    out = []
    for i, u in enumerate(onset_raw):
        start = (u.get("starttime") or {}).get("_seconds")
        games = [
            e for e in trial_entries(u) if (i, int(e.get("gameId", -1))) in keep_games
        ]
        if not games:
            continue
        cohort = "onset0" if start < TIMED_BUILD_START else "timed"
        out.append(session(u, len(out), cohort=cohort, studyData=games))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--study1", type=Path, required=True)
    p.add_argument("--hint-onset", type=Path, required=True)
    p.add_argument(
        "--study2-games",
        type=Path,
        required=True,
        help="Study 2 parquet of record (user_id = export index) listing the games to keep",
    )
    args = p.parse_args()
    study1_raw = json.load(open(args.study1))
    onset_raw = json.load(open(args.hint_onset))
    for sessions, dst in (
        (study1_sessions(study1_raw, onset_raw), OUT_STUDY1),
        (study2_sessions(onset_raw, args.study2_games), OUT_STUDY2),
    ):
        dst.parent.mkdir(parents=True, exist_ok=True)
        with open(dst, "w") as f:
            json.dump(sessions, f, separators=(",", ":"))
        print(f"{len(sessions)} participants -> {dst}")


if __name__ == "__main__":
    main()
