import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from constants import TYPE_ORDER  # noqa: E402

OUT = Path(__file__).resolve().parent / "data"
TABLES = REPO / "analysis/tables"
Z = 1.96
AGENTS = {"people": "human", "bayesian": "bayesian", "centaur": "centaur"}
ONSETS = [0, 4, 8, 12]


def arm_spread(a):
    try:
        v = np.asarray(a, float).ravel()
    except (TypeError, ValueError):
        return np.nan
    v = v[np.isfinite(v)]
    return float(v.max() - v.min()) if v.size >= 2 else np.nan


def normalize_regret(df):  # same rule as analysis/freq_lmm.normalize_regret
    out = df.copy()
    sp = out["type"].map({"linear_contextual": 7.0}).astype(float).fillna(70.0)
    if "ogArms" in out.columns:
        from_arms = out["ogArms"].map(arm_spread)
        use = (
            from_arms.notna() & (from_arms > 1.0) & (out["type"] != "linear_contextual")
        )
        sp = sp.where(~use, from_arms)
    out["regret"] = 100.0 * out["regret"].astype(float) / sp
    return out


def r(x, k=2):
    return None if pd.isna(x) else round(float(x), k)


def load_trials():
    h = normalize_regret(
        pd.read_parquet(
            REPO / "behavioral/instructed_bandits/data/behavioral_clean.parquet"
        )
    )
    b = normalize_regret(
        pd.read_parquet(REPO / "bayesian/data/bayesian_sim_trials.parquet").rename(
            columns={"trial_num": "trial"}
        )
    )
    c = pd.read_parquet(REPO / "centaur/data/centaur_clean.parquet").rename(
        columns={"trial_num": "trial", "ogHints": "ogHint"}
    )
    c.loc[c["type"] == "bernoulli", "regret"] *= 100
    c = normalize_regret(c)
    return {"human": h, "bayesian": b, "centaur": c}


def curves(frames):
    out = {}
    for t in TYPE_ORDER:
        out[t] = {}
        for agent, df in frames.items():
            sub = df[df["type"] == t]
            out[t][agent] = {}
            for hint, g in sub.groupby("ogHint"):
                s = g.groupby("trial")["regret"]
                m, se = s.mean(), s.sem().fillna(0)
                key = "none" if hint == "No hint." else hint
                out[t][agent][key] = {
                    "n": int(
                        g.groupby(["user_id", "gameId"]).ngroups
                        if agent == "human"
                        else len(g) / g["trial"].nunique()
                    ),
                    "mean": [r(v) for v in m],
                    "ci": [r(Z * v) for v in se],
                }
    return out


def hint_effects():
    e = pd.read_csv(TABLES / "hint_effects.csv")
    stars = pd.read_csv(
        REPO / "behavioral/instructed_bandits/analysis/hint_effects.csv"
    )
    stars = {
        (t, h): s
        for t, h, s in zip(stars["type"], stars["ogHint"], stars["stars"].fillna(""))
    }
    out = {}
    for t in TYPE_ORDER:
        sub = e[e["task_type"] == t]
        human = sub[sub["agent"] == "people"]
        order = human[
            human["hint_id"] != f"{human['hint_id'].iloc[0].split('_')[0]}_nohint"
        ]
        order = order.sort_values("rr_own_base", ascending=False)["hint_text"].tolist()
        rows = []
        for hint in order:
            row = {"hint": hint, "stars": stars.get((t, hint), "")}
            for agent, key in AGENTS.items():
                m = sub[(sub["agent"] == agent) & (sub["hint_text"] == hint)]
                if len(m):
                    m = m.iloc[0]
                    row[key] = [
                        r(m["rr_own_base"]),
                        r(m["rr_own_lo"]),
                        r(m["rr_own_hi"]),
                    ]
            rows.append(row)
        base = {}
        for agent, key in AGENTS.items():
            m = sub[(sub["agent"] == agent) & (sub["hint_text"] == "No hint.")].iloc[0]
            base[key] = [
                r(m["rr_human_base"]),
                r(m["rr_human_lo"]),
                r(m["rr_human_hi"]),
            ]
        h0 = human.iloc[0]
        out[t] = {
            "hints": rows,
            "nohint": base,
            "human_band": [
                r(h0["human_nohint_lo"] - h0["human_nohint_regret"]),
                r(h0["human_nohint_hi"] - h0["human_nohint_regret"]),
            ],
        }
    return out


def timing():
    tm = pd.read_csv(TABLES / "timing.csv")
    stars = pd.read_csv(REPO / "behavioral/hint_time/analysis/hint_effects.csv")
    stars = {
        (t, h, int(o)): s
        for t, h, o, s in zip(
            stars["type"],
            stars["ogHint"],
            stars["hint_timing"],
            stars["stars"].fillna(""),
        )
    }
    out = {}
    for t in ["stationary", "patch_foraging"]:
        sub = tm[tm["task_type"] == t]
        hints = (
            sub[sub["agent"] == "people"]
            .groupby("hint_text")["rr_own_base"]
            .first()
            .sort_values(ascending=False)
            .index
        )
        rows = []
        for hint in hints:
            row = {"hint": hint, "stars": [stars.get((t, hint, o), "") for o in ONSETS]}
            for agent, key in AGENTS.items():
                m = (
                    sub[(sub["agent"] == agent) & (sub["hint_text"] == hint)]
                    .set_index("onset")
                    .reindex(ONSETS)
                )
                row[key] = [
                    [r(a), r(b), r(c)]
                    for a, b, c in zip(m["rr_own_base"], m["rr_own_lo"], m["rr_own_hi"])
                ]
            rows.append(row)
        h0 = sub[sub["agent"] == "people"].iloc[0]
        out[t] = {
            "hints": rows,
            "human_band": [
                r(h0["human_nohint_lo"] - h0["human_nohint_regret"]),
                r(h0["human_nohint_hi"] - h0["human_nohint_regret"]),
            ],
        }
    return out


def study2_curves():
    df = normalize_regret(
        pd.read_parquet(REPO / "behavioral/hint_time/data/behavioral_clean.parquet")
    )
    df = df[df["ogHint"] == "No hint."]
    out = {}
    for t, g in df.groupby("type"):
        s = g.groupby("trial")["regret"]
        out[t] = {"mean": [r(v) for v in s.mean()], "ci": [r(Z * v) for v in s.sem()]}
    return out


def centaur_examples():  # one simulated game per task x hint, the one closest to the condition's median regret
    d = pd.read_parquet(REPO / "centaur/data/centaur_clean.parquet")
    out = {}
    for (t, hint), g in d.groupby(["type", "ogHints"]):
        mr = g.groupby("id")["regret"].mean()
        ex = g[g["id"] == (mr - mr.median()).abs().idxmin()].sort_values("trial_num")
        has_ctx = ex["contexts"].notna().any()
        out.setdefault(t, {})["none" if hint == "No hint." else hint] = {
            "hint": ex["hint"]
            .iloc[0]
            .replace(
                "choose . When", "choose 1. When"
            ),  # typo in the base Centaur run config, shown as intended
            "choice": ex["choice"].astype(int).tolist(),
            "optimal": ex["optimal"].astype(int).tolist(),
            "probs": [[round(float(p), 3) for p in row] for row in ex["probs"]],
            "arms": [[round(float(v), 1) for v in row] for row in ex["arms"]],
            "context": ex["contexts"].astype(int).tolist() if has_ctx else None,
        }
    return out


def main():
    OUT.mkdir(exist_ok=True)
    frames = load_trials()
    json.dump(curves(frames), open(OUT / "curves.json", "w"), separators=(",", ":"))
    json.dump(
        hint_effects(), open(OUT / "hint_effects.json", "w"), separators=(",", ":")
    )
    json.dump(
        {"effects": timing(), "curves": study2_curves()},
        open(OUT / "timing.json", "w"),
        separators=(",", ":"),
    )
    json.dump(
        centaur_examples(), open(OUT / "centaur.json", "w"), separators=(",", ":")
    )
    for p in sorted(OUT.glob("*.json")):
        print(p.name, p.stat().st_size // 1024, "KB")


if __name__ == "__main__":
    main()
