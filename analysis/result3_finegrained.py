import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # analysis/, for exp_paths
from exp_paths import SIM_DIR, FIG_DIR
from constants import TYPE_ORDER
import paper_style as PS

HUMAN_PARQUET = (
    _REPO_ROOT / "behavioral/instructed_bandits/data/behavioral_clean.parquet"
)
CENTAUR_PARQUET = _REPO_ROOT / "centaur/data/centaur_clean.parquet"
CENTAUR_IBT_PARQUET = _REPO_ROOT / "centaur/data/centaur_IBT_clean.parquet"
LLAMA_PARQUET = _REPO_ROOT / "centaur/data/llama_clean.parquet"
BAYES_PARQUET = SIM_DIR / "bayesian_sim_trials.parquet"
FIGURES_DIR = FIG_DIR

METRICS = ["switch_rate", "entropy", "exploitation_onset"]
METRIC_LABELS = {
    "switch_rate": "Switch Rate",
    "entropy": "Choice Entropy",
    "exploitation_onset": "Exploitation Onset",
}
TYPE_LABELS = PS.TYPE_LABELS
TYPE_PALETTE = PS.TYPE_PALETTE
# Rows: (y agent, x agent).
PAIRS = [("bayesian", "people"), ("centaur", "people")]
AGENT_LABELS = {"people": "Human", "bayesian": "Bayesian", "centaur": "Centaur"}

_THREE_CTX_CANONICAL = (
    "When you see a square, choose G. When you see a circle, choose A. "
    "When you see a triangle, explore all buttons to find the best."
)
_THREE_CTX_PREFIX = "When you see a square, choose"


def _normalize_centaur_hint(h):
    if h.startswith(_THREE_CTX_PREFIX) and "triangle" in h:
        return _THREE_CTX_CANONICAL
    return h


# --- metric helpers (one game = one choice sequence) ---


def _switch_rate(c):
    return float(np.mean(c[1:] != c[:-1])) if len(c) > 1 else np.nan


def _entropy(c):
    counts = np.bincount(c.astype(int))
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def _exploitation_onset(c, run=5):
    for i in range(len(c) - run + 1):
        if len(set(c[i : i + run])) == 1:
            return i / len(c)
    return 1.0


def game_metrics(c):
    c = np.asarray(c, dtype=int)
    return {
        "switch_rate": _switch_rate(c),
        "entropy": _entropy(c),
        "exploitation_onset": _exploitation_onset(c),
    }


# --- loaders: one row per game with the 3 metrics ---


def load_human():
    df = pd.read_parquet(HUMAN_PARQUET)
    rows = []
    for (_, _), g in df.groupby(["user_id", "gameId"]):
        g = g.sort_values("trial")
        c = g["choice"].dropna().values
        if len(c) < 5:
            continue
        rows.append(
            {
                "type": g["type"].iloc[0],
                "ogHint": g["ogHint"].iloc[0],
                **game_metrics(c),
            }
        )
    return pd.DataFrame(rows)


def load_llm(parquet_path):
    df = pd.read_parquet(parquet_path)
    df["ogHint"] = df["ogHints"].map(_normalize_centaur_hint)
    rows = []
    for _, g in df.groupby("id"):
        g = g.sort_values("trial_num")
        c = g["choice"].dropna().values
        if len(c) < 5:
            continue
        rows.append(
            {
                "type": g["type"].iloc[0],
                "ogHint": g["ogHint"].iloc[0],
                **game_metrics(c),
            }
        )
    return pd.DataFrame(rows)


def load_bayesian():
    df = pd.read_parquet(BAYES_PARQUET)
    rows = []
    for (btype, hint, _), g in df.groupby(["type", "ogHint", "sim_id"]):
        g = g.sort_values("trial_num")
        c = g["choice"].dropna().values
        if len(c) < 5:
            continue
        rows.append({"type": btype, "ogHint": hint, **game_metrics(c)})
    return pd.DataFrame(rows)


def agg(df):
    return df.groupby(["type", "ogHint"])[METRICS].mean().reset_index()


def _scatter(ax, wide, metric, yk, xk, r, lim, first_col):
    x = wide[xk].to_numpy()
    y = wide[yk].to_numpy()
    types = wide.index.get_level_values("task_type").to_numpy()
    for btype in TYPE_ORDER:
        mask = types == btype
        if not mask.any():
            continue
        ax.scatter(
            x[mask],
            y[mask],
            s=PS.SCATTER_S,
            color=TYPE_PALETTE[btype],
            edgecolors="black",
            linewidths=1.5,
            alpha=0.9,
            zorder=3,
            label=TYPE_LABELS[btype],
        )
    ax.plot(lim, lim, "--", color="gray", lw=2, alpha=0.6, zorder=1)
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_aspect("equal", adjustable="box")
    ax.text(
        0.95,
        0.06,
        f"r = {r:.2f}",
        transform=ax.transAxes,
        fontweight="bold",
        va="bottom",
        ha="right",
    )
    ax.set_xlabel(AGENT_LABELS[xk])
    ax.set_ylabel(AGENT_LABELS[yk] if first_col else "")


def main():
    PS.set_theme()

    print("Loading tables …")
    tr = PS.read_table("trajectory.csv")
    corr = PS.read_table("trajectory_corr.csv")
    wide = {
        m: tr.pivot(index=["task_type", "hint_id"], columns="agent", values=m)
        for m in METRICS
    }

    fig, axes = plt.subplots(2, 3, figsize=(22, 14), dpi=100)
    fig.subplots_adjust(
        left=0.07, right=0.99, top=0.95, bottom=0.165, hspace=0.30, wspace=0.22
    )

    for col, metric in enumerate(METRICS):
        w = wide[metric]
        vals = w[["people", "bayesian", "centaur"]].to_numpy()
        lo, hi = vals.min(), vals.max()
        pad = (hi - lo) * PS.AXIS_PAD
        lim = (lo - pad, hi + pad)
        for row, (yk, xk) in enumerate(PAIRS):
            ax = axes[row, col]
            r = corr[
                (corr["y"] == yk) & (corr["x"] == xk) & (corr["metric"] == metric)
            ]["pearson_r"].iloc[0]
            _scatter(ax, w, metric, yk, xk, r, lim, first_col=(col == 0))
            if row == 0:
                ax.set_title(METRIC_LABELS[metric], pad=14)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.53, 0.0),
        columnspacing=1.6,
        handletextpad=0.5,
    )
    PS.save(fig, FIGURES_DIR, "result3_finegrained")


if __name__ == "__main__":
    main()
