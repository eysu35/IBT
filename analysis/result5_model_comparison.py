import json
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # analysis/, for exp_paths
from exp_paths import FIG_DIR, FIT_JSON
import paper_style as PS

_REPO_ROOT = Path(__file__).resolve().parents[1]

BAYES_JSON = FIT_JSON
CENTAUR_JSON = _REPO_ROOT / "centaur/fit/data/model_comparison.json"
FIGURES_DIR = FIG_DIR
TYPE_LABELS = PS.TYPE_LABELS

# (label, json, model name in the json)
FIT_MODELS = [
    ("Random", CENTAUR_JSON, "Random (uniform)"),
    ("LLaMA", CENTAUR_JSON, "Llama (fitted T)"),
    ("Bayesian", BAYES_JSON, "Bayesian + hints (fitted T + hs)"),
    ("CentaurIBT", CENTAUR_JSON, "CentaurIBT (fitted T)"),
    ("Centaur", CENTAUR_JSON, "Centaur (fitted T)"),
]
LETTER = PS.PANEL_LETTER + 6  # panel letters, a little larger than the shared default
DIFF_MIN = 1.5  # % of arm spread (~1 regret point on the 70-spread set); smaller significant gaps are not worth a row


FIXED_FIT_SUMMARY = _REPO_ROOT / "bayesian/fit/variants/fit_summary.json"
FIXED_FIT_VARIANT = "full"


def load_fits():
    cache, rows = {}, {}
    for label, jpath, mname in FIT_MODELS:
        if jpath not in cache:
            cache[jpath] = {m["model"]: m for m in json.load(open(jpath))["models"]}
        models = cache[jpath]
        m = models.get(mname)
        if m is None:
            base = mname.split(" (")[0]
            matches = [v for k, v in models.items() if k.startswith(base)]
            if not matches:
                raise KeyError(f"No model matching {mname!r} in {jpath}")
            m = matches[0]
        rows[label] = dict(
            mean_ll=m["mean_ll"], se_ll=m.get("se_ll", 0.0), accuracy=m.get("accuracy")
        )
    if FIXED_FIT_SUMMARY.exists():
        summ = {r["model"]: r for r in json.load(open(FIXED_FIT_SUMMARY))["summary"]}
        if FIXED_FIT_VARIANT in summ:
            r = summ[FIXED_FIT_VARIANT]
            rows["Bayesian"] = dict(
                mean_ll=r["mean_ll"], se_ll=r["se_ll"], accuracy=r["accuracy"]
            )
            print(
                f"Bayesian fit from {FIXED_FIT_SUMMARY.name} (variant {FIXED_FIT_VARIANT!r})"
            )
        else:
            print(
                f"WARNING: {FIXED_FIT_SUMMARY} has no {FIXED_FIT_VARIANT!r} variant; using the repo fit"
            )
    else:
        print(
            "WARNING: fits/fit_summary.json not found; panel a uses bayesian/fit/data/model_comparison.json"
        )
    print(
        "held-out fits:",
        {k: (round(v["mean_ll"], 4), v["accuracy"]) for k, v in rows.items()},
    )
    return rows


def differing(md):
    d = md[md["differ"] & (md["abs_diff"] >= DIFF_MIN)].copy()
    d["closer_rank"] = d["closer"].map({"centaur": 0, "bayesian": 1})
    d = d.sort_values(["closer_rank", "abs_diff"], ascending=[True, False]).reset_index(
        drop=True
    )
    d["num"] = np.arange(1, len(d) + 1)
    d["hint_text"] = d["hint_text"].map(
        PS.hint_label
    )  # display wording only (paper_style.HINT_DISPLAY)
    return d


def panel_fit_bars(ax, fits):
    labels = [m[0] for m in FIT_MODELS]
    y = np.arange(len(labels))
    for yi, l in zip(y, labels):
        v, e = -fits[l]["mean_ll"], PS.Z95 * fits[l]["se_ll"]
        ax.barh(
            yi,
            v,
            height=0.68,
            color=PS.MODEL_COLORS[l],
            edgecolor="black",
            linewidth=1.2,
            alpha=PS.BAR_ALPHA,
            zorder=2,
        )
        ax.errorbar(
            v,
            yi,
            xerr=e,
            fmt="none",
            color="black",
            lw=2.0,
            capsize=5,
            capthick=1.6,
            zorder=3,
        )
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_ylim(len(labels) - 0.5, -0.5)
    ax.set_xlabel("Negative log-likelihood")
    hi = max(-fits[l]["mean_ll"] + PS.Z95 * fits[l]["se_ll"] for l in labels)
    ax.set_xlim(0, hi * (1 + PS.AXIS_PAD))
    ax.grid(axis="y", visible=False)


CLOSER_COLOR = {
    "centaur": PS.AGENT_COLORS["Centaur"],
    "bayesian": PS.AGENT_COLORS["Bayesian"],
}
AGREE_COLOR = "#9a9a9a"


def panel_scatter(ax, md, dif):
    keys = {
        (t, h): c for t, h, c in zip(dif["task_type"], dif["hint_id"], dif["closer"])
    }
    allv = []
    for _, r in md.iterrows():
        x, y = r["bayesian"], r["centaur"]
        allv += [r["bayesian_lo"], r["bayesian_hi"], r["centaur_lo"], r["centaur_hi"]]
        closer = keys.get((r["task_type"], r["hint_id"]))
        col = CLOSER_COLOR[closer] if closer else AGREE_COLOR
        ax.errorbar(
            x,
            y,
            xerr=[[x - r["bayesian_lo"]], [r["bayesian_hi"] - x]],
            yerr=[[y - r["centaur_lo"]], [r["centaur_hi"] - y]],
            fmt="none",
            color=col,
            lw=1.6 if closer else 1.0,
            alpha=0.95 if closer else 0.5,
            zorder=3 if closer else 2,
        )
        ax.scatter(
            x,
            y,
            marker="o",
            s=PS.SCATTER_S * (1.9 if closer else 1.25),
            color=col if closer else "white",
            edgecolors="black" if closer else col,
            linewidths=1.6 if closer else 1.8,
            zorder=6 if closer else 4,
        )
    # Limits from the point estimates (the widest CIs may run off the edge), so the cloud fills the box.
    pts = np.concatenate([md["bayesian"].to_numpy(), md["centaur"].to_numpy()])
    lo, hi = pts.min(), pts.max()
    span = hi - lo
    lo, hi = lo - 0.08 * span, hi + 0.08 * span
    ax.plot([lo, hi], [lo, hi], "--", color="gray", lw=1.5, alpha=0.6, zorder=1)
    ax.axhline(0, color="gray", lw=0.8, alpha=0.4, zorder=0)
    ax.axvline(0, color="gray", lw=0.8, alpha=0.4, zorder=0)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)  # same limits on both axes; the box is not forced square
    r = np.corrcoef(md["bayesian"], md["centaur"])[0, 1]
    ax.text(
        0.04,
        0.96,
        f"r = {r:.2f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=PS.SMALL_LABEL + 12,
        fontweight="bold",
    )
    ax.set_xlabel("Bayesian\nregret reduction (%)")
    ax.set_ylabel("Centaur\nregret reduction (%)")


def panel_advantage(ax, dif, n_sim=20000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(dif)
    vals, los, his = [], [], []
    for _, r in dif.iterrows():
        h = rng.normal(r["people"], r["people_se"], n_sim)
        b = rng.normal(r["bayesian"], r["bayesian_se"], n_sim)
        c = rng.normal(r["centaur"], r["centaur_se"], n_sim)
        adv = np.abs(b - h) - np.abs(c - h)
        vals.append(abs(r["bayesian"] - r["people"]) - abs(r["centaur"] - r["people"]))
        lo_, hi_ = np.percentile(adv, [2.5, 97.5])
        los.append(lo_)
        his.append(hi_)
    for yi, (v, lo_, hi_) in enumerate(zip(vals, los, his)):
        col = PS.AGENT_COLORS["Centaur"] if v > 0 else PS.AGENT_COLORS["Bayesian"]
        ax.barh(
            yi,
            v,
            height=0.62,
            color=col,
            edgecolor="black",
            linewidth=1.0,
            alpha=PS.BAR_ALPHA,
            zorder=2,
        )
        ax.errorbar(
            v,
            yi,
            xerr=[[v - lo_], [hi_ - v]],
            fmt="none",
            color="black",
            lw=1.8,
            capsize=4,
            capthick=1.4,
            zorder=3,
        )
    ax.axvline(0, color="black", lw=1.2, zorder=1)
    n_c = int((dif["closer"] == "centaur").sum())
    if 0 < n_c < n:
        ax.axhline(n_c - 0.5, color="gray", lw=1.0, alpha=0.5)
    # Hint text on the RIGHT of the bars (bars on the left), left-aligned.
    ax.set_yticks(range(n))
    ax.set_yticklabels(
        [
            f"{TYPE_LABELS[t]}\n{PS.wrap(h, 58)}"
            for t, h in zip(dif["task_type"], dif["hint_text"])
        ],
        fontsize=PS.SMALL_LABEL + 7,
        linespacing=1.055,
    )
    ax.tick_params(axis="y", length=0, pad=10)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_xlabel("Centaur advantage")
    ax.grid(axis="y", visible=False)
    lo_x, hi_x = min(los), max(his)
    span = hi_x - lo_x
    ax.set_xlim(lo_x - 0.12 * span, hi_x + 0.03 * span)
    xl, xr = ax.get_xlim()
    ax.text(
        xl,
        -0.45,
        "Centaur\ncloser",
        ha="left",
        va="top",
        fontsize=PS.SMALL_LABEL + 6,
        linespacing=0.95,
        color=PS.AGENT_COLORS["Centaur"],
        fontweight="bold",
        zorder=5,
    )
    ax.text(
        xr,
        n - 0.55,
        "Bayesian\ncloser",
        ha="right",
        va="bottom",
        fontsize=PS.SMALL_LABEL + 6,
        linespacing=0.95,
        color=PS.AGENT_COLORS["Bayesian"],
        fontweight="bold",
        zorder=5,
    )


METRICS = [
    ("switch_rate", "Switch rate"),
    ("entropy", "Choice entropy"),
    ("exploitation_onset", "Exploitation onset"),
]


def fit_hint_labels(fig, ax, dif, right_edge, fs_min=20, fs_max=32, gap_px=14):
    rend = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    rows = list(zip(dif["task_type"], dif["hint_text"]))
    ax.yaxis.tick_right()  # measure the hint text where it will be drawn (right of the bars)
    ax.tick_params(axis="y", length=0, pad=30)

    def apply(fs, width):
        ax.set_yticklabels(
            [PS.wrap(h, width) for _, h in rows], fontsize=fs, linespacing=1.05
        )
        fig.canvas.draw()
        return [t.get_window_extent(rend) for t in ax.get_yticklabels()]

    def fits(boxes):
        if any(inv.transform((b.x1, 0))[0] > right_edge for b in boxes):
            return False
        ordered = sorted(boxes, key=lambda b: -b.y1)  # top row first
        return all(a.y0 - b.y1 >= gap_px for a, b in zip(ordered, ordered[1:]))

    best = None
    for fs in range(fs_min, fs_max + 1):
        ok = None
        for width in range(80, 28, -2):
            boxes = apply(fs, width)
            if any(inv.transform((b.x1, 0))[0] > right_edge for b in boxes):
                continue  # too wide: wrap tighter
            ok = width if fits(boxes) else None
            break  # tighter wrap only adds lines
        if ok is None:
            break
        best = (fs, ok)
    if best is None:
        best = (fs_min, 60)
    boxes = apply(*best)
    texts = [t.get_text() for t in ax.get_yticklabels()]
    heights = np.array([b.height for b in boxes])
    ax_px = ax.get_window_extent()
    gap = (ax_px.height - heights.sum()) / (len(heights) + 1)
    tops = ax_px.y1 - gap - np.concatenate([[0], np.cumsum(heights[:-1] + gap)])
    x_px = ax_px.x1 + 34
    for txt, top in zip(texts, tops):
        xf, yf = inv.transform((x_px, top))
        fig.text(xf, yf, txt, ha="left", va="top", fontsize=best[0], linespacing=1.05)
    # Task names on the left of the bars, on the bars' own (even) rows.
    ax.yaxis.tick_left()
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([TYPE_LABELS[t] for t, _ in rows], fontsize=best[0])
    ax.tick_params(axis="y", length=0, pad=10)
    print(f"panel d labels: fontsize {best[0]}, wrap {best[1]}, gap {gap:.0f}px")


def task_means(tr):
    rows = []
    for metric, _ in METRICS:
        w = tr.pivot(index=["task_type", "hint_id"], columns="agent", values=metric)
        for btype, g in w.groupby(level="task_type"):
            rows.append(
                dict(
                    metric=metric,
                    task_type=btype,
                    people=float(g["people"].mean()),
                    bayesian=float(g["bayesian"].mean()),
                    centaur=float(g["centaur"].mean()),
                    n=len(g),
                )
            )
    return pd.DataFrame(rows)


B_AGENTS = [("Bayesian", "bayesian"), ("Human", "people"), ("Centaur", "centaur")]
METRIC_UNITS = {
    "switch_rate": "Switch rate",
    "entropy": "Choice entropy",
    "exploitation_onset": "Exploitation onset",
}


def panel_task_levels(axes, tm):
    for ax, (metric, label) in zip(axes, METRICS):
        m = tm[tm["metric"] == metric]
        for x, (agent, key) in enumerate(B_AGENTS):
            ax.bar(
                x,
                m[key].mean(),
                width=0.62,
                color=PS.AGENT_COLORS[agent],
                edgecolor="black",
                linewidth=1.0,
                alpha=PS.BAR_ALPHA,
                zorder=2,
            )
        for _, r in m.iterrows():
            c = PS.TYPE_PALETTE[r["task_type"]]
            ys = [r[key] for _, key in B_AGENTS]
            ax.plot([0, 1, 2], ys, "-", color="#555555", lw=1.6, alpha=0.6, zorder=3)
            ax.scatter(
                [0, 1, 2],
                ys,
                s=PS.SCATTER_S * 1.0,
                color=c,
                edgecolors="black",
                linewidths=1.0,
                zorder=4,
                label=TYPE_LABELS[r["task_type"]],
            )
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["", "", ""])
        ax.set_xlim(-0.6, 2.6)
        ax.set_ylabel(METRIC_UNITS[metric])
        ax.grid(visible=False)
        PS.tight_y(ax, 0, max(m[key].max() for _, key in B_AGENTS), pad=0.08)


def main():
    PS.set_theme(font_scale=3.8)
    fits = load_fits()
    md = PS.read_table("model_diff.csv")
    dif = differing(md)

    # 2 x 2: a fit bars | b fine-grained levels ; c model-vs-model scatter | d Centaur advantage bars
    tr = PS.read_table("trajectory.csv")
    tm = task_means(tr)

    fig = plt.figure(figsize=(32, 18.5), dpi=100)
    outer = fig.add_gridspec(
        2,
        1,
        height_ratios=[0.5, 1.0],
        left=0.10,
        right=0.99,
        top=0.95,
        bottom=0.15,
        hspace=0.3,
    )
    top = outer[0].subgridspec(1, 2, width_ratios=[1.0, 2.3], wspace=0.4)
    bottom = outer[1].subgridspec(1, 2, width_ratios=[0.78, 1.75], wspace=0.32)
    ax_a = fig.add_subplot(top[0, 0])
    gs_b = top[0, 1].subgridspec(1, 3, wspace=0.65)
    ax_bs = [fig.add_subplot(gs_b[0, i]) for i in range(3)]
    ax_c = fig.add_subplot(bottom[0, 0])
    ax_d = fig.add_subplot(bottom[0, 1])
    panel_fit_bars(ax_a, fits)
    panel_task_levels(ax_bs, tm)
    panel_scatter(ax_c, md, dif)
    panel_advantage(ax_d, dif)
    ax_b = ax_bs[0]

    fig.canvas.draw()
    pc = ax_c.get_position(original=False)
    pa = ax_a.get_position()
    ax_a.set_position([pc.x0, pa.y0, pc.width, pa.height])
    gap_ab, gap_b, right = 0.075, 0.055, 0.99
    x0 = pc.x1 + gap_ab
    w = (right - x0 - 2 * gap_b) / 3
    for i, ax in enumerate(ax_bs):
        pb = ax.get_position()
        ax.set_position([x0 + i * (w + gap_b), pb.y0, w, pb.height])

    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    b_left = (
        ax_bs[0].get_tightbbox(rend).transformed(fig.transFigure.inverted()).x0
    )  # b's y-label edge
    ylo, yhi = ax_bs[0].get_ylim()
    b_ticks = min(  # b's tick numbers edge
        t.get_window_extent(rend).transformed(fig.transFigure.inverted()).x0
        for t, v in zip(ax_bs[0].get_yticklabels(), ax_bs[0].get_yticks())
        if t.get_text() and ylo <= v <= yhi  # visible ticks only
    )
    b_right = ax_bs[2].get_position().x1
    pc = ax_c.get_position()
    ax_d.set_position([b_ticks + 0.09, pc.y0, (b_right - b_ticks) * 0.26, pc.height])
    fit_hint_labels(fig, ax_d, dif, right_edge=b_right)
    # shift so the task names (left tick labels) start at b's y-label edge
    fig.canvas.draw()
    d_left = ax_d.get_tightbbox(rend).transformed(fig.transFigure.inverted()).x0
    q = ax_d.get_position()
    ax_d.set_position([q.x0 + (b_left - d_left), q.y0, q.width, q.height])

    # Panel letters: one baseline per row (a/b, c/d) and one x per column (a/c, b/d).
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    tb = lambda ax: ax.get_tightbbox(rend).transformed(fig.transFigure.inverted())
    ba, bb_, bc, bd = tb(ax_a), tb(ax_b), tb(ax_c), tb(ax_d)
    x_left = min(ba.x0, bc.x0)
    y_top = max(ba.y1, bb_.y1) + 0.005
    y_bot = max(bc.y1, bd.y1) + 0.005
    for x, y, letter in (
        (x_left, y_top, "a"),
        (b_left, y_top, "b"),
        (x_left, y_bot, "c"),
        (b_left, y_bot, "d"),
    ):
        fig.text(
            x, y, letter, fontsize=LETTER, fontweight="bold", va="bottom", ha="left"
        )
    # Task colours: one legend row under the three b panels.
    th, tl = ax_bs[0].get_legend_handles_labels()
    seen, handles_t, labels_t = set(), [], []
    for h, l in zip(th, tl):
        if l not in seen:
            seen.add(l)
            handles_t.append(h)
            labels_t.append(l)
    b0, b2 = ax_bs[0].get_position(), ax_bs[2].get_position()
    fig.legend(
        handles_t,
        labels_t,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5 * (b0.x0 + b2.x1), b0.y0 - 0.022),
        fontsize=PS.SMALL_LABEL + 7,
        handletextpad=0.2,
        columnspacing=1.2,
        labelspacing=0.3,
    )

    handles = [
        mpatches.Patch(
            facecolor=PS.AGENT_COLORS[m], edgecolor="black", alpha=PS.BAR_ALPHA, label=m
        )
        for m in ("Bayesian", "Human", "Centaur")
    ]
    # Model colours: one legend under the whole figure.
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(
            0.5,
            min(bc.y0, bd.y0) - 0.012,
        ),  # just under the lowest x-label of c / d
        columnspacing=2.0,
        handletextpad=0.5,
    )
    PS.save(fig, FIGURES_DIR, "result5_model_comparison")
    print(
        f"panel d: {len(dif)} differing conditions of {len(md)}; Centaur closer on "
        f"{int((dif['closer'] == 'centaur').sum())}, Bayesian on {int((dif['closer'] == 'bayesian').sum())}"
    )


if __name__ == "__main__":
    main()
