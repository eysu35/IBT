import sys
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # analysis/, for exp_paths
from exp_paths import SIM_DIR, FIG_DIR
import freq_lmm as F
import paper_style as PS

_REPO_ROOT = Path(__file__).resolve().parents[1]

HUMAN_CSV = _REPO_ROOT / "behavioral/hint_time/analysis/hint_effects.csv"
CENTAUR_HT = _REPO_ROOT / "centaur/data/centaur_hint_time_clean.parquet"
BAYES_TIMING = SIM_DIR / "bayesian_timing_trials.parquet"
FIGURES_DIR = FIG_DIR

TYPE_ORDER = ["stationary", "patch_foraging"]
TYPE_LABELS = {t: PS.TYPE_LABELS[t] for t in TYPE_ORDER}
TIMINGS = np.array([0, 4, 8, 12])

MODEL_TO_HUMAN = {
    "{min2} is the worst.": "Avoid {min1} and {min2}.",
    "{max1} is better than {min2}, {min1}, and {mid}.": "{max1} and {max2} are better than the others.",
}


def _norm(h):
    return MODEL_TO_HUMAN.get(h, h)


def plot_panel(ax, sub, band, ylim):
    PS.no_hint_band(ax, band[0], band[1], axis="y")
    cells = {
        a: sub[sub["agent"] == key].set_index("onset").reindex(TIMINGS)
        for a, key in PS.AGENT_KEY.items()
    }
    for agent in ("Centaur", "Bayesian"):
        st, col = PS.AGENT_STYLE[agent], PS.AGENT_COLORS[agent]
        c = cells[agent]
        ax.plot(
            TIMINGS,
            c["rr_own_base"],
            ls=PS.AGENT_LS[agent],
            color=col,
            lw=PS.LW_MODEL,
            alpha=0.9,
            zorder=st["z"] - 1,
        )
        ax.errorbar(
            TIMINGS,
            c["rr_own_base"],
            yerr=[c["rr_own_base"] - c["rr_own_lo"], c["rr_own_hi"] - c["rr_own_base"]],
            fmt="none",
            color=col,
            capsize=4,
            lw=PS.ERR_LW_MODEL,
            capthick=1.5,
            alpha=0.85,
            zorder=st["z"] - 1,
        )
        ax.scatter(
            TIMINGS,
            c["rr_own_base"],
            marker=st["marker"],
            s=st["s"] * 0.7,
            color="white",
            edgecolors=col,
            linewidths=PS.MARKER_EDGE,
            zorder=st["z"],
        )
    h = cells["Human"]
    st, col = PS.AGENT_STYLE["Human"], PS.AGENT_COLORS["Human"]
    ax.plot(
        TIMINGS,
        h["rr_own_base"],
        ls=PS.AGENT_LS["Human"],
        color=col,
        lw=PS.LW_HUMAN,
        zorder=st["z"] - 1,
    )
    ax.errorbar(
        TIMINGS,
        h["rr_own_base"],
        yerr=[h["rr_own_base"] - h["rr_own_lo"], h["rr_own_hi"] - h["rr_own_base"]],
        fmt="none",
        color=col,
        capsize=5,
        lw=PS.ERR_LW_HUMAN,
        capthick=2.0,
        zorder=st["z"] - 1,
    )
    ax.scatter(
        TIMINGS,
        h["rr_own_base"],
        marker=st["marker"],
        s=st["s"] * 0.75,
        color=col,
        edgecolors="black",
        linewidths=1.8,
        zorder=st["z"],
    )
    ax.set_xticks(TIMINGS)
    ax.set_xlim(-1.0, 13.0)
    ax.set_ylim(*ylim)


TRIALS_PARQUET = _REPO_ROOT / "behavioral/hint_time/data/behavioral_clean.parquet"

ONSET_COLORS = {0: "#d98bb4", 4: "#c9608e", 8: "#9c2a61", 12: "#5a0f36"}

# Short labels for the Study 2 (human) wording of every hint.
TITLE_WRAP = 24  # characters per title line; hints longer than two lines are cut to a verbatim prefix

TASK_LS = {"stationary": "-", "patch_foraging": (0, (4, 2))}
GREY = "#b0b0b0"
FONT_SCALE = 3.4  # larger than the shared default (2.5): this figure is wide and was reading small
LAB = PS.SMALL_LABEL + 7  # tags, stars, curve labels
TITLE = PS.PANEL_TITLE + 11  # panel-c titles


def load_curves():
    df = F.normalize_regret(pd.read_parquet(TRIALS_PARQUET))
    df = df[df["ogHint"] == "No hint."]
    per_subj = df.groupby(["type", "user_id", "trial"])["regret"].mean().reset_index()
    g = per_subj.groupby(["type", "trial"])["regret"]
    out = g.mean().to_frame("mean")
    out["se"] = g.sem()
    out["n"] = g.size()
    return out.reset_index()


def load_people(timing):
    stars = pd.read_csv(HUMAN_CSV)
    stars = stars[stars["ogHint"] != "No hint."][
        ["type", "ogHint", "hint_timing", "stars"]
    ]
    stars = stars.rename(
        columns={"type": "task_type", "ogHint": "hint_text", "hint_timing": "onset"}
    )
    people = timing[timing["agent"] == "people"].merge(
        stars, on=["task_type", "hint_text", "onset"], how="left"
    )
    people["stars"] = people["stars"].fillna("")
    return people


def panel_a(ax, curves):
    col = PS.AGENT_COLORS["Human"]
    for btype in TYPE_ORDER:
        c = curves[curves["type"] == btype].sort_values("trial")
        x, m, se = c["trial"].to_numpy(), c["mean"].to_numpy(), c["se"].to_numpy()
        ax.fill_between(
            x,
            m - PS.Z95 * se,
            m + PS.Z95 * se,
            color=col,
            alpha=PS.BAND_ALPHA_CURVE,
            lw=0,
            zorder=2,
        )
        ax.plot(x, m, ls=TASK_LS[btype], color=col, lw=PS.LW_CURVE, zorder=3)
        ax.annotate(
            TYPE_LABELS[btype],
            xy=(x[-1], m[-1]),
            xytext=(0, 14),
            textcoords="offset points",
            fontsize=LAB,
            color=col,
            fontweight="bold",
            va="bottom",
            ha="right",
        )
        for onset in TIMINGS:
            ax.scatter(
                onset,
                m[x == onset],
                marker="o",
                s=PS.SCATTER_S * 0.9,
                color=ONSET_COLORS[onset],
                edgecolors="black",
                linewidths=1.5,
                zorder=6,
            )
    for onset in TIMINGS:
        ax.axvline(
            onset, color=ONSET_COLORS[onset], lw=1.2, ls=":", alpha=0.7, zorder=1
        )
    ax.set_xticks(TIMINGS.tolist() + [16, 19])
    ax.set_xlim(-0.6, 19.8)
    ax.set_xlabel("Trial")
    ax.set_ylabel("Regret (%)")
    lo = (curves["mean"] - PS.Z95 * curves["se"]).min()
    hi = (curves["mean"] + PS.Z95 * curves["se"]).max()
    PS.tight_y(ax, min(0, lo), hi, pad=0.06)


def spread(ax, ys, min_px):
    order = np.argsort(ys)
    px = [ax.transData.transform((0, ys[i]))[1] for i in order]
    for k in range(1, len(px)):
        if px[k] - px[k - 1] < min_px:
            px[k] = px[k - 1] + min_px
    out = np.empty(len(ys))
    for k, i in enumerate(order):
        out[i] = ax.transData.inverted().transform((0, px[k]))[1]
    return out


def panel_b(ax, people):
    ax.axhline(0, **PS.ZERO_LINE)
    col = PS.AGENT_COLORS["Human"]
    keys = [
        (t, h) for (t, h), _ in people.groupby(["task_type", "hint_text"], sort=False)
    ]
    sig = {
        k
        for k in keys
        if (
            people[(people.task_type == k[0]) & (people.hint_text == k[1])]["stars"]
            != ""
        ).any()
    }
    star_cells, tagged = [], []
    for t, h in keys:
        g = (
            people[(people.task_type == t) & (people.hint_text == h)]
            .set_index("onset")
            .reindex(TIMINGS)
        )
        y = g["rr_own_base"].to_numpy()
        hl = (t, h) in sig
        ax.plot(
            TIMINGS,
            y,
            "-",
            color=col if hl else GREY,
            lw=PS.LW_HUMAN if hl else 2.0,
            zorder=5 if hl else 2,
        )
        for x, yy in zip(TIMINGS, y):
            ax.scatter(
                x,
                yy,
                marker="o",
                s=PS.SCATTER_S * (0.8 if hl else 0.45),
                color=ONSET_COLORS[x],
                edgecolors="black" if hl else "white",
                linewidths=1.4 if hl else 0.8,
                zorder=6 if hl else 3,
            )
        if hl:
            star_cells += [
                (x, yy, st) for x, yy, st in zip(TIMINGS, y, g["stars"]) if st
            ]
            tagged.append((y[0], t, h))
    tagged.sort(reverse=True)  # largest effect at onset 0 first = c1
    ax.set_xticks(TIMINGS)
    ax.set_xlim(-3.0, 12.9)
    ax.set_xlabel("Hint onset (trial)")
    ax.set_ylabel("Regret reduction (%)")
    lo, hi = people["rr_own_base"].min(), people["rr_own_base"].max()
    PS.tight_y(ax, lo, hi + 1.5, pad=0.08)
    ys = spread(
        ax, np.array([y0 for y0, _, _ in tagged]), min_px=LAB * 1.6 * ax.figure.dpi / 72
    )
    for k, ((y0, t, h), y) in enumerate(zip(tagged, ys), start=1):
        ax.annotate(
            f"c{k}",
            xy=(0, y0),
            xytext=(-0.6, y),
            fontsize=LAB,
            color=col,
            fontweight="bold",
            va="center",
            ha="right",
            zorder=8,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=col, lw=1.2),
        )
    for x in TIMINGS:
        cells = [(yy, st) for xx, yy, st in star_cells if xx == x]
        if not cells:
            continue
        ys = spread(
            ax, np.array([c[0] for c in cells]), min_px=LAB * 1.3 * ax.figure.dpi / 72
        )
        for (yy, st), y in zip(cells, ys):
            ax.annotate(
                st,
                xy=(x, y),
                xytext=(0, 9),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=LAB,
                fontweight="bold",
                color=PS.STAR_COLOR,
                zorder=7,
            )
    return [(t, h) for _, t, h in tagged]


def onset_handles():
    return [
        mlines.Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            color=ONSET_COLORS[o],
            markeredgecolor="black",
            markeredgewidth=1.2,
            markersize=17,
            label=f"Hint onset {o}",
        )
        for o in TIMINGS
    ]


def main():
    PS.set_theme(font_scale=FONT_SCALE)
    timing = PS.read_table("timing.csv")
    curves = load_curves()

    fig = plt.figure(figsize=(36, 10.2), dpi=100)
    TOP, BOTTOM = 0.80, 0.285
    # a | b | c1 c2 c3 c4 : equal gaps between a, b and the c group; the c axes sit close together.
    spans = [
        (0.045, 0.205),
        (0.265, 0.425),
        (0.485, 0.60),
        (0.615, 0.73),
        (0.745, 0.86),
        (0.875, 0.99),
    ]
    axes = [fig.add_axes([l, BOTTOM, r - l, TOP - BOTTOM]) for l, r in spans]
    ax_a, ax_b, ex_axes = axes[0], axes[1], axes[2:]

    panel_a(ax_a, curves)
    examples = panel_b(ax_b, load_people(timing))

    subs, bands = [], []
    for btype, hint in examples:
        sub = timing[(timing["task_type"] == btype) & (timing["hint_text"] == hint)]
        h0 = sub[sub["agent"] == "people"].iloc[0]
        subs.append(sub)
        bands.append((h0["human_nohint_lo"], h0["human_nohint_hi"]))
    lo = min(min(s["rr_own_lo"].min(), s["rr_own_base"].min()) for s in subs)
    hi = max(max(s["rr_own_hi"].max(), s["rr_own_base"].max()) for s in subs)
    half = max((b[1] - b[0]) / 2 for b in bands)
    lo, hi = min(lo, -half), max(hi, half)
    span = hi - lo
    ylim = (lo - PS.AXIS_PAD * span, hi + PS.AXIS_PAD * span)
    for k, (ax, (btype, hint), sub, band) in enumerate(
        zip(ex_axes, examples, subs, bands), start=1
    ):
        plot_panel(ax, sub, band, ylim)
        ax.set_title(
            f"c{k}  {TYPE_LABELS[btype]}\n{PS.wrap(PS.hint_prefix(hint, TITLE_WRAP), TITLE_WRAP)}",
            fontsize=TITLE,
            pad=10,
        )
        ax.set_xlabel("Hint onset (trial)")
    ex_axes[0].set_ylabel("Regret reduction (%)")
    for ax in ex_axes[1:]:
        ax.tick_params(labelleft=False)

    # Every y-label the same distance left of its axis; panel letters on one line above the labels.
    fig.canvas.draw()
    YLAB_GAP = 0.031  # figure fraction (~1.1 in)
    for ax in (ax_a, ax_b, ex_axes[0]):
        w = ax.get_position().width
        ax.yaxis.set_label_coords(-YLAB_GAP / w, 0.5)
    fig.canvas.draw()
    for ax, letter in zip((ax_a, ax_b, ex_axes[0]), "abc"):
        bb = ax.yaxis.label.get_window_extent().transformed(fig.transFigure.inverted())
        fig.text(
            bb.x0,
            0.94,
            letter,
            fontsize=PS.PANEL_LETTER,
            fontweight="bold",
            va="top",
            ha="left",
        )

    PS.bottom_legends(
        fig,
        [
            onset_handles(),
            PS.agent_handles(lines=True, markersize=20, model_markersize=18)
            + [PS.band_handle()],
        ],
        y0=0.02,
        dy=0.062,
    )
    PS.save(fig, FIGURES_DIR, "result4_hint_timing")


if __name__ == "__main__":
    main()
