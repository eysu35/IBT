import subprocess
import sys
import textwrap
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import pandas as pd
import seaborn as sns

_ANALYSIS = Path(__file__).resolve().parent
_REPO = _ANALYSIS.parent
sys.path.insert(0, str(_REPO))
from constants import TYPE_ORDER  # noqa: E402

TABLES_DIR = _ANALYSIS / "tables"

AGENT_COLORS = {"Human": "#4b5aa8", "Bayesian": "#0a9396", "Centaur": "#ca6702"}
MODEL_COLORS = {
    **AGENT_COLORS,
    "Random": "#c8c8c8",
    "CentaurIBT": "#eaa865",
    "LLaMA": "#8c8c8c",
}
STAR_COLOR = "#000000"
BAR_ALPHA = 0.85  # fill alpha for bars
BAND_ALPHA_CURVE = 0.2  # fill alpha for 95% bands around learning curves

Z95 = 1.96
AXIS_PAD = (
    0.04  # fraction of the data range left free beyond the outermost point / bar end
)

FONT_SCALE = 2.5  # one font scale for every figure
SMALL_LABEL = 19  # long tick labels (hint text) and in-panel annotations
PANEL_TITLE = 17  # per-hint panel titles in the timing figure
PANEL_LETTER = 36  # lowercase bold panel letters, as in Nature Human Behaviour
SAVE_DPI = 200
BAND_COLOR = "gray"
BAND_ALPHA = 0.12
BAND_LEGEND_ALPHA = 0.3
ZERO_LINE = dict(color="gray", lw=1.2, ls="--", alpha=0.6)
# Mark weights shared by every figure (lines, marker edges, error bars).
LW_HUMAN = 3.5  # human line (timing curves)
LW_MODEL = 3.0  # model lines (timing curves, no-hint baseline lines)
LW_CURVE = 3.0  # learning curves
MARKER_EDGE = 2.4  # edge width of open (model) markers
ERR_LW_HUMAN = 3.0  # human error bars
ERR_LW_MODEL = 2.2  # model error bars
SCATTER_S = 220  # per-condition scatter points (figs 3, 5)
# Line styles when an agent is drawn as a line (hint-timing curves, legend entries).
AGENT_LS = {
    "Human": "-",
    "Bayesian": (0, (1.5, 2.5)),
    "Centaur": (0, (5, 2.5, 1.5, 2.5)),
}

TYPE_LABELS = {
    "stationary": "Stationary",
    "drifting": "Drifting",
    "time_delayed": "Time Delayed",
    "patch_foraging": "Patch Foraging",
    "two_context": "Two Context",
    "three_context": "Three Context",
    "bernoulli": "Bernoulli",
    "linear_contextual": "Linear Contextual",
}
TYPE_PALETTE = dict(zip(TYPE_ORDER, sns.color_palette("tab10", len(TYPE_ORDER))))

# Marker geometry per agent: shape, filled or open, vertical offset within a row, size, z-order.
AGENT_STYLE = {
    "Human": dict(marker="o", filled=True, dy=+0.22, s=340, z=6),
    "Bayesian": dict(marker="s", filled=False, dy=0.0, s=220, z=5),
    "Centaur": dict(marker="D", filled=False, dy=-0.22, s=220, z=4),
}
AGENT_KEY = {"Human": "people", "Bayesian": "bayesian", "Centaur": "centaur"}


def set_theme(font_scale=FONT_SCALE):
    sns.set_theme(
        style="whitegrid",
        font_scale=font_scale,
        rc={
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Arimo", "Liberation Sans", "DejaVu Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.titleweight": "bold",
            "legend.frameon": False,
            "mathtext.default": "regular",
        },
    )


def panel_letter(fig, ax, letter, dx=0.0, dy=0.005):
    fig.canvas.draw()
    bb = ax.get_tightbbox(fig.canvas.get_renderer()).transformed(
        fig.transFigure.inverted()
    )
    fig.text(
        bb.x0 + dx,
        bb.y1 + dy,
        letter,
        fontsize=PANEL_LETTER,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


HINT_DISPLAY = {
    "The points you get on the first pick of a button are the most representative of the button value.": "The points you get on the first pick of a button are the most representative of the button value.",
}


def hint_label(text):
    return HINT_DISPLAY.get(text, text)


def wrap(text, width=38):
    return textwrap.fill(text, width)


_PREFIX_STOP = {
    "a",
    "an",
    "the",
    "of",
    "to",
    "and",
    "but",
    "if",
    "that",
    "your",
    "you",
    "it",
    "with",
    "for",
    "is",
    "are",
    "will",
    "by",
    "in",
    "on",
}


def hint_prefix(text, width=24, lines=2):
    text = hint_label(text)
    if len(textwrap.wrap(text, width)) <= lines:
        return text
    words = text.split()
    while words:
        cand = " ".join(words) + "\u2026"
        if (
            len(textwrap.wrap(cand, width)) <= lines
            and words[-1].lower().strip(",.") not in _PREFIX_STOP
        ):
            return cand
        words.pop()
    return text


def draw_point(
    ax, agent, x, lo, hi, y, scale=1.0, color=None, interval=True, filled=None
):
    st = AGENT_STYLE[agent]
    color = color or AGENT_COLORS[agent]
    fill = st["filled"] if filled is None else filled
    ax.scatter(
        x,
        y,
        color=color if fill else "white",
        marker=st["marker"],
        s=st["s"] * scale,
        edgecolors="black" if st["filled"] else color,
        linewidths=1.8 if st["filled"] else MARKER_EDGE,
        zorder=st["z"],
    )
    if not interval:
        return
    ax.errorbar(
        x,
        y,
        xerr=[[x - lo], [hi - x]],
        fmt="none",
        color=color,
        capsize=5 if st["filled"] else 4,
        lw=ERR_LW_HUMAN if st["filled"] else ERR_LW_MODEL,
        capthick=2.0 if st["filled"] else 1.5,
        alpha=1.0 if st["filled"] else 0.85,
        zorder=st["z"] - 1,
    )


def no_hint_band(ax, lo, hi, axis="x"):
    half = (hi - lo) / 2
    if axis == "x":
        ax.axvspan(-half, half, color=BAND_COLOR, alpha=BAND_ALPHA, zorder=0)
        ax.axvline(0, **ZERO_LINE)
    else:
        ax.axhspan(-half, half, color=BAND_COLOR, alpha=BAND_ALPHA, zorder=0)
        ax.axhline(0, **ZERO_LINE)


def tight_x(ax, lo, hi, pad=AXIS_PAD):
    span = (hi - lo) or 1.0
    ax.set_xlim(lo - pad * span, hi + pad * span)


def tight_y(ax, lo, hi, pad=AXIS_PAD, extra_bottom=0.0):
    span = (hi - lo) or 1.0
    ax.set_ylim(lo - (pad + extra_bottom) * span, hi + pad * span)


def place_stars(ax, star_rows, gap_frac=0.03, star_frac=0.14, fontsize=None):
    xlo, xhi = ax.get_xlim()
    gap, star_w = gap_frac * (xhi - xlo), star_frac * (xhi - xlo)
    need = xhi
    for y, xr, stars in star_rows:
        sx = xr + gap
        kw = dict(fontsize=fontsize) if fontsize else {}
        ax.text(
            sx,
            y,
            stars,
            va="center",
            ha="left",
            color=STAR_COLOR,
            fontweight="bold",
            **kw,
        )
        need = max(need, sx + star_w)
    if need > xhi:
        ax.set_xlim(xlo, need)


def agent_handles(lines=False, markersize=22, model_markersize=18, people=True):
    h = []
    if people:
        h.append(
            mlines.Line2D(
                [],
                [],
                marker="o",
                color=AGENT_COLORS["Human"],
                linestyle="-" if lines else "none",
                linewidth=LW_HUMAN,
                markersize=markersize,
                label="Human",
            )
        )
    for name in ("Bayesian", "Centaur"):
        h.append(
            mlines.Line2D(
                [],
                [],
                marker=AGENT_STYLE[name]["marker"],
                markerfacecolor="white",
                markeredgecolor=AGENT_COLORS[name],
                color=AGENT_COLORS[name],
                markeredgewidth=MARKER_EDGE,
                linestyle=AGENT_LS[name] if lines else "none",
                linewidth=LW_MODEL,
                markersize=model_markersize,
                label=name,
            )
        )
    return h


def band_handle(label="Human no-hint ± 95% CI"):
    return mpatches.Patch(color=BAND_COLOR, alpha=BAND_LEGEND_ALPHA, label=label)


def content_center(fig):
    fig.canvas.draw()
    bb = fig.get_tightbbox(fig.canvas.get_renderer())
    return 0.5 * (bb.x0 + bb.x1) / fig.get_figwidth()


def bottom_legends(fig, rows, cx=None, y0=0.042, dy=0.034, columnspacing=1.6):
    if cx is None:
        cx = content_center(fig)
    for i, handles in enumerate(rows):
        fig.legend(
            handles=handles,
            loc="lower center",
            ncol=len(handles),
            frameon=False,
            bbox_to_anchor=(cx, y0 - i * dy),
            columnspacing=columnspacing,
            handletextpad=0.5,
        )


def git_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=_REPO, text=True
        ).strip()
    except Exception:
        return "unknown"


def save(fig, fig_dir, name, dpi=SAVE_DPI, pad=0.03):
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    meta = {"Creator": f"IBT {git_hash()}"}
    for ext in ("pdf", "png"):
        p = fig_dir / f"{name}.{ext}"
        fig.savefig(
            p,
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=pad,
            metadata=meta if ext == "pdf" else None,
        )
        print(f"Saved: {p}")


def read_table(name):
    p = TABLES_DIR / name
    if not p.exists():
        raise FileNotFoundError(f"{p} missing — run `make tables`")
    return pd.read_csv(p)
