"""House style shared by every paper figure: palette, panel, type sizes, whiskers, value labels, legend.

    import style; style.apply(); fig, axes = style.figure(1, 3)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RANDOM, LEWM, TEMPORAL, QUASIMETRIC = "#D9D9D9", "#7F7F7F", "#1F3A63", "#4C72B0"
PANEL, TEXT, TICK, WHISKER, ZERO = "#EEEEF3", "#222", "#555", "#333333", "#C8C8CE"
HATCH = "////"
PAIR_GAP = 0.04                                       # space between the two bars of a pair, in bar-slot units
FONT, TITLE, LABEL, VALUE, VALUE_SMALL, LEGEND = 13.5, 15, 13.5, 11, 10, 13.5  # 12.5 in canvas at a 5.5 in column: 5.9 / 6.6 / 5.9 / 5.1 / 4.6 / 5.3 pt on the page
FULL, HALF, ROW = 12.5, 6.25, 3.9                    # figure width in inches at full / half column, height per row
DPI = 170


def tint(hex_colour, f=0.55):
    """hex colour blended toward white by f; the hatch line colour for a bar of that colour."""
    r, g, b = (int(hex_colour[i:i + 2], 16) for i in (1, 3, 5))
    return "#%02X%02X%02X" % tuple(int(c + (255 - c) * f) for c in (r, g, b))


def apply():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": FONT, "axes.facecolor": PANEL, "axes.edgecolor": PANEL,
                         "hatch.linewidth": 0.9, "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})


def figure(nrows, ncols, width="full", **kw):
    w = FULL if width == "full" else HALF
    return plt.subplots(nrows, ncols, figsize=(w, ROW * nrows + (0.8 if nrows > 1 else 0)), **kw)


def clean(ax, yticks=None, ylim=None, zero=True):
    """Panel dressing: white horizontal grid, no spines, no tick marks, grey tick labels."""
    if ylim is not None: ax.set_ylim(*ylim)
    if yticks is not None: ax.set_yticks(list(yticks))
    ax.grid(axis="y", color="white", lw=1.2, zorder=0); ax.set_axisbelow(True); ax.tick_params(length=0, colors=TICK)
    if zero: ax.axhline(0, color=ZERO, lw=0.8, zorder=2)
    for sp in ax.spines.values():
        sp.set_visible(False)


def title(ax, text):
    ax.set_title(text, fontsize=TITLE, pad=10, color=TEXT)


def ylabel(ax, text):
    ax.set_ylabel(text, fontsize=LABEL, color="#333")


def whisker(ax, x, lo, hi, half=0.06):
    ax.plot([x, x], [lo, hi], color=WHISKER, lw=1.0, zorder=5, solid_capstyle="butt")
    for yy in (lo, hi):
        ax.plot([x - half, x + half], [yy, yy], color=WHISKER, lw=1.0, zorder=5)


def value(ax, x, y, text, fs=VALUE):
    ax.text(x, y, text, ha="center", va="bottom", fontsize=fs, color=TICK)


def legend(fig, handles, labels, **kw):
    opts = dict(loc="lower center", ncol=len(handles), frameon=False, fontsize=LEGEND, bbox_to_anchor=(0.5, 0.01), handlelength=1.4, columnspacing=2.2)
    opts.update(kw); fig.legend(handles, labels, **opts)


def save(fig, out):
    fig.savefig(out, dpi=DPI, facecolor="white"); print(out)
