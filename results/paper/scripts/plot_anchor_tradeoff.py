"""The stop-gradient ablation in one figure: CEM success (top row, mean over the 20k/24k/28k checkpoints) and
object-position R^2 (bottom row, held-out episodes, final checkpoint) per environment. Random (random policy /
random-init encoder) and LeWM as single bars, then each SpecWM kernel as a pair: default (solid) and with a
stop-gradient on the prediction target (hatched). Bars are seed means, whiskers one cross-seed standard deviation
(CALVIN: one seed). Data: data/finals_seeds.csv, data/cem_3env_floor.csv, data/probe_heldout_unified_<env>.csv.

    python results/paper/scripts/plot_anchor_tradeoff.py --out results/paper/figures/ablations/fig6_stop_gradient.png
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))
import style  # noqa: E402
ENVS = [("cube", "OGBench Cube"), ("scene", "OGBench Scene"), ("calvin", "CALVIN")]
SINGLES = [("random", "#D9D9D9", "Random"), ("sigreg", "#7F7F7F", "LeWM")]
KERNELS = [("SpecWM (temporal)", "#1F3A63", "temporal", "temporal-detached"),
           ("SpecWM (quasimetric)", "#4C72B0", "attached", "detached")]
CALVIN_RUNS = r"random-init|calvin-lewm-mixlate-\d|calvin-mixlate-far-(det-)?\d|calvin-mixlate-lag-far-(det-)?\d"
HATCH = "////"


def tint(hex_colour, f=0.55):
    r, g, b = (int(hex_colour[i:i + 2], 16) for i in (1, 3, 5))
    return "#%02X%02X%02X" % tuple(int(c + (255 - c) * f) for c in (r, g, b))


def cem_table():
    t = pd.read_csv(HERE / "data" / "finals_seeds.csv").groupby(["env", "arm"])["mean_20k_30k"].agg(["mean", "std", "count"])
    floor = pd.read_csv(HERE / "data" / "cem_3env_floor.csv").set_index("env")["random_policy"]
    for env, v in floor.items():
        t.loc[(env, "random"), :] = (float(v), np.nan, 1)
    return t


def probe_table():
    files = sorted((HERE / "data").glob("probe_heldout_unified_*.csv")) + [HERE / "data" / "probe_heldout_paper_calvin.csv"]
    v = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
    v = v[(v.env != "calvin") | v.run.str.match(CALVIN_RUNS)]
    if "subset" in v: v = v[v.subset.fillna("all") == "all"]
    if "centred" in v: v = v[v.centred.astype(str) != "True"]
    if "head" in v: v = v[v["head"].fillna("linear") == "linear"]
    v = v[((v.step == -1) | (v.run == "random-init")) & ~v.run.str.contains(r"-n\d+w-|-qrl-|fs5|grouped|play|kab")]
    v = v.drop_duplicates(subset=["env", "run"], keep="last")
    lag, det = v.run.str.contains("-lag-"), v.run.str.contains("-det-")
    arm = np.where(lag, np.where(det, "temporal-detached", "temporal"), np.where(det, "detached", "attached"))
    arm = np.where(v.run.str.contains("lewm"), "sigreg", np.where(v.run == "random-init", "random", arm))
    return v.assign(arm=arm).groupby(["env", "arm"])["obj"].agg(["mean", "std", "count"])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
    rows = [(cem_table(), "Success (%)", (0, 100), range(0, 101, 20), lambda v: f"{v:.0f}", 1.2),
            (probe_table(), "Object-position R²", (0, 1.08), [0, 0.2, 0.4, 0.6, 0.8, 1.0], lambda v: f"{v:.2f}", 0.013)]
    style.apply()
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 8.6), sharey="row")
    fig.subplots_adjust(left=0.08, right=0.995, top=0.94, bottom=0.14, wspace=0.07, hspace=0.14)
    W, G = 0.53, 0.5
    for r, (tab, ylabel, ylim, yticks, fmt, pad) in enumerate(rows):
        for c, (env, title) in enumerate(ENVS):
            ax = axes[r, c]; x = 0.0

            def bar(xc, arm, colour, fs=style.VALUE, hatch=None, ref=None):
                if (env, arm) not in tab.index:                                  # not yet scored: dashed outline at the default's height
                    ax.bar(xc, ref, W - style.PAIR_GAP, facecolor="none", edgecolor=colour, linewidth=1.0, linestyle=(0, (3, 2)), zorder=3)
                    ax.text(xc, ref + pad, "tbd", ha="center", va="bottom", fontsize=style.VALUE_SMALL, color="#9A9A9A"); return
                m, s, n = tab.loc[(env, arm)]
                ax.bar(xc, m, W - style.PAIR_GAP, color=colour, zorder=3, edgecolor=tint(colour) if hatch else "none", linewidth=0, hatch=hatch)
                top = max(m, 0)
                if n > 1 and np.isfinite(s):
                    for yy in (m - s, m + s):
                        ax.plot([xc - 0.06, xc + 0.06], [yy, yy], color="#333333", lw=1.0, zorder=5)
                    ax.plot([xc, xc], [m - s, m + s], color="#333333", lw=1.0, zorder=5)
                    top = m + s
                ax.text(xc, top + pad, fmt(m), ha="center", va="bottom", fontsize=fs, color="#555")

            for arm, colour, _ in SINGLES:
                bar(x, arm, colour); x += W + G
            for _, colour, att, det in KERNELS:
                bar(x, att, colour, fs=style.VALUE_SMALL); bar(x + W, det, colour, fs=style.VALUE_SMALL, hatch=HATCH, ref=float(tab.loc[(env, att), "mean"]))
                x += 2 * W + G
            ax.set_xlim(-0.5, x - G - W + 0.5); ax.set_xticks([])
            ax.set_ylim(*ylim); ax.set_yticks(list(yticks)); ax.grid(axis="y", color="white", lw=1.2, zorder=0); ax.set_axisbelow(True)
            ax.tick_params(length=0, colors="#555"); ax.axhline(0, color="#C8C8CE", lw=0.8, zorder=2)
            for sp in ax.spines.values():
                sp.set_visible(False)
            if r == 0:
                ax.set_title(title, fontsize=style.TITLE, pad=10, color="#222")
        axes[r, 0].set_ylabel(ylabel, fontsize=style.LABEL, color="#333")
    h = [mp.Patch(facecolor=c, edgecolor="none") for _, c, _ in SINGLES] + [mp.Patch(facecolor=c, edgecolor="none") for _, c, _, _ in KERNELS]
    h.append(mp.Patch(facecolor="#6E6E74", edgecolor=tint("#6E6E74"), hatch=HATCH, linewidth=0))
    lab = [l for _, _, l in SINGLES] + [n for n, _, _, _ in KERNELS] + ["stop-gradient on prediction target"]
    fig.legend(h, lab, loc="lower center", ncol=3, frameon=False, fontsize=style.LEGEND, bbox_to_anchor=(0.53, 0.005), handlelength=1.6, columnspacing=2.0)
    style.save(fig, a.out)


if __name__ == "__main__":
    main()
