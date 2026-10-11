"""Figure 8: CEM success across the three environments, mean over the 20k-30k snapshots (20k, 24k, 28k) and
over training seeds, in the grouped design of Fig. 4: the random policy and LeWM as single bars, then one group
per kernel with the prediction anchor attached (solid) and detached (hatched in a tint of the kernel's colour).
A thin whisker is +/- one sample standard deviation over seeds where more than one seed exists. Values from
data/finals_seeds.csv (scripts/seed_aggregate.py: seed 1 here, seeds 2-3 the collaborator's replicates), floors from
data/cem_3env_floor.csv. Scene and cube at O25, CALVIN at O50, 100 tasks (offsets reported in the appendix, not on the figure).

    python results/paper/scripts/plot_cem_3env.py --out results/paper/fig8_cem_3env.png
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mp
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))
import style  # noqa: E402
ENVS = [("cube", "OGBench Cube"), ("scene", "OGBench Scene"), ("calvin", "CALVIN")]
SINGLES = [("random", "#D9D9D9", "Random"), ("sigreg", "#7F7F7F", "LeWM")]
NAMES = {"Temporal": "SpecWM (temporal)", "Quasimetric": "SpecWM (quasimetric)"}
KERNELS = [("Temporal", "#1F3A63", "temporal", "temporal-detached"), ("Quasimetric", "#4C72B0", "attached", "detached")]
HATCH = "////"


def tint(hex_colour, f=0.55):
    r, g, b = (int(hex_colour[i:i + 2], 16) for i in (1, 3, 5))
    return "#%02X%02X%02X" % tuple(int(c + (255 - c) * f) for c in (r, g, b))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True)
    ap.add_argument("--col", default="mean_20k_30k", help="finals.csv column to draw (mean_20k_30k | final | late_mean)")
    ap.add_argument("--ablation", action="store_true", help="attached and detached anchor side by side (appendix); default attached only")
    a = ap.parse_args()
    f = pd.read_csv(HERE / "data" / "finals_seeds.csv")
    agg = f.groupby(["env", "arm"])[a.col].agg(["mean", "std", "count"])
    floor = pd.read_csv(HERE / "data" / "cem_3env_floor.csv").set_index("env")["random_policy"]
    style.apply()
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.9), sharey=True)
    fig.subplots_adjust(left=0.07, right=0.995, top=0.88, bottom=0.22, wspace=0.07)
    W, G = 0.53, 0.5
    if not a.ablation: W, G = 0.94, 0.48                                      # attached-only: fat bars, labels clear
    for ax, (env, title) in zip(axes, ENVS):
        g = agg.loc[env] if env in agg.index.get_level_values(0) else agg.iloc[0:0]; x = 0.0; ticks, labels = [], []

        def bar(xc, arm, colour, width, fs=style.VALUE, hatch=None):
            val = float(floor[env]) if arm == "random" else float(g.loc[arm, "mean"])
            ax.bar(xc, val, width, color=colour, zorder=3, edgecolor=tint(colour) if hatch else "none", linewidth=0, hatch=hatch)
            top = val
            if arm != "random" and g.loc[arm, "count"] > 1:                       # +/- one sample standard deviation over seeds
                lo, hi = val - g.loc[arm, "std"], val + g.loc[arm, "std"]
                ax.plot([xc, xc], [lo, hi], color="#333333", lw=1.0, zorder=5, solid_capstyle="butt")
                ax.plot([xc - 0.06, xc + 0.06], [lo, lo], color="#333333", lw=1.0, zorder=5); ax.plot([xc - 0.06, xc + 0.06], [hi, hi], color="#333333", lw=1.0, zorder=5)
                top = hi
            ax.text(xc, top + 1.2, f"{val:.0f}", ha="center", va="bottom", fontsize=fs, color="#555")

        for arm, colour, lab in SINGLES:
            bar(x, arm, colour, W - style.PAIR_GAP); ticks.append(x); labels.append(lab); x += W + G
        for kernel, colour, att, det in KERNELS:
            if not a.ablation:
                bar(x, att, colour, W - style.PAIR_GAP); ticks.append(x); labels.append(kernel); x += W + G
                continue
            for k, arm in enumerate((att, det)):
                if arm in g.index:
                    bar(x + k * W, arm, colour, W - style.PAIR_GAP, fs=style.VALUE_SMALL, hatch=HATCH if k else None)
                else:
                    ref = float(g.loc[att, "mean"])
                    ax.bar(x + k * W, ref, W - style.PAIR_GAP, facecolor="none", edgecolor=colour, linewidth=1.0, linestyle=(0, (3, 2)), zorder=3)
                    ax.text(x + k * W, ref + 1.2, "tbd", ha="center", va="bottom", fontsize=style.VALUE_SMALL, color="#9A9A9A")
            ticks.append(x + W / 2); labels.append(kernel); x += W * 2 + G
        ax.set_xticks([]); ax.set_xlim(-0.5, x - G - W + 0.5) if a.ablation else ax.set_xlim(-0.72, x - W - G + 0.72)
        ax.set_ylim(0, 100); ax.set_yticks(range(0, 101, 20)); ax.grid(axis="y", color="white", lw=1.2, zorder=0); ax.set_axisbelow(True); ax.tick_params(length=0, colors="#555")
        ax.axhline(0, color="#C8C8CE", lw=0.8, zorder=2)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(title, fontsize=style.TITLE, pad=10, color="#222")
    axes[0].set_ylabel("Success (%)", fontsize=style.LABEL, color="#333")
    hs = [mp.Patch(facecolor=c, edgecolor="none") for _, c, _ in SINGLES] + [mp.Patch(facecolor=c, edgecolor="none") for _, c, _, _ in KERNELS]
    ls = [lab for _, _, lab in SINGLES] + [NAMES[k] for k, _, _, _ in KERNELS]
    if a.ablation:
        hs.append(mp.Patch(facecolor="#6E6E74", edgecolor=tint("#6E6E74"), hatch=HATCH, linewidth=0)); ls.append("prediction anchor detached")
    fig.legend(hs, ls, loc="lower center", ncol=len(hs), frameon=False, fontsize=style.LEGEND, bbox_to_anchor=(0.5, 0.01), handlelength=1.4, columnspacing=2.2)
    fig.subplots_adjust(bottom=0.17)
    style.save(fig, a.out)


if __name__ == "__main__":
    main()
