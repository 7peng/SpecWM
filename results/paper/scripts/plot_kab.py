"""Centering / tempering ablation of the fully specified state kernel on OGBench Scene, in the Fig. 4 bar design: one
group per probe target plus the mean, four bars per group (the paper form solid; without centering, without tempering
and with neither in the hatches of the legend). Held-out linear-probe R^2 at the final checkpoint from
data/probe_heldout_unified_scene.csv.

    python results/paper/scripts/plot_kab.py --out results/paper/figA_kab.png
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mp
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
STEM = "scene-kab-full-state-buttons-pa05-rel-"
ARMS = [("raw kernel", STEM + "raw-s2", "xxxx"), ("+ centering", STEM + "notemper-s2", "...."),
        ("+ tempering", STEM + "nocenter-s2", "////"), ("+ centering + tempering", STEM + "s2", None)]
TARGETS = [("cube_xyz", "cube"), ("arm", "arm"), ("drawer", "drawer"), ("window", "window"), ("buttons", "buttons")]
COLOUR = "#4C72B0"


def tint(hex_colour, f=0.55):
    r, g, b = (int(hex_colour[i:i + 2], 16) for i in (1, 3, 5))
    return "#%02X%02X%02X" % tuple(int(c + (255 - c) * f) for c in (r, g, b))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
    v = pd.read_csv(HERE / "data" / "probe_heldout_unified_scene.csv")
    if "subset" in v: v = v[v.subset.fillna("all") == "all"]
    if "head" in v: v = v[v["head"].fillna("linear") == "linear"]
    v = v[v.step == -1].drop_duplicates("run", keep="last").set_index("run")
    cols = [c for c, _ in TARGETS]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "axes.facecolor": "#EEEEF3", "axes.edgecolor": "#EEEEF3", "hatch.linewidth": 0.9})
    fig, ax = plt.subplots(figsize=(8.4, 3.9))
    fig.subplots_adjust(left=0.08, right=0.99, top=0.9, bottom=0.26)
    W, G = 0.2, 0.32; x = 0.0; ticks, labels = [], []
    for col, label in TARGETS + [("mean", "mean")]:
        for k, (_, run, hatch) in enumerate(ARMS):
            if run not in v.index: continue
            val = float(v.loc[run, cols].mean()) if col == "mean" else float(v.loc[run, col])
            ax.bar(x + k * W, val, W - 0.02, color=COLOUR, zorder=3, edgecolor=tint(COLOUR) if hatch else "none", linewidth=0, hatch=hatch)
            ax.text(x + k * W, val + 0.012, f"{val:.2f}", ha="center", va="bottom", fontsize=7.5, color="#555", rotation=90)
        ticks.append(x + 1.5 * W); labels.append(label); x += 4 * W + G
    ax.set_xticks(ticks); ax.set_xticklabels(labels); ax.set_xlim(-0.3, x - G + 0.1)
    ax.set_ylim(0, 1.15); ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0]); ax.grid(axis="y", color="white", lw=1.2, zorder=0); ax.set_axisbelow(True)
    ax.tick_params(length=0, colors="#555"); ax.axhline(0, color="#C8C8CE", lw=0.8, zorder=2)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title("Fully specified state kernel, OGBench Scene", fontsize=13.5, pad=10, color="#222"); ax.set_ylabel("R²", fontsize=12, color="#333")
    hs = [mp.Patch(facecolor=COLOUR, edgecolor=tint(COLOUR) if h else "none", hatch=h, linewidth=0) for _, _, h in ARMS]
    fig.legend(hs, [l for l, _, _ in ARMS], loc="lower center", ncol=4, frameon=False, fontsize=11, bbox_to_anchor=(0.5, 0.01), handlelength=1.4, columnspacing=2.0)
    fig.savefig(a.out, dpi=170, facecolor="white"); print(a.out)


if __name__ == "__main__":
    main()
