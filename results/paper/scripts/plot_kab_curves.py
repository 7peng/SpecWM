"""Convergence of the centering / tempering ablation: held-out linear-probe R^2 against training step for the four
forms of the fully specified state kernel on OGBench Scene (mean over the five targets, left; cube position, right).
Reads data/probe_heldout_unified_scene.csv (every 2k snapshot to 16k, then every 4k).

    python results/paper/scripts/plot_kab_curves.py --out results/paper/figA_kab_curves.png
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
STEM = "scene-kab-full-state-buttons-pa05-rel-"
ARMS = [("raw kernel", STEM + "raw-s2", "-."), ("+ centering", STEM + "notemper-s2", ":"),
        ("+ tempering", STEM + "nocenter-s2", "--"), ("+ centering + tempering", STEM + "s2", "-")]
TARGETS = ["cube_xyz", "arm", "drawer", "window", "buttons"]
COLOUR = "#4C72B0"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
    v = pd.read_csv(HERE / "data" / "probe_heldout_unified_scene.csv")
    if "subset" in v: v = v[v.subset.fillna("all") == "all"]
    if "head" in v: v = v[v["head"].fillna("linear") == "linear"]
    v = v[v.step > 0].drop_duplicates(["run", "step"], keep="last")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "axes.facecolor": "#EEEEF3", "axes.edgecolor": "#EEEEF3"})
    fig, (ax_m, ax_c) = plt.subplots(1, 2, figsize=(10.5, 3.9))
    fig.subplots_adjust(left=0.07, right=0.99, top=0.88, bottom=0.3, wspace=0.18)
    for label, run, ls in ARMS:
        r = v[v.run == run].sort_values("step")
        if r.empty: continue
        ax_m.plot(r.step / 1000, r[TARGETS].mean(1), ls, color=COLOUR, lw=2.0, marker="o", ms=3.5, label=label, zorder=3)
        ax_c.plot(r.step / 1000, r["cube_xyz"], ls, color=COLOUR, lw=2.0, marker="o", ms=3.5, zorder=3)
    for ax, title in ((ax_m, "mean over targets"), (ax_c, "cube position")):
        ax.set_xlim(0, 33); ax.set_xticks([4, 8, 12, 16, 20, 24, 28, 32]); ax.set_xlabel("Training step (k)")
        ax.set_ylim(0, 1.05); ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.grid(color="white", lw=1.2, zorder=0); ax.set_axisbelow(True); ax.tick_params(length=0, colors="#555")
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(title, fontsize=13, pad=8, color="#222")
    ax_m.set_ylabel("Held-out probe R²")
    h, l = ax_m.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=11, bbox_to_anchor=(0.5, 0.01), handlelength=2.6, columnspacing=2.0)
    fig.savefig(a.out, dpi=170, facecolor="white"); print(a.out)


if __name__ == "__main__":
    main()
