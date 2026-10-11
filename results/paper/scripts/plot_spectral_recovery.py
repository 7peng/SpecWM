"""Figure 5: linear-probe R^2 of each kernel eigenvector from the final embedding (9-mode running mean) for the
fully specified and underspecified arms and the untrained encoder, one panel per kernel (state, proprioception,
depth). Reads battery/conversion.csv (results/paper/scripts/spectral_battery.py).

    python results/paper/scripts/plot_spectral_recovery.py --out results/paper/figures/ablations/fig5_spectral_recovery.png
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))
import style  # noqa: E402
KERNELS = ["state", "proprio", "depth"]
HUE = {"state": "#55A868", "proprio": "#8172B3", "depth": "#C44E52"}
TINT = {"state": "#A8D3B3", "proprio": "#C0B8D9", "depth": "#E2A6A8"}
DARK = "#444444"
MODES = 128
BANDS = ((0, 8, "#DCDCDC"), (8, 32, "#E4E4E4"), (32, 128, "#EBEBEB"))


def clean(ax):
    ax.grid(color="white", lw=1.2, zorder=0); ax.set_axisbelow(True); ax.tick_params(length=0, colors=style.TICK)
    for sp in ax.spines.values():
        sp.set_visible(False)


def final_modes(conv, k, run):
    g = conv[(conv.kernel == k) & (conv.run == run)].sort_values("step")
    return g[[f"m{i}" for i in range(MODES)]].to_numpy()[-1]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
    conv = pd.read_csv(HERE / "battery" / "conversion.csv")
    style.apply()
    fig, axes = plt.subplots(1, 3, figsize=(style.FULL, 4.3), sharey=True)
    fig.subplots_adjust(left=0.095, right=0.985, top=0.89, bottom=0.26, wspace=0.12)
    x = np.arange(1, MODES + 1)
    for c, k in enumerate(KERNELS):
        lines = (("Fully specified", k + "_full", HUE[k], 2.4, "-"), ("Underspecified", k, TINT[k], 2.4, "-"), ("Untrained encoder", "untrained", DARK, 1.0, (0, (2, 2))))
        ax = axes[c]
        for lab, run, col, lw, ls in lines:
            y = pd.Series(final_modes(conv, k, run)).rolling(9, center=True, min_periods=1).mean().to_numpy()
            ax.plot(x, y, color=col, lw=lw, ls=ls, label=lab)
        for lo, hi, shade in BANDS:
            ax.axvspan(lo + 0.5, hi + 0.5, color=shade, zorder=0, lw=0)
        ax.set_xlim(1, MODES); ax.set_ylim(0, 1); ax.set_xticks([1, 32, 64, 96, 128]); clean(ax); style.title(ax, f"{k.capitalize()} kernel"); ax.set_xlabel("Eigenvector index", fontsize=style.LABEL)
    style.ylabel(axes[0], "R² of eigenvector\nfrom embedding")
    h, l = axes[0].get_legend_handles_labels()
    style.legend(fig, h, l, ncol=3, handlelength=2.0)
    style.save(fig, a.out)


if __name__ == "__main__":
    main()
