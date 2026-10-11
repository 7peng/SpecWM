"""The two Scene ablations of Appendix B (CEM cost: L2 vs cosine; normalisation layer: BatchNorm vs LayerNorm) for LeWM
and SpecWM (temporal), in the house style, on the common full-width canvas with each panel at Fig. 3's panel size and bar geometry, centred (so every
figure prints at one scale at width=\linewidth). Bars are the displayed aggregates recorded in data/scene_ablations_displayed.json
(cost: three seeds at their final three checkpoints; normalisation: one seed, final three checkpoints), whiskers the
recorded error magnitudes.

    python results/paper/scripts/plot_scene_ablations.py --which both --out results/paper/figures/ablations/figB_scene_ablations.png
    python results/paper/scripts/plot_scene_ablations.py --which cost --out results/paper/figures/ablations/fig8_cost_ablation.png   # likewise normalization
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))
import style  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

LABELS = {"cost": ["LeWM\nL2", "LeWM\nCosine", "Temporal\nL2", "Temporal\nCosine"],
          "normalization": ["LeWM\nBatchNorm", "LeWM\nLayerNorm", "Temporal\nBatchNorm", "Temporal\nLayerNorm"]}


PANEL_W = 0.935 / (3 + 2 * 0.07)          # Fig. 3: three panels over 93.5% of the width, wspace 0.07
PANEL_GAP = PANEL_W * 0.07


def panel(fig, ax, which, d):
    means, errs = d["displayed_means"], d["displayed_error_magnitudes"]
    W, G = 0.94, 0.48                                                                 # Fig. 3 bar width and gap
    xs = [i * (W + G) for i in range(4)]; colours = [style.LEWM] * 2 + [style.TEMPORAL] * 2
    for x, m, e, c in zip(xs, means, errs, colours):
        ax.bar(x, m, W - 0.04, color=c, zorder=3)
        style.whisker(ax, x, m - e, m + e)
        style.value(ax, x, m + e + 1.2, f"{m:.0f}")
    ax.set_xticks(xs, LABELS[which], fontsize=style.VALUE_SMALL); ax.set_xlim(-0.72, xs[-1] + 0.72)
    style.clean(ax, yticks=range(0, 101, 20), ylim=(0, 100))
    style.title(ax, "OGBench Scene")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--which", choices=[*LABELS, "both"], required=True); ap.add_argument("--out", required=True); a = ap.parse_args()
    data = json.load(open(HERE / "data" / "scene_ablations_displayed.json"))
    which = list(LABELS) if a.which == "both" else [a.which]
    style.apply()
    if len(which) == 1:                                   # one panel on its own tight canvas, a little taller than a Fig. 3 row
        pw = PANEL_W * style.FULL; fig = plt.figure(figsize=(pw + 1.0, 4.6))
        axes = [fig.add_axes([0.8 / (pw + 1.0), 0.19, pw / (pw + 1.0), 0.71])]
    else:                                                 # both panels centred on the common canvas
        fig = plt.figure(figsize=(style.FULL, style.ROW))
        total = len(which) * PANEL_W + (len(which) - 1) * PANEL_GAP; left = 0.5 - total / 2
        axes = [fig.add_axes([left + i * (PANEL_W + PANEL_GAP), 0.27, PANEL_W, 0.61]) for i in range(len(which))]
    for ax, w in zip(axes, which):
        panel(fig, ax, w, data[w])
    for ax in axes[1:]:
        ax.tick_params(labelleft=False)
    style.ylabel(axes[0], "Success (%)")
    style.save(fig, a.out)


if __name__ == "__main__":
    main()
