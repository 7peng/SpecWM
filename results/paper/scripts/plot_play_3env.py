"""Noised-data ablation across the three environments in the Fig. 3 / Fig. 4 design (one 12.5 x 3.9 figure per metric):
Random as a single bar, then LeWM and SpecWM (temporal) each as a pair, trained on the late-onset mixture (solid) and on
the original play data only (hatched in a tint). `--metric cem` draws success (mean over the 20k, 24k, 28k snapshots and
over seeds, whisker +/- one SD); `--metric probe` draws held-out object-position R^2 at the final checkpoint.
`--metric both` stacks the two rows (CEM above probes, the Fig. 6 layout). `--sigreg` adds the play + SIGReg bar (dotted hatch) where it exists. A bar not yet scored is a dashed outline at its
mixture value marked "tbd". Data: data/finals_seeds.csv and data/cem_3env_floor.csv (mixture CEM, floors),
results/<run>-s<step>/O<k>.txt (play CEM), data/probe_heldout_unified_<env>.csv and data/probe_heldout_paper_calvin.csv (probes).

    python results/paper/scripts/plot_play_3env.py --metric cem   --out results/paper/figD_play_cem_3env.png
    python results/paper/scripts/plot_play_3env.py --metric probe --out results/paper/figD_play_probe_3env.png
"""
import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cells import cem_cell  # noqa: E402

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))
import style  # noqa: E402
RES = HERE.parent
ENVS = [("cube", "OGBench Cube", 25), ("scene", "OGBench Scene", 25), ("calvin", "CALVIN", 50)]
CELLS = (20000, 24000, 28000)
RUNS = {  # env -> (arm, variant) -> runs; mixture temporal on CALVIN is the far-frame form
    "cube": {("lewm", "mixture"): ["cube-lewm-mixlate-1", "cube-lewm-mixlate-2", "cube-lewm-mixlate-3"], ("lewm", "play"): ["cube-lewm-play-1"],
             ("temporal", "mixture"): ["cube-mixlate-lag-1", "cube-mixlate-lag-2", "cube-mixlate-lag-3"], ("temporal", "play"): ["cube-play-lag-1"]},
    "scene": {("lewm", "mixture"): ["scene-lewm-mixlate-1", "scene-lewm-mixlate-2", "scene-lewm-mixlate-3"], ("lewm", "play"): ["scene-lewm-play-1"],
              ("temporal", "mixture"): ["scene-mixlate-lag-1", "scene-mixlate-lag-2", "scene-mixlate-lag-3"], ("temporal", "play"): ["scene-play-lag-1"]},
    "calvin": {("lewm", "mixture"): ["calvin-lewm-mixlate-1", "calvin-lewm-mixlate-2", "calvin-lewm-mixlate-3"], ("lewm", "play"): ["calvin-lewm-play-1"],
               ("temporal", "mixture"): ["calvin-mixlate-lag-far-4", "calvin-mixlate-lag-far-5", "calvin-mixlate-lag-far-6"], ("temporal", "play"): ["calvin-play-lag-far-1"],
               ("temporal", "play + SIGReg"): ["calvin-play-lag-far-sigreg-1"]},
}
MIX_ARM = {"cube": {"lewm": "sigreg", "temporal": "temporal"}, "scene": {"lewm": "sigreg", "temporal": "temporal"}, "calvin": {"lewm": "sigreg", "temporal": "temporal"}}
GROUPS = [("lewm", "LeWM", "#7F7F7F"), ("temporal", "SpecWM (temporal)", "#1F3A63")]
HATCH = {"play": "////", "play + SIGReg": "...."}
RANDOM = "#D9D9D9"


def tint(hex_colour, f=0.55):
    r, g, b = (int(hex_colour[i:i + 2], 16) for i in (1, 3, 5))
    return "#%02X%02X%02X" % tuple(int(c + (255 - c) * f) for c in (r, g, b))


def cell(run, step, off):
    return cem_cell(run, f"s{step}", off)


def agg(vals):
    vals = [v for v in vals if np.isfinite(v)]
    if not vals: return np.nan, np.nan
    return float(np.mean(vals)), float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan


def cem_table():
    f = pd.read_csv(HERE / "data" / "finals_seeds.csv")
    floor = pd.read_csv(HERE / "data" / "cem_3env_floor.csv").set_index("env")["random_policy"]
    t = {}
    for env, _, off in ENVS:
        t[(env, "random")] = (float(floor[env]), np.nan)
        for arm, _, _ in GROUPS:
            for (a, variant), runs in RUNS[env].items():
                if a != arm: continue
                if variant == "mixture":
                    g = f[(f.env == env) & (f.arm == MIX_ARM[env][arm])]["mean_20k_30k"]
                    t[(env, arm, variant)] = agg(g.tolist())
                else:
                    t[(env, arm, variant)] = agg([np.mean([cell(r, s, off) for s in CELLS]) if all(np.isfinite(cell(r, s, off)) for s in CELLS) else np.nan for r in runs])
    return t


def probe_table():
    rows = []
    for p in sorted((HERE / "data").glob("probe_heldout_*.csv")):
        v = pd.read_csv(p)
        if "subset" in v: v = v[v.subset.fillna("all") == "all"]
        if "centred" in v: v = v[v.centred.astype(str) != "True"]
        if "head" in v: v = v[v["head"].fillna("linear") == "linear"]
        rows.append(v[(v.step == -1) | (v.run == "random-init")][["env", "run", "obj"]])
    v = pd.concat(rows, ignore_index=True).drop_duplicates(subset=["env", "run"], keep="last").set_index(["env", "run"])["obj"]
    t = {}
    for env, _, _ in ENVS:
        t[(env, "random")] = (float(v.get((env, "random-init"), np.nan)), np.nan)
        for (arm, variant), runs in RUNS[env].items():
            t[(env, arm, variant)] = agg([float(v.get((env, r), np.nan)) for r in runs])
    return t


def draw(ax, env, t, ylim, yticks, fmt, pad, sigreg):
    """One environment panel: Random, then LeWM and SpecWM (temporal) as mixture / play pairs."""
    W, G = 0.53, 0.5; x = 0.0
    val = t[(env, "random")][0]
    ax.bar(x, val, W - style.PAIR_GAP, color=RANDOM, zorder=3); ax.text(x, val + pad, fmt(val), ha="center", va="bottom", fontsize=style.VALUE_SMALL, color="#555")
    x += W + G
    for arm, _, colour in GROUPS:
        variants = ["mixture", "play"] + (["play + SIGReg"] if sigreg and (arm, "play + SIGReg") in RUNS[env] else [])
        ref = t[(env, arm, "mixture")][0]
        for k, variant in enumerate(variants):
            xc = x + k * W; val, sd = t.get((env, arm, variant), (np.nan, np.nan))
            if not np.isfinite(val):
                ax.bar(xc, ref, W - style.PAIR_GAP, facecolor="none", edgecolor=colour, linewidth=1.0, linestyle=(0, (3, 2)), zorder=3)
                ax.text(xc, ref + pad, "tbd", ha="center", va="bottom", fontsize=style.VALUE_SMALL, color="#9A9A9A"); continue
            h = HATCH.get(variant)
            ax.bar(xc, val, W - style.PAIR_GAP, color=colour, zorder=3, edgecolor=tint(colour) if h else "none", linewidth=0, hatch=h)
            top = val
            if np.isfinite(sd):
                ax.plot([xc, xc], [val - sd, val + sd], color="#333333", lw=1.0, zorder=5)
                for yy in (val - sd, val + sd): ax.plot([xc - 0.06, xc + 0.06], [yy, yy], color="#333333", lw=1.0, zorder=5)
                top = val + sd
            ax.text(xc, top + pad, fmt(val), ha="center", va="bottom", fontsize=style.VALUE_SMALL, color="#555")
        x += len(variants) * W + G
    ax.set_xticks([]); ax.set_xlim(-0.5, x - G - W + 0.5)
    ax.set_ylim(*ylim); ax.set_yticks(list(yticks)); ax.grid(axis="y", color="white", lw=1.2, zorder=0); ax.set_axisbelow(True)
    ax.tick_params(length=0, colors="#555"); ax.axhline(0, color="#C8C8CE", lw=0.8, zorder=2)
    for sp in ax.spines.values():
        sp.set_visible(False)


METRICS = {"cem": (cem_table, "Success (%)", (0, 100), range(0, 101, 20), (lambda v: f"{v:.0f}"), 1.2),
           "probe": (probe_table, "R²", (0, 1.06), [0, 0.2, 0.4, 0.6, 0.8, 1.0], (lambda v: f"{v:.2f}"), 0.013)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--metric", choices=["cem", "probe", "both"], required=True)
    ap.add_argument("--sigreg", action="store_true", help="add the play + SIGReg bar where a run exists")
    a = ap.parse_args()
    rows = ["cem", "probe"] if a.metric == "both" else [a.metric]
    style.apply()
    fig, axes = plt.subplots(len(rows), 3, figsize=(12.5, 3.9 if len(rows) == 1 else 8.6), sharey="row", squeeze=False)
    fig.subplots_adjust(left=0.08, right=0.995, top=0.88 if len(rows) == 1 else 0.94, bottom=0.17 if len(rows) == 1 else 0.11, wspace=0.07, hspace=0.14)
    for r, metric in enumerate(rows):
        table, ylabel, ylim, yticks, fmt, pad = METRICS[metric]; t = table()
        for c, (env, title, _) in enumerate(ENVS):
            draw(axes[r][c], env, t, ylim, yticks, fmt, pad, a.sigreg)
            if r == 0: axes[r][c].set_title(title, fontsize=style.TITLE, pad=10, color="#222")
        axes[r][0].set_ylabel("Object-position R²" if (metric == "probe" and len(rows) > 1) else ylabel, fontsize=style.LABEL, color="#333")
    hs = [mp.Patch(facecolor=RANDOM, edgecolor="none")] + [mp.Patch(facecolor=c, edgecolor="none") for _, _, c in GROUPS]
    ls = ["Random"] + [l for _, l, _ in GROUPS]
    hs.append(mp.Patch(facecolor="#6E6E74", edgecolor=tint("#6E6E74"), hatch="////", linewidth=0)); ls.append("play data only")
    if a.sigreg:
        hs.append(mp.Patch(facecolor="#6E6E74", edgecolor=tint("#6E6E74"), hatch="....", linewidth=0)); ls.append("play data only + SIGReg")
    fig.legend(hs, ls, loc="lower center", ncol=len(hs), frameon=False, fontsize=style.LEGEND, bbox_to_anchor=(0.5, 0.01), handlelength=1.4, columnspacing=2.2)
    style.save(fig, a.out)


if __name__ == "__main__":
    main()
