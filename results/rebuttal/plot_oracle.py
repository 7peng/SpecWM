"""Expert hitting time (OGBench Markov oracle, noise-free) against reachability-graph distance for held-out
Scene pairs that differ in one object factor, and each representation's rank correlation with the expert
time per factor.
    python results/rebuttal/plot_oracle.py ORACLE.csv OUT.png"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "paper" / "scripts"))
from correlate import spearman  # noqa: E402
import style as S  # noqa: E402

d = pd.read_csv(sys.argv[1]); ok = d[d.oracle > 0]
FACT = ["cube", "button", "drawer", "window"]
REPS = [("SpecWM (temporal)", "scene-mixlate-lag-1", S.TEMPORAL), ("SpecWM (quasimetric)", "scene-mixlate-1", S.QUASIMETRIC),
        ("LeWM", "scene-lewm-mixlate-1", S.LEWM), ("Raw state", "raw-state", "#000000"), ("Raw pixels", "raw-pixels", "#B07A2A")]
S.apply()
fig, axes = plt.subplots(1, 2, figsize=(S.FULL, S.ROW * 1.6), gridspec_kw=dict(width_ratios=[1, 1.25]))
ax = axes[0]
for f, m in zip(FACT, ["o", "s", "^", "D"]):
    g = ok[ok.factor == f]
    ax.scatter(g.oracle, g.graph, s=22, marker=m, alpha=0.75, label=f"{f}  (ρ {spearman(g.oracle, g.graph):+.2f}, n {len(g)})", zorder=3)
lim = [1, 2000]; ax.plot(lim, lim, color=S.ZERO, lw=1, ls="--", zorder=2); ax.text(1.3, 2.2, "graph = expert", fontsize=S.VALUE, color=S.TICK, rotation=33)
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(1, 200); ax.set_ylim(1, 2000)
S.clean(ax, zero=False); ax.grid(axis="x", color="white", lw=1.2, zorder=0)
ax.set_xlabel("expert hitting time (steps)", fontsize=S.LABEL, color="#333"); S.ylabel(ax, "graph distance (steps)")
S.title(ax, "Graph distance vs expert"); ax.legend(frameon=False, fontsize=S.VALUE + 1, loc="lower right")
ax = axes[1]; x = np.arange(len(FACT)); w = 0.16
for k, (lab, col, c) in enumerate(REPS):
    v = [spearman(ok[ok.factor == f][col], ok[ok.factor == f].oracle) for f in FACT]
    ax.bar(x + (k - 2) * w, v, w * 0.92, color=c, label=lab, zorder=3)
S.clean(ax, yticks=[-0.2, 0, 0.2, 0.4], ylim=(-0.3, 0.5)); ax.set_xticks(x); ax.set_xticklabels(FACT, fontsize=S.LABEL)
S.ylabel(ax, "Spearman ρ with expert time"); S.title(ax, "Representation distance vs expert (seed 1)")
ax.text(0.99, 0.02, "±0.12 sampling error at n ≈ 70", transform=ax.transAxes, ha="right", fontsize=S.VALUE, color=S.TICK)
h, l = ax.get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=5, frameon=False, fontsize=S.LEGEND - 1, bbox_to_anchor=(0.5, -0.03))
fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig(sys.argv[2], dpi=200, bbox_inches="tight"); print(f"[out] {sys.argv[2]}")
