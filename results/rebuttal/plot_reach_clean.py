"""Rebuttal figure: rank correlation between embedding distance and three dataset distances on held-out
Scene frames, per distance band: (a) collection-policy time within an episode, (b) reachability-graph
distance within an episode, (c) graph distance between independent episodes. Mean over three seeds, range
shaded; SpecWM in cosine, LeWM in L2, the raw-state and raw-pixel references in L2.
    python results/rebuttal/plot_reach_clean.py EMBDIR GRAPH.npz OUT.png"""
import sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "paper" / "scripts"))
from correlate import model_dist, spearman  # noqa: E402
import style as S  # noqa: E402

BANDS = [(0, 25), (25, 50), (50, 100), (100, 150), (150, 300), (300, 750)]
DARK = "#444444"
REPS = [("SpecWM (temporal)", "scene-mixlate-lag", "cos", S.TEMPORAL, 2.4, "-"),
        ("SpecWM (quasimetric)", "scene-mixlate", "cos", S.QUASIMETRIC, 2.4, "-"),
        ("LeWM", "scene-lewm-mixlate", "l2", S.LEWM, 2.4, "-"),
        ("Raw state", "raw-state", "l2", DARK, 1.4, (0, (5, 2))),
        ("Raw pixels", "raw-pixels", "l2", DARK, 1.0, (0, (2, 2)))]


def clean(ax):
    ax.grid(color="white", lw=1.2, zorder=0); ax.set_axisbelow(True); ax.tick_params(length=0, colors=S.TICK)
    for sp in ax.spines.values():
        sp.set_visible(False)


emb, g, out = Path(sys.argv[1]), np.load(sys.argv[2]), sys.argv[3]
A, D, ep, t = g["anchors"], g["D"], g["ep"], g["t"]
same = ep[A][:, None] == ep[None, :]
fwd = same & (t[None, :] > t[A][:, None])
indep = ~same & ((ep[A] % 1000)[:, None] != (ep % 1000)[None, :])
dt = (t[None, :] - t[A][:, None]).astype(np.float64)
PANELS = [("Collection-policy time", fwd, dt, "Recorded gap (steps)"),
          ("Reachability, same episode", fwd, D, "Graph distance (steps)"),
          ("Reachability, across episodes", indep, D, "Graph distance (steps)")]
rng = np.random.default_rng(0)
idx = {}
for title, m, T, _ in PANELS:
    for b in BANDS:
        ai, ti = np.nonzero(m & np.isfinite(T) & (T >= b[0]) & (T < b[1]))
        k = rng.choice(len(ai), min(200000, len(ai)), replace=False) if len(ai) >= 500 else None
        idx[title, b] = (ai[k], ti[k]) if k is not None else None
cache = Path(out).with_suffix(".npz")
if cache.exists():
    z = np.load(cache, allow_pickle=True); res = z["res"].item()
else:
    res = {}
    for label, prefix, metric, *_ in REPS:
        names = [prefix] if prefix.startswith("raw") else [f"{prefix}-{s}" for s in (1, 2, 3)]
        for nm in names:
            Dm = model_dist(np.load(emb / f"{nm}.npy"), A, metric)
            for title, _, T, _ in PANELS:
                for b in BANDS:
                    ij = idx[title, b]
                    res.setdefault((label, title, b), []).append(np.nan if ij is None else spearman(Dm[ij], T[ij]))
        print(f"[clean] {label} done", flush=True)
    np.savez(cache, res=np.array(res, dtype=object))

S.apply()
fig, axes = plt.subplots(1, 3, figsize=(S.FULL, 4.3), sharey=True)
fig.subplots_adjust(left=0.095, right=0.985, top=0.89, bottom=0.30, wspace=0.12)
x = np.arange(len(BANDS))
for ax, (title, _, _, xlab) in zip(axes, PANELS):
    for label, prefix, _, c, lw, ls in REPS:
        r = np.array([res[label, title, b] for b in BANDS]); mu = np.nanmean(r, 1)
        if r.shape[1] > 1:
            ax.fill_between(x, np.nanmin(r, 1), np.nanmax(r, 1), color=c, alpha=0.2, lw=0, zorder=2)
        ax.plot(x, mu, color=c, lw=lw, ls=ls, label=label, zorder=3)
    ax.set_xlim(-0.3, len(BANDS) - 0.7); ax.set_ylim(-0.05, 0.9); ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
    ax.set_xticks(x); ax.set_xticklabels([f"≤{hi}" for _, hi in BANDS], fontsize=S.VALUE + 1)
    clean(ax); S.title(ax, title); ax.set_xlabel(xlab, fontsize=S.LABEL)
S.ylabel(axes[0], "Spearman ρ with\nembedding distance")
h, l = axes[0].get_legend_handles_labels()
S.legend(fig, h, l, ncol=5, handlelength=2.0)
S.save(fig, out); fig.savefig(Path(out).with_suffix(".pdf"), facecolor="white")
for label, *_ in REPS:
    print(label, " | ".join(f"{p[0][:14]} " + " ".join(f"{np.nanmean(res[label, p[0], b]):+.2f}" for b in BANDS) for p in PANELS))
