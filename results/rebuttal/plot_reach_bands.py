"""Spearman correlation between embedding distance and a dataset distance on held-out Scene frames, per band
of that distance. --truth graph: reachability-graph distance, pairs within one episode vs from independent
episodes. --truth time: the recorded (collection-policy) gap, pairs within one episode. Mean over seeds with
the seed range shaded; each model uses its planning distance (cosine for SpecWM, L2 for LeWM), the raw-state
and raw-pixel references L2. Writes the per-band values with a grouped bootstrap interval to OUT.csv.
    python results/rebuttal/plot_reach_bands.py EMBDIR GRAPH.npz OUT.png [--truth graph|time] [--boot 100]"""
import argparse, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "paper" / "scripts"))
from correlate import model_dist, spearman  # noqa: E402
import style as S  # noqa: E402

BANDS = [(0, 25), (25, 50), (50, 100), (100, 150), (150, 300), (300, 750)]
FAMILIES = [("SpecWM (temporal)", "scene-mixlate-lag", "cos", S.TEMPORAL, "-"),
            ("SpecWM (quasimetric)", "scene-mixlate", "cos", S.QUASIMETRIC, "-"),
            ("LeWM", "scene-lewm-mixlate", "l2", S.LEWM, "-"),
            ("Untrained", "random-init", "cos", "#A6A6A6", ":"),
            ("Raw state", "raw-state", "l2", "#000000", "--"),
            ("Raw pixels", "raw-pixels", "l2", "#B07A2A", "-.")]
SINGLE = {"random-init", "raw-state", "raw-pixels"}

ap = argparse.ArgumentParser(); ap.add_argument("emb"); ap.add_argument("graph"); ap.add_argument("out")
ap.add_argument("--truth", choices=["graph", "time"], default="graph"); ap.add_argument("--boot", type=int, default=100); a = ap.parse_args()
emb, g = Path(a.emb), np.load(a.graph)
A, D, ep, t = g["anchors"], g["D"], g["ep"], g["t"]
same = ep[A][:, None] == ep[None, :]
fwd = same & (t[None, :] > t[A][:, None])
if a.truth == "graph":
    T = D; masks = {"Same episode": fwd, "Independent episodes": ~same & ((ep[A] % 1000)[:, None] != (ep % 1000)[None, :])}
    xlab = "graph distance between the frames (steps)"
else:
    T = (t[None, :] - t[A][:, None]).astype(np.float64); masks = {"Same episode, collection-policy time": fwd}
    xlab = "recorded gap between the frames (steps)"
rng = np.random.default_rng(0)
idx = {}
for pn, m in masks.items():
    for b in BANDS:
        ai, ti = np.nonzero(m & np.isfinite(T) & (T >= b[0]) & (T < b[1]))
        k = rng.choice(len(ai), min(200000, len(ai)), replace=False) if len(ai) >= 500 else None
        idx[pn, b] = (ai[k], ti[k]) if k is not None else None
res, rows = {}, []
for label, prefix, metric, _, _ in FAMILIES:
    names = [prefix] if prefix in SINGLE else [f"{prefix}-{s}" for s in (1, 2, 3)]
    for nm in names:
        Dm = model_dist(np.load(emb / f"{nm}.npy"), A, metric)
        for (pn, b), ij in idx.items():
            if ij is None:
                res.setdefault((label, pn, b), []).append(np.nan); continue
            dm, tr = Dm[ij], T[ij]; r = spearman(dm, tr)
            res.setdefault((label, pn, b), []).append(r)
            e = ep[A][ij[0]]; eu, inv = np.unique(e, return_inverse=True)
            by = np.split(np.argsort(inv, kind="stable"), np.cumsum(np.bincount(inv))[:-1])
            sub = rng.choice(len(dm), min(50000, len(dm)), replace=False)
            bs = []
            for _ in range(a.boot):
                w = np.concatenate([by[k] for k in rng.integers(0, len(eu), len(eu))])
                w = w[np.isin(w, sub)]
                bs.append(spearman(dm[w], tr[w]))
            rows.append(f"{label},{nm},{pn},{b[0]}-{b[1]},{r:.4f},{np.quantile(bs, .025):.4f},{np.quantile(bs, .975):.4f},{len(dm)}")
    print(f"[bands] {label} done", flush=True)
Path(a.out).with_suffix(".csv").write_text("family,model,panel,band,rho,ci_lo,ci_hi,n_pairs\n" + "\n".join(rows) + "\n")

S.apply()
fig, axes = plt.subplots(1, len(masks), figsize=(S.FULL if len(masks) > 1 else S.HALF * 1.3, S.ROW * 1.6), sharey=True, squeeze=False)
axes = axes[0]; x = np.arange(len(BANDS))
for ax, pn in zip(axes, masks):
    for label, prefix, _, c, ls in FAMILIES:
        r = np.array([res[label, pn, b] for b in BANDS])
        mu = np.nanmean(r, 1)
        if r.shape[1] > 1:
            ax.fill_between(x, np.nanmin(r, 1), np.nanmax(r, 1), color=c, alpha=0.18, lw=0, zorder=3)
        ax.plot(x, mu, color=c, ls=ls, lw=2.2, marker="o", ms=5, label=label, zorder=4)
    S.clean(ax, yticks=[-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1.0], ylim=(-0.24, 1.02))
    ax.set_xticks(x); ax.set_xticklabels([f"{lo}–{hi}" for lo, hi in BANDS], fontsize=S.VALUE + 1, color=S.TICK)
    ax.set_xlabel(xlab, fontsize=S.LABEL, color="#333")
    S.title(ax, pn)
S.ylabel(axes[0], "Spearman ρ")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=S.LEGEND, bbox_to_anchor=(0.5, -0.04))
fig.tight_layout(rect=(0, 0.13, 1, 1)); fig.savefig(a.out, dpi=200, bbox_inches="tight"); print(f"[out] {a.out}")
for label, *_ in FAMILIES:
    print(label, " | ".join(f"{pn[:5]} " + " ".join(f"{np.nanmean(res[label, pn, b]):+.2f}" for b in BANDS) for pn in masks))
