"""Rebuttal figure, bar form: rank correlation between embedding distance and three dataset distances on
held-out Scene frames (collection-policy time within an episode; reachability-graph distance within an
episode and between independent episodes), in three distance bands. Bars are the mean over three training
seeds with one standard deviation; raw pixels is a single reference. SpecWM in cosine, LeWM in L2, references in L2.
    python results/rebuttal/plot_reach_bars.py EMBDIR GRAPH.npz OUT.png"""
import sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mp
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "paper" / "scripts"))
from correlate import model_dist, spearman  # noqa: E402
import style as S  # noqa: E402

BANDS = [(0, 25), (25, 100), (100, 750)]
REPS = [("SpecWM (temporal)", "scene-mixlate-lag", "cos", S.TEMPORAL),
        ("LeWM", "scene-lewm-mixlate", "l2", S.LEWM), ("Raw pixels", "raw-pixels", "l2", S.RANDOM)]

emb, g, out = Path(sys.argv[1]), np.load(sys.argv[2]), sys.argv[3]
A, D, ep, t = g["anchors"], g["D"], g["ep"], g["t"]
same = ep[A][:, None] == ep[None, :]
fwd = same & (t[None, :] > t[A][:, None])
indep = ~same & ((ep[A] % 1000)[:, None] != (ep % 1000)[None, :])
dt = (t[None, :] - t[A][:, None]).astype(np.float64)
PANELS = [("Collection-policy time", fwd, dt), ("Reachability, same episode", fwd, D), ("Reachability, across episodes", indep, D)]
cache = Path(out).with_suffix(".npz")
if cache.exists():
    res = np.load(cache, allow_pickle=True)["res"].item()
else:
    rng = np.random.default_rng(0); idx = {}
    for title, m, T in PANELS:
        for b in BANDS:
            ai, ti = np.nonzero(m & np.isfinite(T) & (T >= b[0]) & (T < b[1]))
            k = rng.choice(len(ai), min(200000, len(ai)), replace=False)
            idx[title, b] = (ai[k], ti[k])
    res = {}
    for label, prefix, metric, _ in REPS:
        for nm in ([prefix] if prefix.startswith("raw") else [f"{prefix}-{s}" for s in (1, 2, 3)]):
            Dm = model_dist(np.load(emb / f"{nm}.npy"), A, metric)
            for title, _, T in PANELS:
                for b in BANDS:
                    ij = idx[title, b]; res.setdefault((label, title, b), []).append(spearman(Dm[ij], T[ij]))
        print(f"[bars] {label} done", flush=True)
    np.savez(cache, res=np.array(res, dtype=object))

S.apply()
fig, axes = plt.subplots(1, 3, figsize=(S.FULL, 5.6), sharey=True)
fig.subplots_adjust(left=0.07, right=0.995, top=0.90, bottom=0.21, wspace=0.08)
W, G = 0.26, 0.34; BW = W - S.PAIR_GAP
for ax, (title, _, _) in zip(axes, PANELS):
    x, ticks = 0.0, []
    for b in BANDS:
        for k, (label, prefix, _, c) in enumerate(REPS):
            r = np.array(res[label, title, b]); val = float(r.mean()); xc = x + k * W
            if c is None:
                ax.bar(xc, val, BW, facecolor="none", edgecolor="#444444", linewidth=1.1, linestyle=(0, (3, 2)), zorder=3)
            else:
                ax.bar(xc, val, BW, color=c, zorder=3)
            top = max(val, 0)
            if len(r) > 1:
                lo, hi = val - r.std(ddof=1), val + r.std(ddof=1)
                S.whisker(ax, xc, lo, hi, half=0.035); top = max(hi, 0)
            ax.text(xc, top + 0.012, f"{val:.2f}", ha="center", va="bottom", fontsize=S.VALUE_SMALL - 1.5, color="#555")
        ticks.append(x + W * (len(REPS) - 1) / 2); x += W * len(REPS) + G
    ax.set_xticks(ticks); ax.set_xticklabels([f"{lo}–{hi}" if lo else f"≤{hi}" for lo, hi in BANDS], fontsize=S.LABEL)
    ax.set_xlim(-0.5 * W - 0.12, x - G - 0.5 * W + 0.12)
    ax.set_ylim(-0.04, 0.95); ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
    ax.grid(axis="y", color="white", lw=1.2, zorder=0); ax.set_axisbelow(True); ax.tick_params(length=0, colors=S.TICK)
    ax.axhline(0, color=S.ZERO, lw=0.8, zorder=2)
    for sp in ax.spines.values():
        sp.set_visible(False)
    S.title(ax, title); ax.set_xlabel("Steps apart", fontsize=S.LABEL)
axes[0].set_ylabel("Spearman ρ", fontsize=S.LABEL, color="#333")
hs = [mp.Patch(facecolor="none", edgecolor="#444444", linewidth=1.1, linestyle=(0, (3, 2))) if c is None else mp.Patch(facecolor=c, edgecolor="none") for _, _, _, c in REPS]
S.legend(fig, hs, [lab for lab, *_ in REPS], ncol=5)
S.save(fig, out); fig.savefig(Path(out).with_suffix(".pdf"), facecolor="white")
for label, *_ in REPS:
    print(label, " | ".join(f"{p[0][:14]} " + " ".join(f"{np.mean(res[label, p[0], b]):+.2f}" for b in BANDS) for p in PANELS))
