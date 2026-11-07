"""Spearman correlation between each model's embedding distance and dataset distances on held-out Scene
frames: (a) recorded time within an episode (the behaviour policy), (b) reachability-graph distance within
an episode and across independent episodes. Bootstrap over anchor episodes; mean and spread over seeds.
    python results/rebuttal/correlate.py EMBDIR GRAPH.npz [GRAPH2.npz ...] --out table.csv"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata


def spearman(x, y):
    return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])


def model_dist(Z, A, metric):
    Z = Z.astype(np.float32)
    if metric == "cos":
        Zn = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)
        return 1.0 - Zn[A] @ Zn.T
    sq = (Z ** 2).sum(1)
    return np.sqrt(np.maximum(sq[A][:, None] + sq[None, :] - 2 * Z[A] @ Z.T, 0))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("embdir"); ap.add_argument("graphs", nargs="+"); ap.add_argument("--out", required=True)
    ap.add_argument("--pairs", type=int, default=400000); ap.add_argument("--boot", type=int, default=200); a = ap.parse_args()
    rng = np.random.default_rng(0)
    emb = Path(a.embdir); rows = np.load(emb / "rows.npy")
    models = sorted(p.stem for p in emb.glob("*.npy") if p.stem != "rows")
    recs = []
    for gpath in a.graphs:
        g = np.load(gpath); A, D, ep, t, eps = g["anchors"], g["D"], g["ep"], g["t"], float(g["eps"])
        assert np.array_equal(g["rows"], rows), "graph targets and embedding rows differ"
        same = ep[A][:, None] == ep[None, :]
        indep = ~same & ((ep[A] % 1000)[:, None] != (ep % 1000)[None, :])
        fwd = same & (t[None, :] > t[A][:, None])
        dt = (t[None, :] - t[A][:, None]).astype(np.float64)
        sets = {"time (within episode)": (fwd, dt), "graph (within episode)": (fwd, D), "graph (independent episodes)": (indep & np.isfinite(D), D)}
        picks = {}
        for name, (mask, truth) in sets.items():
            ai, ti = np.nonzero(mask)
            k = rng.choice(len(ai), min(a.pairs, len(ai)), replace=False)
            picks[name] = (ai[k], ti[k], truth[ai[k], ti[k]])
        anchor_ep = ep[A]
        for m in models:
            Z = np.load(emb / f"{m}.npy")
            for metric in (("cos", "l2") if "lewm" in m else ("cos",)):
                Dm = model_dist(Z, A, metric)
                for name, (ai, ti, truth) in picks.items():
                    dm = Dm[ai, ti]; rho = spearman(dm, truth)
                    eps_u, inv = np.unique(anchor_ep[ai], return_inverse=True)
                    by_ep = np.split(np.argsort(inv, kind="stable"), np.cumsum(np.bincount(inv))[:-1]); bs = []
                    for _ in range(a.boot):
                        w = np.concatenate([by_ep[e] for e in rng.integers(0, len(eps_u), len(eps_u))])
                        bs.append(spearman(dm[w], truth[w]))
                    recs.append(dict(eps=eps, measure=name, model=m, metric=metric, rho=rho, ci_lo=np.quantile(bs, .025), ci_hi=np.quantile(bs, .975), n_pairs=len(ai)))
            print(f"[eps {eps:g}] {m} done", flush=True)
    d = pd.DataFrame(recs); d.to_csv(a.out, index=False)
    d["family"] = d.model.str.replace(r"-\d+$", "", regex=True)
    s = d.groupby(["eps", "measure", "family", "metric"]).rho.agg(["mean", "std", "count"]).round(3)
    print(s.to_string())


if __name__ == "__main__":
    main()
