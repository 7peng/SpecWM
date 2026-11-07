"""Dataset reachability over a Scene corpus: a graph over every frame with one directed edge per recorded
step (cost 1) and an undirected edge between any two frames whose full state is within eps typical steps
of each other (L-inf over factors, each factor in units of its median one-step change; cost = the
distance rounded up). Weighted shortest paths from held-out anchors to held-out targets, one graph per eps
from a single neighbour pass. Prints the validation checks, saves the anchor distances, draws an audit of
short cross-episode paths.
    python results/rebuttal/reach_graph.py DATASET.h5 OUTDIR [--eps 2 3] [--anchors 300] [--workers 8]"""
import argparse, multiprocessing as mp, sys, time
from pathlib import Path
import h5py, hdf5plugin  # noqa: F401
import numpy as np
import torch
from scipy import sparse
from scipy.sparse.csgraph import shortest_path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "paper" / "scripts"))
from probe_val_episodes import val_episodes  # noqa: E402

FLOOR = 1e-3


def factors(q, b):
    """Per-factor state rows: arm joints as cos/sin, cube quaternion with its sign fixed."""
    arm = q[:, 0:6]
    quat = q[:, 17:21] * np.sign(q[:, 17:18] + 1e-12)
    return {"arm": np.concatenate([np.cos(arm), np.sin(arm)], 1), "gripper": q[:, 6:14], "cube_pos": q[:, 14:17],
            "cube_quat": quat, "drawer": q[:, 23:24], "window": q[:, 24:25], "buttons": b.astype(np.float64)}


def tolerances(F, same):
    """Median per-step L-inf change of each factor over steps that move it by more than FLOOR."""
    tau = {}
    for k, v in F.items():
        if k == "buttons":
            tau[k] = 1e-3
            continue
        d = np.abs(np.diff(v, axis=0))[same].max(1)
        tau[k] = float(np.median(d[d > FLOOR]))
    return tau


def proximity_edges(X, groups, eps, device, chunk=512):
    """All pairs (i, j), i < j, within one group, with L-inf distance <= eps; returns i, j and the distance."""
    I, J, Dv = [], [], []
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g); Xt = torch.from_numpy(X[idx]).float().to(device); it = torch.from_numpy(idx).to(device)
        for s in range(0, len(Xt), chunk):
            D = torch.cdist(Xt[s:s + chunk], Xt, p=float("inf"))
            i, j = torch.nonzero(D <= eps, as_tuple=True)
            keep = j > i + s
            i, j = i[keep], j[keep]
            I.append(it[i + s].cpu().numpy()); J.append(it[j].cpu().numpy()); Dv.append(D[i, j].half().cpu().numpy())
        del Xt; torch.cuda.empty_cache()
    return np.concatenate(I), np.concatenate(J), np.concatenate(Dv).astype(np.float32)


_G = _V = None


def _sp(src):
    """Weighted shortest paths from the given anchors, kept on the target columns only."""
    return shortest_path(_G, method="D", directed=True, indices=src)[:, _V].astype(np.float32)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("dataset"); ap.add_argument("outdir")
    ap.add_argument("--eps", type=float, nargs="+", default=[2.0, 3.0]); ap.add_argument("--anchors", type=int, default=300)
    ap.add_argument("--workers", type=int, default=8); ap.add_argument("--seed", type=int, default=0); a = ap.parse_args()
    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True); t0 = time.time()
    f = h5py.File(a.dataset, "r")
    e = f["ep_idx"][:].astype(np.int64); val, n_ep = val_episodes(e)
    q = f["qpos"][:].astype(np.float64); b = f["button_states"][:]
    n = len(e); same = e[1:] == e[:-1]
    starts = np.r_[0, np.flatnonzero(~same) + 1]; t = np.arange(n) - np.repeat(starts, np.diff(np.r_[starts, n]))
    isval = np.isin(e, list(val)); V = np.flatnonzero(isval); fam = e % 1000
    F = factors(q, b); tau = tolerances(F, same & isval[1:] & isval[:-1])
    X = np.concatenate([F[k] / tau[k] for k in F], 1)
    groups = b.astype(np.int64) @ (2 ** np.arange(b.shape[1]))
    print(f"[graph] {n:,} frames, {n_ep} episodes; targets {len(V):,} frames of {len(val)} validation episodes; tolerances "
          + ", ".join(f"{k} {v:.2e}" for k, v in tau.items()), flush=True)
    cache = out / f"edges_eps{max(a.eps):g}.npz"
    if cache.exists():
        z = np.load(cache); pi, pj, pd = z["i"], z["j"], z["d"]
    else:
        pi, pj, pd = proximity_edges(X, groups, max(a.eps), "cuda")
        np.savez(cache, i=pi, j=pj, d=pd)
    print(f"[edges] {len(pi):,} pairs within eps {max(a.eps):g} in {time.time() - t0:.0f}s", flush=True)

    rng = np.random.default_rng(a.seed)
    anchors = np.sort(rng.choice(V, a.anchors, replace=False)); A = np.searchsorted(V, anchors)
    ev, tv, fv = e[V], t[V], fam[V]
    same_ep = ev[A][:, None] == ev[None, :]
    twin = (fv[A][:, None] == fv[None, :]) & ~same_ep
    indep = ~same_ep & ~twin
    fwd = same_ep & (tv[None, :] > tv[A][:, None])
    dt = (tv[None, :] - tv[A][:, None]).astype(np.float64)
    si = np.flatnonzero(same)
    global _G, _V
    for eps in sorted(a.eps):
        k = pd <= eps; ci, cj, cw = pi[k], pj[k], np.ceil(np.maximum(pd[k], 1e-6))
        cross = e[ci] != e[cj]; tw = cross & (fam[ci] == fam[cj])
        hasx = np.zeros(n, bool); hasx[ci[cross & ~tw]] = True; hasx[cj[cross & ~tw]] = True
        src = np.r_[si, ci, cj]; dst = np.r_[si + 1, cj, ci]; w = np.r_[np.ones(len(si)), cw, cw].astype(np.float32)
        key = src.astype(np.int64) * n + dst; o_ = np.lexsort((w, key)); key, w = key[o_], w[o_]
        first = np.r_[True, key[1:] != key[:-1]]          # one edge per (src, dst), at its smallest cost
        _G, _V = sparse.csr_matrix((w[first], (key[first] // n, key[first] % n)), shape=(n, n)), V
        t1 = time.time()
        with mp.get_context("fork").Pool(a.workers) as pool:
            D = np.concatenate(pool.map(_sp, np.array_split(anchors, a.workers * 4)))
        fin = np.isfinite(D)
        print(f"\n[eps {eps:g}] {k.sum():,} proximity edges ({cross.mean():.1%} cross-episode, {tw.sum() / max(cross.sum(), 1):.1%} of those twin); "
              f"held-out frames with an independent-episode edge {hasx[V].mean():.1%}; shortest paths in {time.time() - t1:.0f}s", flush=True)
        print(f"[eps {eps:g}] reachable: same episode forward {fin[fwd].mean():.1%}, backward {fin[same_ep & ~fwd].mean():.1%}, "
              f"twin {fin[twin].mean():.1%}, independent episodes {fin[indep].mean():.1%}")
        print(f"[eps {eps:g}] bound: forward pairs with graph distance > recorded gap: {(D[fwd] > dt[fwd]).sum()} of {fwd.sum():,} (must be 0)")
        r = D[fwd] / dt[fwd]
        for lo, hi in [(1, 10), (10, 50), (50, 200), (200, 1000)]:
            m = (dt[fwd] >= lo) & (dt[fwd] < hi)
            print(f"[eps {eps:g}] gap {lo:4d}-{hi:4d}: graph/recorded p10 {np.quantile(r[m], .1):.2f} p50 {np.median(r[m]):.2f} p90 {np.quantile(r[m], .9):.2f}")
        di = D[indep & fin]
        if len(di):
            print(f"[eps {eps:g}] independent-episode distances: p10 {np.quantile(di, .1):.0f} p50 {np.median(di):.0f} p90 {np.quantile(di, .9):.0f}")
        np.savez_compressed(out / f"scene_eps{eps:g}.npz", anchors=A, rows=V, ep=ev, t=tv, D=D, tau=np.array(list(tau.values())), eps=eps)

        from PIL import Image, ImageDraw
        ii, jj = np.nonzero(indep & fin & (D <= 6))
        pick = rng.choice(len(ii), min(8, len(ii)), replace=False) if len(ii) else []
        S = 160; im = Image.new("RGB", (2 * S, S * max(1, len(pick))), "white"); dr = ImageDraw.Draw(im)
        for r_, kk in enumerate(pick):
            for c_, idx in enumerate((V[A[ii[kk]]], V[jj[kk]])):
                im.paste(Image.fromarray(f["pixels"][idx]).resize((S, S)), (c_ * S, r_ * S))
            dr.text((4, r_ * S + 2), f"eps {eps:g} graph {D[ii[kk], jj[kk]]:.0f}", fill=(255, 255, 0))
        im.save(out / f"audit_eps{eps:g}.png")
        print(f"[eps {eps:g}] audit: {len(ii):,} independent-episode pairs at graph distance <= 6; {len(pick)} drawn")


if __name__ == "__main__":
    main()
