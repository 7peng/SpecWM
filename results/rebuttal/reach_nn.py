"""For every held-out frame of a Scene corpus, the L-inf distance (in typical one-step units, the
reachability graph's state scaling) to its nearest frame in an independent episode: any other episode
except its own late-onset twin (e and e +- 1000, which replays e's actions until the noise onset).
Prints the share of held-out frames with an independent neighbour within eps, for a range of eps.
    python results/rebuttal/reach_nn.py DATASET.h5 OUT.npz"""
import sys
from pathlib import Path
import h5py, hdf5plugin  # noqa: F401
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "paper" / "scripts"))
import reach_graph as R  # noqa: E402
from probe_val_episodes import val_episodes  # noqa: E402

f = h5py.File(sys.argv[1], "r")
ep = f["ep_idx"][:].astype(np.int64); val, _ = val_episodes(ep)
q = f["qpos"][:].astype(np.float64); b = f["button_states"][:]
same = ep[1:] == ep[:-1]; isval = np.isin(ep, list(val))
F = R.factors(q, b); tau = R.tolerances(F, same & isval[1:] & isval[:-1])
X = np.concatenate([F[k] / tau[k] for k in F], 1)
groups = b.astype(np.int64) @ (2 ** np.arange(b.shape[1])); family = ep % 1000
nn = np.full(len(ep), np.inf, np.float32)
for g in np.unique(groups):
    allg = np.flatnonzero(groups == g); qg = allg[isval[allg]]
    if not len(qg):
        continue
    Xa = torch.from_numpy(X[allg]).float().cuda(); fa = torch.from_numpy(family[allg]).cuda()
    for s in range(0, len(qg), 512):
        qi = qg[s:s + 512]
        D = torch.cdist(torch.from_numpy(X[qi]).float().cuda(), Xa, p=float("inf"))
        D[torch.from_numpy(family[qi]).cuda()[:, None] == fa[None, :]] = float("inf")
        nn[qi] = D.min(1).values.cpu().numpy()
    del Xa, D; torch.cuda.empty_cache()
v = nn[isval]
print(f"{isval.sum():,} held-out frames; nearest independent-episode neighbour, L-inf in typical steps: "
      f"p10 {np.quantile(v, .1):.1f} p25 {np.quantile(v, .25):.1f} p50 {np.median(v):.1f} p75 {np.quantile(v, .75):.1f} p90 {np.quantile(v, .9):.1f}")
for eps in (1, 2, 3, 4, 6, 8, 12, 16):
    print(f"eps {eps:2d}: {np.mean(v <= eps):6.1%} of held-out frames have an independent-episode neighbour")
np.savez_compressed(sys.argv[2], nn=v, tau=np.array(list(tau.values())))
