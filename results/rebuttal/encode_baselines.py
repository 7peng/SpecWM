"""Reference 'embeddings' for the held-out frames, in the same layout as encode_heldout.py: raw-state (the
reachability graph's scaled full state, so its distance is the graph's own notion of closeness) and
raw-pixels (the frame average-pooled to 28x28x3). Episodes are read by parallel worker processes.
    python results/rebuttal/encode_baselines.py DATASET.h5 OUTDIR [--workers 8]"""
import argparse, sys
from pathlib import Path
import h5py, hdf5plugin  # noqa: F401
import numpy as np
import torch
import torch.nn.functional as Fn
from torch.utils.data import DataLoader
sys.path.insert(0, str(Path(__file__).resolve().parent))
import reach_graph as R  # noqa: E402
from encode_heldout import Episodes, val_episodes  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("dataset"); ap.add_argument("outdir"); ap.add_argument("--workers", type=int, default=8); a = ap.parse_args()
out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
with h5py.File(a.dataset, "r") as f:
    ep = f["ep_idx"][:].astype(np.int64); q = f["qpos"][:].astype(np.float64); b = f["button_states"][:]
val, _ = val_episodes(ep); isval = np.isin(ep, list(val)); rows = np.flatnonzero(isval)
assert np.array_equal(rows, np.load(out / "rows.npy")), "rows differ from the model embeddings"
same = ep[1:] == ep[:-1]
F = R.factors(q, b); tau = R.tolerances(F, same & isval[1:] & isval[:-1])
X = np.concatenate([F[k] / tau[k] for k in F], 1)[rows]
np.save(out / "raw-state.npy", X.astype(np.float32))
cut = np.r_[0, np.flatnonzero(np.diff(ep[rows]) != 0) + 1, len(rows)]
spans = [(int(rows[cut[i]]), int(rows[cut[i + 1] - 1]) + 1) for i in range(len(cut) - 1)]
P = np.zeros((len(rows), 28 * 28 * 3), np.float16); offs = np.r_[0, np.cumsum([e - s for s, e in spans])]
for k, px in DataLoader(Episodes(a.dataset, spans), batch_size=None, num_workers=a.workers):
    x = px.cuda().permute(0, 3, 1, 2).float() / 255.0
    P[offs[k]:offs[k] + len(x)] = Fn.adaptive_avg_pool2d(x, 28).flatten(1).cpu().numpy()
np.save(out / "raw-pixels.npy", P)
print(f"[baselines] raw-state {X.shape}, raw-pixels {P.shape} -> {out}")
