"""Per-step change of each Scene state factor on the held-out validation episodes: the scale for the
reachability graph's same-state tolerance."""
import sys
from pathlib import Path
import h5py, hdf5plugin  # noqa: F401
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "paper" / "scripts"))
from probe_val_episodes import val_episodes  # noqa: E402

FACTORS = {"arm": slice(0, 6), "gripper": slice(6, 14), "cube_pos": slice(14, 17), "cube_quat": slice(17, 21), "drawer": slice(23, 24), "window": slice(24, 25)}

f = h5py.File(sys.argv[1], "r")
ep = f["ep_idx"][:].astype(np.int64); val, n_ep = val_episodes(ep)
rows = np.flatnonzero(np.isin(ep, list(val)))
q = f["qpos"][rows[0]:rows[-1] + 1][rows - rows[0]].astype(np.float64); b = f["button_states"][rows[0]:rows[-1] + 1][rows - rows[0]]
e = ep[rows]; same = e[1:] == e[:-1]
print(f"{len(val)} validation episodes of {n_ep}, {len(rows)} frames, qpos {q.shape}, buttons {b.shape}")
for k, s in FACTORS.items():
    d = np.abs(np.diff(q[:, s], axis=0))[same].max(1)
    mv = d[d > 1e-6]
    print(f"{k:10s} per-step max|delta|: moving {len(mv) / len(d):5.1%} of steps; quantiles of moving steps "
          + " ".join(f"p{p}={np.quantile(mv, p / 100):.2e}" for p in (10, 25, 50, 75, 90)) + f" | range {np.ptp(q[:, s]):.3f}")
print("button changes per step:", float((np.diff(b, axis=0)[same] != 0).any(1).mean()))
