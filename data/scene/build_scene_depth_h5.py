"""Assemble scene_play_depth.h5: the source Scene HDF5 plus a `depth_z` column, the gated and z-scored
front-camera depth of render_scene_depth.py (Fig. 5 depth runs).

Per-cell mean and standard deviation are computed over every frame in float64. A cell is kept when its
standard deviation is at least SIGMA_MIN metres and fewer than CLIP_MAX of its frames sit at the
renderer's depth clip; kept cells are z-scored and stored as float16. Every source column is copied
unchanged; `pixels` is a relative external link to the source file, so the output must sit beside it.

    python data/scene/build_scene_depth_h5.py --src $SCENE_STAGE_DIR/scene_play.h5 --shard-dir <shard-dir> \
        --dst $SCENE_STAGE_DIR/scene_play_depth.h5
"""
import argparse
import os
import sys

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render_scene_depth import D_CLIP, RES  # noqa: E402

SIGMA_MIN = 1e-3
CLIP_MAX = 0.01
PER_EP = ("ep_len", "ep_offset")


def _shards(shard_dir, ep_len):
    paths = []
    for ep, n in enumerate(ep_len):
        p = os.path.join(shard_dir, f"ep{ep:04d}.npy")
        a = np.load(p, mmap_mode="r")
        assert a.shape == (int(n), RES * RES) and a.dtype == np.float16, (p, a.shape, a.dtype)
        paths.append(p)
    return paths


def _load(p, device):
    return torch.from_numpy(np.load(p)).to(device).double()


def cell_stats(paths, device):
    """Two-pass float64 per-cell mean and std, and the fraction of frames at the depth clip."""
    C = RES * RES
    n, s, nclip = 0, torch.zeros(C, dtype=torch.float64, device=device), torch.zeros(C, dtype=torch.float64, device=device)
    for p in paths:
        x = _load(p, device)
        assert torch.isfinite(x).all(), p
        s += x.sum(0)
        nclip += (x >= D_CLIP).double().sum(0)
        n += x.shape[0]
    mean = s / n
    v = torch.zeros(C, dtype=torch.float64, device=device)
    for p in paths:
        d = _load(p, device) - mean
        v += (d * d).sum(0)
    return n, mean.cpu().numpy(), torch.sqrt(v / n).cpu().numpy(), (nclip / n).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--shard-dir", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    assert os.path.dirname(os.path.abspath(args.src)) == os.path.dirname(os.path.abspath(args.dst)), \
        "--dst must sit beside --src: pixels are a relative external link"
    device = torch.device(args.device)

    with h5py.File(args.src, "r", swmr=True) as s:
        ep_len = np.asarray(s["ep_len"])
    paths = _shards(args.shard_dir, ep_len)
    n, mean, std, clip_frac = cell_stats(paths, device)
    keep = (std >= SIGMA_MIN) & (clip_frac < CLIP_MAX)
    idx = np.nonzero(keep)[0]
    print(f"[depth] {n:,} frames, kept {len(idx)}/{RES * RES} cells", flush=True)

    idx_t = torch.from_numpy(idx).to(device)
    mu, sd = torch.from_numpy(mean[idx]).to(device), torch.from_numpy(std[idx]).to(device)
    tmp = args.dst + ".tmp"
    with h5py.File(args.src, "r", swmr=True) as s, h5py.File(tmp, "w", libver="latest") as d:
        for k, v in s.attrs.items():
            d.attrs[k] = v
        for k in s:
            if k == "pixels":
                continue
            if k in PER_EP:
                d.create_dataset(k, data=np.asarray(s[k]))
            else:
                s.copy(k, d)
        d["pixels"] = h5py.ExternalLink(os.path.basename(args.src), "/pixels")
        ds = d.create_dataset("depth_z", shape=(n, len(idx)), dtype=np.float16, chunks=(32, len(idx)))
        ptr = 0
        for i, p in enumerate(paths):
            z = ((_load(p, device)[:, idx_t] - mu) / sd).half()
            assert torch.isfinite(z).all(), p
            ds[ptr:ptr + z.shape[0]] = z.cpu().numpy()
            ptr += z.shape[0]
            if i % 100 == 0:
                print(f"  [depth_z] {i + 1}/{len(paths)} episodes", flush=True)
        assert ptr == n
        ds.attrs.update(cell_index=idx.astype(np.int32), raw_mean=mean[idx], raw_std=std[idx],
                        sigma_min=SIGMA_MIN, clip_max=CLIP_MAX, res=RES, camera="front_pixels")
    os.replace(tmp, args.dst)

    with h5py.File(args.dst, "r", swmr=True) as d:
        assert d["pixels"].shape[0] == int(np.asarray(d["ep_len"]).sum()) == d["depth_z"].shape[0]
    print(f"[depth] {args.dst}: depth_z {(n, len(idx))}", flush=True)


if __name__ == "__main__":
    main()
