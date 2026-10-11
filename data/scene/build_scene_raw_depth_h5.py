"""Assemble scene_play_mix_lateonset_depth_raw.h5: the source Scene HDF5 plus a `depth_m` column, the
full-resolution front-camera depth in metres (float16, unstandardised) from render_scene_depth.py.
Every source column is copied unchanged; `pixels` is an external link to the source file. A sample of
rows and every episode boundary is checked against the source, and a manifest is written beside the
output.

    python data/scene/build_scene_raw_depth_h5.py --src $SCENE_STAGE_DIR/scene_play_mix_lateonset.h5 \
        --shard-dir <shard-dir> --dst $SCENE_STAGE_DIR/scene_play_mix_lateonset_depth_raw.h5
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401
import numpy as np


PER_EP = {"ep_len", "ep_offset"}


def sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--shard-dir", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--seed", type=int, default=3072)
    ap.add_argument("--verify-rows", type=int, default=64)
    args = ap.parse_args()

    src, dst, shard_dir = Path(args.src), Path(args.dst), Path(args.shard_dir)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(src, "r", swmr=True) as s:
        ep_len = np.asarray(s["ep_len"], dtype=np.int64)
        ep_offset = np.asarray(s["ep_offset"], dtype=np.int64)
        n_rows = int(ep_len.sum())
        assert n_rows == int(s["pixels"].shape[0])
        assert np.array_equal(ep_offset, np.r_[0, np.cumsum(ep_len[:-1])])
        for ep, length in enumerate(ep_len):
            shard = np.load(shard_dir / f"ep{ep:04d}.npy", mmap_mode="r")
            assert shard.shape == (int(length), 224 * 224), (ep, shard.shape)
            assert shard.dtype == np.float16

        tmp = dst.with_suffix(dst.suffix + ".tmp")
        with h5py.File(tmp, "w", libver="latest") as d:
            for key, value in s.attrs.items():
                d.attrs[key] = value
            for key in s:
                if key == "pixels":
                    continue
                if key in PER_EP:
                    d.create_dataset(key, data=np.asarray(s[key]))
                else:
                    s.copy(key, d)
            d["pixels"] = h5py.ExternalLink(src.name, "/pixels")
            depth = d.create_dataset(
                "depth_m", shape=(n_rows, 224 * 224), dtype=np.float16,
                chunks=(8, 224 * 224),
            )
            observed_min, observed_max = np.inf, -np.inf
            for ep, (offset, length) in enumerate(zip(ep_offset, ep_len)):
                shard = np.load(shard_dir / f"ep{ep:04d}.npy", mmap_mode="r")
                depth[int(offset):int(offset + length)] = shard
                observed_min = min(observed_min, float(np.min(shard)))
                observed_max = max(observed_max, float(np.max(shard)))
                if ep % 25 == 0 or ep + 1 == len(ep_len):
                    print(f"[depth_m] {ep + 1}/{len(ep_len)} episodes", flush=True)
            depth.attrs.update(
                units="metres", standardization="none", variance_gate="none",
                camera="front_pixels", resolution=224, pool=1, n_cells=224 * 224,
                renderer_safety_clip_metres=2.0,
                observed_min_metres=observed_min, observed_max_metres=observed_max,
                source_h5=src.name, git_sha=git_sha(),
            )
        os.replace(tmp, dst)

    rng = np.random.default_rng(args.seed)
    rows = np.sort(rng.choice(n_rows, min(args.verify_rows, n_rows), replace=False))
    boundary = np.unique(np.r_[ep_offset, ep_offset + ep_len - 1])
    check_rows = np.unique(np.r_[rows, boundary])
    keys = ["qpos", "action", "pixels", "ep_idx"]
    hashes = {}
    with h5py.File(src, "r", swmr=True) as s, h5py.File(dst, "r", swmr=True) as d:
        assert d["depth_m"].shape == (n_rows, 224 * 224)
        assert np.isfinite(np.asarray(d["depth_m"][rows], np.float32)).all()
        for key in keys:
            left = np.stack([s[key][int(i)] for i in check_rows])
            right = np.stack([d[key][int(i)] for i in check_rows])
            assert np.array_equal(left, right), key
            hashes[key] = sha256_array(left)
        assert np.array_equal(s["ep_len"][:], d["ep_len"][:])
        assert np.array_equal(s["ep_offset"][:], d["ep_offset"][:])

    manifest = {
        "protocol": "scene-full-resolution-metric-depth-v1",
        "source": str(src), "destination": str(dst), "shard_dir": str(shard_dir),
        "rows": n_rows, "episodes": len(ep_len), "depth_shape": [n_rows, 224 * 224],
        "depth_dtype": "float16", "sample_seed": args.seed,
        "random_rows": rows.tolist(), "checked_rows_sha256": sha256_array(check_rows),
        "source_column_hashes_on_checked_rows": hashes,
        "git_sha": git_sha(),
    }
    dst.with_suffix(dst.suffix + ".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
