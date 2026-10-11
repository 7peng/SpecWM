"""Adversarial audit of the HDF5 clip table (utils.RowReadHDF5Dataset) and its consumers.

Builds a tiny synthetic HDF5 with LABELLED columns (every value encodes its own global row
index) so any reordering, cross-episode read or off-by-one is visible in the output.

    .venv/bin/python tests/adv_scaffold_cliptable.py        (from the repo root)
"""
import os
import sys
import tempfile
import traceback
from pathlib import Path

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

FAIL = []


def check(name, cond, msg=""):
    print(("  PASS " if cond else "  FAIL ") + name + (f"  -- {msg}" if msg else ""))
    if not cond:
        FAIL.append(name)


LENGTHS = [20, 12, 6]
ADIM = 2


def build(path):
    off, rows = [], 0
    for L in LENGTHS:
        off.append(rows)
        rows += L
    with h5py.File(path, "w") as f:
        f.create_dataset("ep_len", data=np.array(LENGTHS, dtype=np.int32))
        f.create_dataset("ep_offset", data=np.array(off, dtype=np.int64))
        # global row index, per column, so a wrong read is legible
        g = np.arange(rows, dtype=np.float32)
        f.create_dataset("step_idx", data=g.reshape(-1, 1))
        ep = np.concatenate([np.full(L, i, np.float32) for i, L in enumerate(LENGTHS)])
        f.create_dataset("ep_idx", data=ep.reshape(-1, 1))
        # action[t] = [global_row, local_step]; NEVER 0 for a real row (offset by 1) so the
        # zero pad is distinguishable from a genuine action
        loc = np.concatenate([np.arange(L, dtype=np.float32) for L in LENGTHS])
        f.create_dataset("action", data=np.stack([g + 1.0, loc + 1.0], 1).astype(np.float32))
        # a 1-frame-chunked "pixel" column to exercise the row-read path
        px = np.zeros((rows, 2, 2, 3), np.uint8)
        px[:, 0, 0, 0] = (g % 251).astype(np.uint8)
        f.create_dataset("pixels", data=px, chunks=(1, 2, 2, 3))
    return off


def main():
    from utils import RowReadHDF5Dataset

    tmp = Path(tempfile.mkdtemp(prefix="advclip_"))
    h5 = tmp / "tiny.h5"
    off = build(h5)

    NS, FS = 4, 3
    span, reach = NS * FS, (NS - 1) * FS + 1
    ds = RowReadHDF5Dataset(path=str(h5), frameskip=FS, num_steps=NS,
                            keys_to_load=["step_idx", "ep_idx", "action", "pixels"])

    # ---- 1. the table itself -------------------------------------------------
    want = [(0, s) for s in range(LENGTHS[0] - reach + 1)] + \
           [(1, s) for s in range(LENGTHS[1] - reach + 1)]
    check("clip table = every start whose last KEPT frame is inside the episode",
          list(ds.clip_indices) == want,
          f"{len(ds.clip_indices)} clips, want {len(want)}")
    check("episode shorter than reach is dropped",
          all(ep != 2 for ep, _ in ds.clip_indices))
    check("last frame (num_steps-1)*fs is reachable",
          max(s for e, s in ds.clip_indices if e == 0) + (NS - 1) * FS == LENGTHS[0] - 1)

    # ---- 2. every clip: one episode, right frames, right actions -------------
    bad_ep, bad_frames, bad_act, pad_rows = 0, 0, 0, 0
    for i, (ep, st) in enumerate(ds.clip_indices):
        it = ds[i]
        si = it["step_idx"].numpy().reshape(-1)
        ei = it["ep_idx"].numpy().reshape(-1)
        act = it["action"].numpy().reshape(NS, FS, ADIM)
        px = it["pixels"].numpy()
        if not np.all(ei == ep):
            bad_ep += 1
        want_frames = off[ep] + st + np.arange(NS) * FS
        if not np.array_equal(si, want_frames.astype(np.float32)):
            bad_frames += 1
        # pixels were permuted to (T,C,H,W): channel 0, row 0, col 0 holds row%251
        if not np.array_equal(px[:, 0, 0, 0], (want_frames % 251).astype(np.uint8)):
            bad_frames += 1
        for t in range(NS):
            for k in range(FS):
                grow = off[ep] + st + t * FS + k
                inside = (st + t * FS + k) < LENGTHS[ep]
                if inside:
                    if act[t, k, 0] != grow + 1.0:
                        bad_act += 1
                else:
                    pad_rows += 1
                    if act[t, k, 0] != 0.0:
                        bad_act += 1
    check("frames of a clip all come from ONE episode", bad_ep == 0, f"{bad_ep} clips mixed")
    check("frames are start + t*frameskip (and pixels agree)", bad_frames == 0,
          f"{bad_frames} clips wrong")
    check("action block for frame t is actions t*fs..(t+1)*fs-1, pad = raw 0",
          bad_act == 0, f"{bad_act} wrong entries, {pad_rows} padded rows")
    check("padding exists and is bounded", 0 < pad_rows <= len(ds.clip_indices) * (FS - 1),
          f"{pad_rows} padded action rows over {len(ds.clip_indices)} clips")

    # ---- 3. the pad is inserted BEFORE the transform (raw units) -------------
    from utils import ZScoreNormalizer
    from stable_pretraining import data as dt
    # hdf5 backend: train_common passes tile=1, so the stats are per RAW action column and the
    # transform runs on the (span, adim) array BEFORE Dataset.__getitem__ reshapes it
    mean = torch.zeros(1, ADIM) + 5.0
    std = torch.ones(1, ADIM) * 2.0
    ds.transform = dt.transforms.WrapTorchTransform(
        ZScoreNormalizer(mean, std), source="action", target="action")
    # last clip of ep0 has the most padding
    li = len(ds.clip_indices) - 1
    it = ds[li]
    a = it["action"].numpy().reshape(NS, FS, ADIM)
    ep, st = ds.clip_indices[li]
    tail = [(t, k) for t in range(NS) for k in range(FS) if st + t * FS + k >= LENGTHS[ep]]
    zs = [a[t, k, 0] for t, k in tail]
    check("zero pad is z-SCORED afterwards (raw 0 -> (0-mean)/std), not a post-hoc 0",
          all(abs(v - (0.0 - 5.0) / 2.0) < 1e-5 for v in zs),
          f"padded z-scores {zs} (raw 0 would be -2.5; a post-normalisation pad would be 0.0)")
    ds.transform = None

    # ---- 4. keys_to_cache path must agree with the file path ----------------
    ds_c = RowReadHDF5Dataset(path=str(h5), frameskip=FS, num_steps=NS,
                              keys_to_load=["step_idx", "action"],
                              keys_to_cache=["step_idx", "action"])
    same = all(torch.equal(ds[i]["action"], ds_c[i]["action"])
               and torch.equal(ds[i]["step_idx"], ds_c[i]["step_idx"])
               for i in range(len(ds_c)))
    check("keys_to_cache path returns the same rows as the file path", same)

    # ---- 5. DataLoader worker pickling round trip ---------------------------
    import pickle
    from utils import PickleableHDF5Wrapper
    sub = torch.utils.data.Subset(ds, list(range(len(ds))))
    w = PickleableHDF5Wrapper(sub)
    w2 = pickle.loads(pickle.dumps(w))
    check("PickleableHDF5Wrapper survives pickling and reads the same rows",
          torch.equal(w[0]["step_idx"], w2[0]["step_idx"]) and len(w2) == len(w))

    # ---- 6. train_common's episode split reads THIS table -------------------
    ci = np.asarray(ds.clip_indices, dtype=np.int64)
    check("the hdf5 clip table is an (n_clips, 2) episode/start table",
          ci.ndim == 2 and ci.shape[1] == 2, f"{ci.shape}")
    check("val_split_by_episode recovers the right episode ids from it (width-agnostic)",
          np.array_equal(np.unique(ci[:, 0]), np.array([0, 1])),
          f"{np.unique(ci[:, 0]).tolist()}")

    # ---- 7. frameskip 1 (no padding at all) ---------------------------------
    ds1 = RowReadHDF5Dataset(path=str(h5), frameskip=1, num_steps=4,
                             keys_to_load=["step_idx", "action"])
    n1 = sum(max(0, L - 4 + 1) for L in LENGTHS)
    check("frameskip 1: reach == num_steps, no episode lost, no padding",
          len(ds1) == n1 and all(ds1[i]["action"].numpy().min() > 0 for i in range(len(ds1))),
          f"{len(ds1)} clips vs {n1}")

    print("\n" + ("ALL PASS" if not FAIL else f"FAILURES: {FAIL}"))
    return 1 if FAIL else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(2)
