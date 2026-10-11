"""Attach a frozen teacher's tabulated latents to a scene corpus as a sidecar HDF5: every column of
the source copied, pixels as an external link, plus `teacher_z` (float16) from the .npy that
build_teacher_z.py wrote for the same file.

    python data/scene/make_qmz_sidecar.py --src $SCENE_STAGE_DIR/scene_play_explore_v4.h5 \
        --teacher-z $SCENE_STAGE_DIR/teacher_z_v4.npy --out $SCENE_STAGE_DIR/scene_play_explore_v4_qmz_fs5.h5
"""
import argparse
import os
import shutil

import h5py
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True); ap.add_argument("--teacher-z", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    Z = np.load(a.teacher_z, mmap_mode="r")
    with h5py.File(a.src, "r") as f, h5py.File(a.out, "w") as g:
        n = f["pixels"].shape[0]
        assert Z.shape[0] == n, f"teacher_z has {Z.shape[0]} rows, the corpus {n}"
        for key in f:
            if key == "pixels":
                g[key] = h5py.ExternalLink(os.path.basename(a.src), "/pixels")
            else:
                f.copy(key, g)
        d = g.create_dataset("teacher_z", shape=Z.shape, dtype="float16", chunks=(4096, Z.shape[1]))
        for c in range(0, n, 100000):
            d[c:c + 100000] = np.asarray(Z[c:c + 100000], dtype=np.float16)
    meta = os.path.splitext(a.teacher_z)[0] + ".meta.json"
    if os.path.exists(meta):
        shutil.copy(meta, os.path.splitext(a.out)[0] + ".meta.json")
    with h5py.File(a.out, "r") as g:
        assert g["pixels"].shape[0] == n and g["teacher_z"].shape[0] == n
    print(f"wrote {a.out}: {n} rows, teacher_z {Z.shape[1]}-d")


if __name__ == "__main__":
    main()
