"""Freeze a probe suite into one pack the training callback can probe in seconds.

The callback re-encodes every frame each time it runs, because the encoder is what changed; only
the pixels and labels are fixed. So the pack stores frames already decoded and resized to the
encoder's input size, and gives every cell the same row count so the whole probe is one batched
solve instead of a Python loop over cells. Decoding here rather than at probe time also keeps
the measurement deterministic.

A row is one probe sample: one frame from the suite's primary camera, labelled with the object
positions recorded for that frame. A suite is `<suite>/<split>/<cell>/frames/*.{png,jpg}` plus
`labels.json` per cell, as `data/ogbench/make_ogbench_suite.py` and
`data/calvin/make_calvin_suite.py` write it. Object cells share one row set per suite cell:
a frame is usable only if every object it labels is visible (`px >= MIN_PX`). Output:

    frames.npy   (n, 224, 224, 3) uint8
    cells.npz    `<group>.rows` (cells, rows) int32 row indices into frames.npy,
                 `<group>.Y` (cells, rows, dim) float32 labels
    index.json   row counts, per-cell names and ID/OOD split; rows[:n_train] of every cell are
                 the fitting rows, the rest held out

  python data/probe_suite/build_probe_pack.py --suite <suite> --out <pack> --rows 2667
"""
import os
import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

RES = 224

SUITE = Path(os.environ.get("STABLEWM_HOME", "~/.stable-wm")).expanduser() / "probe_suite"
MIN_PX = 10
TEST_FRAC = 0.25
GROUPS = {"obj_in_cam": 3, "eef_in_cam": 3, "arm_qpos": 7}
LABEL = "in_cam"      # which stored object position fills the obj group: camera-frame or world xyz


def cell_rows(kd, rows, rng):
    """-> (row_frame_names, {group: {cell name: (rows, dim) array}}) or None if the cell has
    fewer than `rows` usable frames.

    Object cells keep a frame only if every labelled object has `px >= MIN_PX`; arm cells
    (records carrying `eef_in_cam`) keep every frame. `rows` frames are drawn without
    replacement and permuted, so the train/test split is positional.
    """
    D = json.loads((kd / "labels.json").read_text())
    assert int(D.get("meta", {}).get("window", 1) or 1) == 1, f"{kd}: one frame per row expected"
    frames = D["frames"]
    if not frames:
        return None
    arm = "eef_in_cam" in frames[0]
    if arm:
        keep = np.arange(len(frames))
    else:
        names = list(frames[0]["objects"])
        keep = np.flatnonzero([min(f["objects"][n]["px"] for n in names) >= MIN_PX
                               for f in frames])
    if len(keep) < rows:
        return None
    sel = keep[rng.choice(len(keep), rows, replace=False)]
    sel = sel[rng.permutation(rows)]                       # split is rows[:n_tr] vs rows[n_tr:]
    ext = D.get("meta", {}).get("ext", "jpg")
    names_out = [f"{frames[i]['frame']:06d}.{ext}" for i in sel]
    last = [frames[i] for i in sel]
    cells = {}
    if arm:
        cells["eef_in_cam"] = {"arm": np.array([f["eef_in_cam"] for f in last], np.float32)}
        cells["arm_qpos"] = {"arm": np.array([f["arm_qpos"] for f in last], np.float32)}
    else:
        cells["obj_in_cam"] = {
            f"{n}:{frames[0]['objects'][n]['category']}":
                np.array([f["objects"][n][LABEL] for f in last], np.float32)
            for n in frames[0]["objects"]}
    return names_out, cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=400, help="frames per object cell (train+test)")
    ap.add_argument("--test-frac", type=float, default=TEST_FRAC,
                    help="held-out share of each cell's rows")
    ap.add_argument("--arm-rows", type=int, default=0,
                    help="frames per arm cell; 0 -> same as --rows")
    ap.add_argument("--suite", default=str(SUITE))
    ap.add_argument("--out", default=str(SUITE / "probe_pack"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--label", default="in_cam", choices=("in_cam", "world"),
                    help="object position label: camera-frame xyz or world xyz")
    args = ap.parse_args()
    global LABEL
    LABEL = args.label
    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    todo, meta = [], []
    cells = {g: {"rows": [], "Y": [], "name": [], "split": []} for g in GROUPS}
    for split in ("id", "ood", "id_arm", "ood_arm"):
        sd = Path(args.suite) / split
        if not sd.is_dir():
            continue
        for kd in sorted(sd.iterdir()):
            if not (kd / "labels.json").exists():
                continue
            rows = (args.arm_rows or args.rows) if split.endswith("_arm") else args.rows
            got = cell_rows(kd, rows, rng)
            if got is None:
                continue
            names, kcells = got
            base = len(todo)
            todo += [kd / "frames" / nm for nm in names]
            idx = np.arange(base, base + len(names), dtype=np.int32)
            dom = "OOD" if split.startswith("ood") else "ID"
            for g, per in kcells.items():
                for cname, Y in per.items():
                    cells[g]["rows"].append(idx)
                    cells[g]["Y"].append(Y)
                    cells[g]["name"].append(f"{split}/{kd.name}/{cname}")
                    cells[g]["split"].append(dom)
            meta.append(dict(split=split, cell=kd.name, n=len(names)))

    from PIL import Image
    shape = (len(todo), RES, RES, 3)
    mm = np.lib.format.open_memmap(out / "frames.npy", mode="w+", dtype=np.uint8, shape=shape)

    def _decode(i):
        mm[i] = np.asarray(Image.open(todo[i]).convert("RGB").resize((RES, RES), Image.BILINEAR))

    with ThreadPoolExecutor(32) as pool:
        list(pool.map(_decode, range(len(todo))))
    mm.flush()
    payload = {}
    for g in GROUPS:
        if not cells[g]["rows"]:
            continue
        payload[f"{g}.rows"] = np.stack(cells[g]["rows"])
        payload[f"{g}.Y"] = np.stack(cells[g]["Y"])
    np.savez(out / "cells.npz", **payload)
    (out / "index.json").write_text(json.dumps(dict(
        rows=args.rows, n_train=int(round(args.rows * (1 - args.test_frac))), min_px=MIN_PX,
        n_frames=len(todo), res=RES, seed=args.seed, label=LABEL,
        groups={g: dict(dim=GROUPS[g], n_cells=len(cells[g]["name"]),
                        rows=int(np.stack(cells[g]["rows"]).shape[1]),
                        n_train=int(round(np.stack(cells[g]["rows"]).shape[1] * (1 - args.test_frac))),
                        names=cells[g]["name"], split=cells[g]["split"])
                for g in GROUPS if cells[g]["name"]},
        cells=meta), indent=1))
    print(f"{len(todo)} rows at {RES}px, {mm.nbytes / 1e9:.2f} GB -> {out}")
    for g in GROUPS:
        if cells[g]["name"]:
            n_ood = sum(s == "OOD" for s in cells[g]["split"])
            print(f"  {g:<12} {len(cells[g]['name']):>4} cells "
                  f"({len(cells[g]['name']) - n_ood} ID / {n_ood} OOD)")


if __name__ == "__main__":
    main()
