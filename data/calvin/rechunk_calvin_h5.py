#!/usr/bin/env python
"""Re-chunk a CALVIN pack so `pixels` holds ONE frame per HDF5 chunk.

WHY. swm's `HDF5Dataset._load_slice` reads the full raw span of a clip and keeps every
`frameskip`-th row. With 64-frame chunks every chunk in the span holds a kept row, so the read
IS the whole span: at frameskip 20 that is 700 x 120 KB = 82 MB per clip to keep 4 MB, ~11 GB/s
across four ranks, and the loader gates the step (experiments/throughput_calvin/PLAN.md). With
one frame per chunk the loader (`utils.RowReadHDF5Dataset`) reads the 35 kept rows directly:
4 MB per clip, measured 1.3 ms against 24 ms.

WHAT. Every dataset except `pixels` is copied verbatim (same dtype, layout, attrs); `pixels`
is rewritten with chunks (1, H, W, C), uncompressed as before. The values are identical, so a
model trained on the old file resumes on the new one unchanged. Write to a temporary name and
`mv` over the original once `verify_rechunk.py` passes: the `calvin_abc_d_qmz.h5` sidecar
reaches `pixels` through an ExternalLink BY FILE NAME.

    python data/calvin/rechunk_calvin_h5.py $CALVIN_STAGE_DIR/calvin_abc_d.h5 \
        $CALVIN_STAGE_DIR/calvin_abc_d.h5.new

`build_calvin_h5.py` writes the 1-frame layout directly now; this tool exists for packs built
before 4 Sep 2026 (the /scratch master among them).
"""
import argparse
import os
import time

import h5py
import hdf5plugin  # noqa: F401  (registers the blosc filter for packs that use it)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--block", type=int, default=2048, help="frames per copy slab")
    ap.add_argument("--uncompress", action="store_true",
                    help="accept a filtered (blosc) `pixels` and write it as PLAIN 1-frame chunks -- the "
                         "OGBench scene/cube packs are blosc-filtered 50-frame chunks; every read then "
                         "decodes the whole chunk, and a 10-frame clip at fs5 decodes 100 frames to keep 10")
    a = ap.parse_args()
    assert not os.path.exists(a.dst), f"{a.dst} exists; remove it first"

    with h5py.File(a.src, "r") as s, h5py.File(a.dst, "w") as d:
        for k in s.keys():
            lk = s.get(k, getlink=True)
            assert isinstance(lk, h5py.HardLink), (
                f"{k} is a {type(lk).__name__}: re-chunk the file it points at, not the sidecar")
        for k in s.keys():
            ds = s[k]
            if k == "pixels":
                assert ds.ndim == 4, ds.shape
                if ds.id.get_create_plist().get_nfilters():
                    assert a.uncompress, ("filtered pack (blosc): pass --uncompress to write `pixels` as "
                                          "plain 1-frame chunks; values are identical, only the layout changes")
                out = d.create_dataset(k, ds.shape, ds.dtype, chunks=(1,) + ds.shape[1:])
                N = ds.shape[0]
                t0 = time.time()
                for bi, at in enumerate(range(0, N, a.block)):
                    n = min(a.block, N - at)
                    out[at:at + n] = ds[at:at + n]
                    if bi % 25 == 0 or at + n == N:
                        el = time.time() - t0
                        rate = (at + n) / max(el, 1e-9)
                        print(f"  pixels {at + n:>9,}/{N:,} {(at + n) / N:5.1%} {rate:,.0f} fr/s "
                              f"{rate * ds.dtype.itemsize * ds[0].size / 1e9:.2f} GB/s "
                              f"eta {(N - at - n) / rate / 60:5.1f} min", flush=True)
            else:
                out = d.create_dataset(k, data=ds[...], chunks=ds.chunks, compression=ds.compression,
                                       compression_opts=ds.compression_opts)
                print(f"  {k} {ds.shape} {ds.dtype} chunks={ds.chunks} copied", flush=True)
            for ak, av in ds.attrs.items():
                out.attrs[ak] = av
        for ak, av in s.attrs.items():
            d.attrs[ak] = av
    print(f"wrote {a.dst} ({os.path.getsize(a.dst) / 1e9:.1f} GB)", flush=True)


if __name__ == "__main__":
    main()
