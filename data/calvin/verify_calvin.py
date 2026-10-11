"""Sanity-check an extracted CALVIN split under $CALVIN_ROOT: per sub-split, count the
episode_*.npz frame files and compare against the frame intervals in ep_start_end_ids.npy.

    python data/calvin/verify_calvin.py [ABC_D]
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np

from utils import load_env


def main():
    load_env()
    split = sys.argv[1] if len(sys.argv) > 1 else "ABC_D"
    root = Path(os.environ["CALVIN_ROOT"]) / f"task_{split}"
    if not root.is_dir():
        sys.exit(f"missing {root}")
    ok = True
    for sub in ("training", "validation"):
        d = root / sub
        n_frames = sum(1 for _ in d.glob("episode_*.npz"))
        ids = np.load(d / "ep_start_end_ids.npy")
        span = int((ids[:, 1] - ids[:, 0] + 1).sum())
        good = span == n_frames
        ok &= good
        print(f"{sub}: {n_frames} frames, {len(ids)} episodes, "
              f"interval span {span} [{'OK' if good else 'MISMATCH'}]")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
