"""Helpers shared by the state-recoverability probe: dataset resolution, the frame transform,
and `name=col[a:b]` target parsing/extraction from an HDF5 file."""
import os
import re
import sys
from pathlib import Path

import hdf5plugin  # noqa: F401  registers the HDF5 filters some datasets use
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from utils import load_env

load_env()
import stable_pretraining as spt
from torchvision.transforms import v2 as transforms


def resolve_dataset(name: str) -> Path:
    """Path of dataset `name`: an existing absolute path as is, else `<name>.h5` under
    $STABLEWM_HOME (or its datasets/ subdir, or the cwd)."""
    p = Path(name).expanduser()
    if p.is_absolute() and p.exists():
        return p
    home = Path(os.environ.get("STABLEWM_HOME", "~/.stable-wm")).expanduser()
    fname = name if name.endswith(".h5") else f"{name}.h5"
    for c in (home / fname, home / "datasets" / fname, Path(fname)):
        if c.exists():
            return c
    return home / fname  # let h5py raise with a meaningful path


_TARGET_RE = re.compile(r"^([A-Za-z0-9_]+)=([A-Za-z0-9_]+)(?:\[([^\]]+)\])?$")


def parse_target(spec: str):
    """"name=col", "name=col[a:b]" or "name=col[i]" -> (name, col, idx) with idx None, a slice
    or an int."""
    m = _TARGET_RE.match(spec)
    if not m:
        raise ValueError(f"bad --target {spec!r}: expected name=col or name=col[a:b] or name=col[i]")
    name, col, sub = m.groups()
    if sub is None:
        idx = None
    elif ":" in sub:
        a, b = sub.split(":", 1)
        idx = slice(int(a) if a else None, int(b) if b else None)
    else:
        idx = int(sub)
    return name, col, idx


def extract(f, col, idx, rows):
    """Column `col` of h5 file `f` at `rows` (sorted unique indices), flattened per row and
    sliced by `idx` -> (len(rows), k) array."""
    if col not in f:
        raise KeyError(f"column {col!r} not in dataset; have: {sorted(f.keys())}")
    data = f[col][rows]
    if data.ndim == 1:
        data = data[:, None]
    flat = data.reshape(len(data), -1)
    if idx is None:
        return flat
    if isinstance(idx, slice):
        return flat[:, idx]
    return flat[:, idx:idx + 1]


def img_transform(img_size: int):
    """uint8 HWC frame -> float CHW tensor, ImageNet-normalised and resized to `img_size`."""
    return transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(**spt.data.dataset_stats.ImageNet),
        transforms.Resize(size=img_size),
    ])
