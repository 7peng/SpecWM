import math
import os
import re
from pathlib import Path

import numpy as np
import torch
from stable_pretraining import data as dt
from stable_worldmodel.data.formats.hdf5 import HDF5Dataset as _SwmHDF5Dataset
from lightning.pytorch.callbacks import Callback


def _register_omegaconf_resolvers():
    """Register the `${eval:...}` resolver the training configs and saved checkpoint configs use."""
    try:
        from omegaconf import OmegaConf
        OmegaConf.register_new_resolver("eval", lambda e: eval(e), replace=True)
    except Exception:
        pass


def load_env():
    """Load repo-root .env then .env.example into os.environ; already-set vars win."""
    _register_omegaconf_resolvers()
    root = Path(__file__).resolve().parents[1]
    for name in (".env", ".env.example"):
        p = root / name
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def get_uint8_pixels_pipeline(source: str, target: str, img_size: int = 224):
    """Resized uint8 frames only; ImageNet normalization runs on the GPU (GPUPixelNormalize)."""
    to_image = dt.transforms.ToImage(
        dtype=torch.uint8, scale=False, source=source, target=target)
    resize = dt.transforms.Resize(img_size, source=source, target=target)
    return dt.transforms.Compose(to_image, resize)


class GPUPixelNormalize(torch.nn.Module):
    """GPU ImageNet normalizer for uint8 pixel batches. The mean/std/luma buffers match the
    checkpoints on disk."""

    def __init__(self):
        super().__init__()
        imagenet = dt.dataset_stats.ImageNet
        self.register_buffer("mean", torch.tensor(imagenet["mean"]).view(1, 1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(imagenet["std"]).view(1, 1, 3, 1, 1))
        self.register_buffer("luma", torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1))

    def forward(self, pixels_u8):
        """(B,T,C,H,W) uint8 -> (ImageNet-normalized float, None)."""
        assert pixels_u8.dtype == torch.uint8, (
            f"GPUPixelNormalize expects uint8 pixels, got {pixels_u8.dtype}: the loader is already "
            f"normalizing on the CPU, so this forward would normalize twice")
        x = pixels_u8.float().div_(255.0)
        return x.sub_(self.mean).div_(self.std), None


class RowReadHDF5Dataset(_SwmHDF5Dataset):
    """swm's HDF5Dataset with a per-row read for columns chunked one frame per chunk, a clip table
    whose last kept frame lies inside the episode, and optional far frames.

    Columns chunked (1, H, W, C) are read one kept row at a time; every other column takes swm's
    slab read. `action` keeps the full raw span (the base class reshapes it to
    (num_steps, frameskip * dim)); `keys_to_cache` columns never touch the file.

    `far_frames` extra rows per clip are drawn from the same episode, |gap| log-uniform in
    `far_gap` raw frames from the clip centre, and appended after the contiguous clip (no action
    rows)."""

    def __init__(self, *a, **k):
        self.far_frames = int(k.pop("far_frames", 0) or 0)
        self.far_gap = tuple(k.pop("far_gap", (300, 3000)))
        super().__init__(*a, **k)
        self._row_read_cols = {}
        # the final frame's action block is zero-padded past the episode end in _load_slice
        reach = (self.num_steps - 1) * self.frameskip + 1
        kept = [(ep, s) for ep, L in enumerate(self.lengths) if L >= reach for s in range(L - reach + 1)]
        dropped = [(ep, L) for ep, L in enumerate(self.lengths) if L < reach]
        print(f"[clip-table] {len(kept)} clips ({self.num_steps} x fs {self.frameskip}, reach {reach}) from "
              f"{len(self.lengths) - len(dropped)} episodes; {len(dropped)} episodes shorter than {reach} "
              f"frames dropped ({sum(L for _, L in dropped)} frames)", flush=True)
        self.clip_indices = kept

    def __getitem__(self, idx):
        if not self.far_frames:
            return super().__getitem__(idx)
        # the far rows join the raw clip before the transform, so both pass through it together
        ep_idx, start = self.clip_indices[idx]
        tf, self.transform = self.transform, None
        try:
            steps = self._load_slice(ep_idx, start, start + self.span)
        finally:
            self.transform = tf
        self._append_far(steps, ep_idx, start)
        steps = tf(steps) if tf else steps
        if "action" in steps:
            steps["action"] = steps["action"].reshape(self.num_steps, -1)
        return steps

    def _append_far(self, steps, ep_idx, start):
        L = int(self.lengths[ep_idx])
        centre = start + ((self.num_steps - 1) * self.frameskip) // 2
        lo, hi = float(self.far_gap[0]), float(self.far_gap[1])
        room = {1: L - 1 - centre, -1: centre}
        t = []
        for s in (torch.randint(0, 2, (self.far_frames,)) * 2 - 1).tolist():
            s = s if room[s] >= lo or room[-s] < lo else -s
            top = max(lo, min(hi, room[s]))
            gap = math.exp(torch.empty(()).uniform_(math.log(lo), math.log(top)).item())
            t.append(int(min(max(round(centre + s * gap), 0), L - 1)))
        rows = sorted(int(self.offsets[ep_idx]) + x for x in t)
        for col in self._keys:
            if col == "action" or col not in steps or not torch.is_tensor(steps[col]):
                continue
            src = self._cache[col] if col in self._cache else self.h5_file[col]
            data = np.stack([src[i] for i in rows])
            x = torch.from_numpy(data)
            if data.ndim == 4 and data.shape[-1] in (1, 3):
                x = x.permute(0, 3, 1, 2)
            steps[col] = torch.cat([steps[col], x.to(steps[col].dtype)], 0)

    def _row_read(self, col):
        if col not in self._row_read_cols:
            d = self.h5_file[col]
            self._row_read_cols[col] = d.chunks is not None and d.chunks[0] == 1 and d.ndim >= 2
        return self._row_read_cols[col]

    def _load_slice(self, ep_idx, start, end):
        self._open()
        g_start, g_end = self.offsets[ep_idx] + start, self.offsets[ep_idx] + end
        steps = {}
        for col in self._keys:
            ep_end = self.offsets[ep_idx] + self.lengths[ep_idx]
            if col in self._cache:
                data = self._cache[col][g_start:min(g_end, ep_end)]
                if col != "action":
                    data = data[:: self.frameskip]
            elif col != "action" and self.frameskip > 1 and self._row_read(col):
                d = self.h5_file[col]
                data = np.stack([d[i] for i in range(g_start, min(g_end, ep_end), self.frameskip)])
            else:
                data = self.h5_file[col][g_start:min(g_end, ep_end)]
                if col != "action":
                    data = data[:: self.frameskip]
            if col == "action" and g_end > ep_end:
                data = np.concatenate([data, np.zeros((g_end - ep_end,) + data.shape[1:], data.dtype)])
            if data.dtype == np.object_ or data.dtype.kind in ("S", "U"):
                val = data[0] if len(data) > 0 else b""
                steps[col] = val.decode() if isinstance(val, bytes) else val
            else:
                steps[col] = torch.from_numpy(data)
                if data.ndim == 4 and data.shape[-1] in (1, 3):
                    steps[col] = steps[col].permute(0, 3, 1, 2)
        return self.transform(steps) if self.transform else steps


class PickleableHDF5Wrapper:
    """Pickleable H5 dataset."""

    def __init__(self, subset):
        self.subset = subset

    def __getstate__(self):
        return self.__dict__.copy()

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __getitem__(self, idx):
        return self.subset[idx]

    def __len__(self):
        return len(self.subset)


class ZScoreNormalizer:
    """Picklable z-score normalizer; a class (not a closure) so it survives DataLoader
    worker pickling."""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        return ((x - self.mean) / self.std).float()


# columns served as they are: the identity columns (looked up by value) and a frozen teacher's cached
# latents (in the teacher's own units)
NO_NORMALIZE = frozenset({"ep_idx", "step_idx", "teacher_z"})

# the columns the loader may z-score; every other non-pixel column of keys_to_load must be named
# in NO_NORMALIZE or in the recipe's `raw_columns` (classify_columns)
NORMALIZE = frozenset({
    "action",             # the predictor's conditioning
    "observation",        # scene's env observation vector
    "qpos",               # simulator state as a feature; arms that read it in metres name it in raw_columns
    "scene_obs",          # CALVIN scene state
    "robot_obs",          # CALVIN proprio
})
assert not (NORMALIZE & NO_NORMALIZE), (
    f"a column cannot be both always and never z-scored: {sorted(NORMALIZE & NO_NORMALIZE)}")


def classify_columns(keys_to_load, raw_cols):
    """Split keys_to_load into (z-scored, served raw), refusing anything classified nowhere."""
    raw_cols = {str(c) for c in (raw_cols or [])}
    norm, raw = [], []
    for col in keys_to_load:
        col = str(col)
        if col.startswith("pixels"):
            continue
        # depth_z arrives pre-normalized (per-cell z-score fit at build time)
        if col.startswith("depth_z") or col in NO_NORMALIZE or col in raw_cols:
            raw.append(col)
            continue
        assert col in NORMALIZE, (
            f"keys_to_load names '{col}', which is classified nowhere: not in utils.NORMALIZE, "
            f"not in NO_NORMALIZE, and not in this recipe's data.dataset.raw_columns.")
        norm.append(col)
    return norm, raw


def get_column_normalizer(dataset, source: str, target: str, tile: int = 1):
    """Normalizer for one column of the dataset. `tile` repeats the per-column stats `tile` times
    to match frameskip action-block stacking."""
    col_data = dataset.get_col_data(source)
    data = torch.from_numpy(np.array(col_data))
    data = data[~torch.isnan(data).any(dim=1)]
    mean = data.mean(0, keepdim=True).clone()
    std = data.std(0, keepdim=True).clone()
    dead = (std < 1e-6).flatten().nonzero().flatten().tolist()
    if dead:
        print(f"[normalizer] {source}: constant columns {dead} (std < 1e-6) clamped to 1e-6", flush=True)
        std = std.clamp_min(1e-6)
    if tile > 1:
        mean = mean.repeat(1, tile)
        std = std.repeat(1, tile)
    return dt.transforms.WrapTorchTransform(ZScoreNormalizer(mean, std), source=source, target=target)


class SaveCkptCallback(Callback):
    """Save the model weights after each epoch through swm's save_pretrained."""

    def __init__(self, run_name, cfg, epoch_interval: int = 1):
        super().__init__()
        self.run_name = run_name
        self.cfg = cfg
        self.epoch_interval = epoch_interval

    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)

        if trainer.is_global_zero:
            if (trainer.current_epoch + 1) % self.epoch_interval == 0:
                self._save(pl_module.model, trainer.current_epoch + 1)

            if (trainer.current_epoch + 1) == trainer.max_epochs:
                self._save(pl_module.model, trainer.current_epoch + 1)

    def _save(self, model, epoch):
        from stable_worldmodel.wm.utils import save_pretrained
        save_pretrained(
            _UncompiledStateDict(model),
            run_name=self.run_name,
            config=self.cfg,
            filename=f'weights_epoch_{epoch}.pt',
        )


class AuxStateCallback(Callback):
    """Save the attached loss modules' state (`heads_epoch_N.pt`) beside each weights_epoch_N.pt,
    so a warm restart restores them too."""

    def __init__(self, run_name, module_names, epoch_interval: int = 1):
        super().__init__()
        self.run_name = run_name
        self.module_names = list(module_names)
        self.epoch_interval = epoch_interval

    def on_train_epoch_end(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        epoch = trainer.current_epoch + 1
        if epoch % self.epoch_interval != 0 and epoch != trainer.max_epochs:
            return
        import stable_worldmodel as swm
        out = Path(swm.data.utils.get_cache_dir(), "checkpoints", self.run_name)
        out.mkdir(parents=True, exist_ok=True)
        sd = {n: getattr(pl_module, n).state_dict()
              for n in self.module_names if hasattr(pl_module, n)}
        tmp = out / f".heads_epoch_{epoch}.pt.tmp"
        torch.save(sd, tmp)
        tmp.rename(out / f"heads_epoch_{epoch}.pt")


def strip_orig_mod(state_dict):
    """Remove the `._orig_mod` segment torch.compile inserts into state_dict keys."""
    return {k.replace('._orig_mod', ''): v for k, v in state_dict.items()}


class _UncompiledStateDict:
    """Exposes a compile-stripped `state_dict()`, the one method save_pretrained calls."""

    def __init__(self, model):
        self._sd = strip_orig_mod(model.state_dict())

    def state_dict(self):
        return self._sd


class StepSnapshotCallback(Callback):
    """Every `every` batches within [start, stop], save the model state_dict (compile-stripped),
    the optimizer state_dicts and the attached loss modules' state to <out_dir>/step_<N>.pt.
    Snapshot names stay on the run's cumulative batch axis across resumes (`step_offset`)."""

    def __init__(self, out_dir, start, stop, every, module_names=(), step_offset=0):
        super().__init__()
        self.out_dir = Path(out_dir)
        self.start, self.stop, self.every = int(start), int(stop), int(every)
        self.module_names = list(module_names)
        self.step_offset = int(step_offset)
        self._done = set()

    def on_train_start(self, trainer, pl_module):
        # a full Lightning resume restores global_step (optimizer steps); the local batch counter
        # below starts at 0, so the offset is taken from it
        opts = trainer.optimizers if isinstance(trainer.optimizers, (list, tuple)) \
            else [trainer.optimizers]
        resumed = int(getattr(trainer, "global_step", 0) or 0) // max(1, len(opts))
        if resumed > self.step_offset:
            self.step_offset = resumed
            print(f"[step-snap] full resume at batch {resumed}: snapshot names continue from "
                  f"there (cumulative step axis preserved)", flush=True)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._batches = getattr(self, "_batches", 0) + 1
        gs = self._batches + self.step_offset
        if (not trainer.is_global_zero or gs < self.start or gs > self.stop
                or gs % self.every != 0 or gs in self._done
                or (self.step_offset > 0 and gs <= self.step_offset)):
            return
        self._done.add(gs)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        opts = trainer.optimizers if isinstance(trainer.optimizers, (list, tuple)) else [trainer.optimizers]
        path = self.out_dir / f"step_{gs}.pt"
        heads = {n: getattr(pl_module, n).state_dict()
                 for n in self.module_names if hasattr(pl_module, n)}
        torch.save({
            "step": gs,
            "model": strip_orig_mod(pl_module.model.state_dict()),
            "optimizers": [o.state_dict() for o in opts],
            "heads": heads,
        }, path)
        print(f"[step-snap] saved model+optimizer+heads({','.join(heads) or 'none'}) "
              f"@ step {gs} -> {path}", flush=True)


class LightningCkptPointer(Callback):
    """Append the directory holding this run_id's Lightning checkpoints to a pointer file, so the
    next leg can resume full state (spt names its checkpoint dirs after a hash, not the run_id)."""

    def __init__(self, pointer_path):
        super().__init__()
        self.pointer_path = Path(pointer_path)

    def on_train_start(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        d = getattr(getattr(trainer, "checkpoint_callback", None), "dirpath", None)
        if not d:
            base = getattr(trainer, "log_dir", None) or getattr(trainer, "default_root_dir", None)
            d = Path(base) / "checkpoints" if base else None
        if not d:
            print("[ckpt-pointer] WARNING: no checkpoint dirpath; next leg cannot full-resume",
                  flush=True)
            return
        d = str(Path(d).resolve())
        try:
            prev = [ln.strip() for ln in self.pointer_path.read_text().splitlines() if ln.strip()]
        except OSError:
            prev = []
        if prev and prev[-1] == d:
            return
        self.pointer_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.pointer_path, "a") as f:
            f.write(d + "\n")
        print(f"[ckpt-pointer] recorded {d}", flush=True)


def find_resume_ckpt(pointer_path, legacy_glob_root=None, run_name=None):
    """The highest-step Lightning checkpoint across every directory the pointer names, or None.
    Callers gate on `auto_resume`."""
    try:
        dirs = [ln.strip() for ln in Path(pointer_path).read_text().splitlines() if ln.strip()]
    except OSError:
        dirs = []
    cands = [c for d in dirs for c in Path(d).glob("epoch=*-step=*.ckpt")]
    if legacy_glob_root and run_name:
        cands += list(Path(legacy_glob_root).glob(f"{run_name}/checkpoints/epoch=*-step=*.ckpt"))
    if not cands:
        return None

    def _step(p):
        m = re.search(r"step=(\d+)", p.name)
        return int(m.group(1)) if m else -1

    return max(cands, key=_step).resolve()


class StepLossLogger(Callback):
    """Append per-step training losses to `<run_dir>/loss_log.csv` every `every` optimizer steps
    (rank 0 only), from `trainer.callback_metrics`."""

    def __init__(self, run_dir, every=50, keys=None):
        super().__init__()
        self.run_dir = Path(run_dir)
        self.every = max(1, int(every))
        self.keys = keys or [
            "fit/loss", "fit/pred_loss", "fit/sigreg_loss", "fit/spec_loss", "fit/pred_anchor_loss",
            "fit/qrl_loss", "fit/emb_norm", "fit/emb_norm_std",
        ]
        self._wrote_header = False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not trainer.is_global_zero:
            return
        gs = int(trainer.global_step)
        if gs == 0 or gs % self.every != 0:
            return
        cm = trainer.callback_metrics

        def get(k):
            v = cm.get(k, None)
            try:
                return float(v)
            except Exception:
                return float("nan")
        try:
            lr = float(trainer.optimizers[0].param_groups[0]["lr"])
        except Exception:
            lr = float("nan")
        path = self.run_dir / "loss_log.csv"
        try:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            new = not path.exists()
            with open(path, "a") as f:
                if new:
                    f.write("step,epoch,lr," + ",".join(k.split("/")[-1] for k in self.keys) + "\n")
                    self._wrote_header = True
                f.write(f"{gs},{int(trainer.current_epoch)},{lr:.4e}," +
                        ",".join(f"{get(k):.6f}" for k in self.keys) + "\n")
        except Exception as e:
            print(f"[loss_log] write failed: {e}", flush=True)
