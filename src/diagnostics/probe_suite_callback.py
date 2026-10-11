"""Training-time probe on the generated suite pack: per-cell ridge R^2 of object (and arm) state
from `emb`, fitted on a background thread every `every_n_steps` optimiser steps and logged under
`<prefix>/` against the probe's own step axis."""
import copy
import json
import threading
import time
from pathlib import Path

import numpy as np
import torch
from lightning.pytorch.callbacks import Callback

ALPHAS = np.logspace(-3, 6, 19)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
# wandb display names; the pack's group keys stay as the target definitions
DISPLAY = {"obj_in_cam": "obj", "eef_in_cam": "eef", "arm_qpos": "arm_qpos"}
ARM_GROUPS = ("eef_in_cam", "arm_qpos")


def _uncompiled(m):
    return getattr(m, "_orig_mod", m)


def _decompile_region(shadow):
    """Replace a region-compiled ViT forward (`_region_fn`, throughput.enable_region_encoder) on
    the shadow with the eager kernel, so the probe never shares dynamo state with training."""
    from throughput import _region
    for m in shadow.modules():
        if hasattr(m, "_region_fn"):
            m._region_fn = _region


def _batched_ridge(X, Y, n_tr, alphas, n_folds=5):
    """X (C,N,D), Y (C,N,k) -> per-cell (R^2, rmse, alpha) on rows[n_tr:]; alpha chosen by
    K-fold CV on the train rows. Standardisation, alpha search and refit see train rows only."""
    Xtr, Ytr, Xte, Yte = X[:, :n_tr], Y[:, :n_tr], X[:, n_tr:], Y[:, n_tr:]
    fold = torch.arange(n_tr, device=X.device) % n_folds
    score = torch.zeros(len(alphas), X.shape[0], device=X.device, dtype=X.dtype)
    for f in range(n_folds):
        tr, va = fold != f, fold == f
        P = _paths(Xtr[:, tr], Ytr[:, tr], Xtr[:, va], alphas)          # (A,C,m,k)
        t = Ytr[:, va]
        sst = ((t - t.mean(1, keepdim=True)) ** 2).sum(1)
        score += (1 - ((P - t) ** 2).sum(2) / sst).mean(-1)
    best = score.argmax(0)                                              # (C,)
    P = _paths(Xtr, Ytr, Xte, alphas)                                   # (A,C,m,k)
    P = P[best, torch.arange(X.shape[0], device=X.device)]
    sst = ((Yte - Yte.mean(1, keepdim=True)) ** 2).sum(1)
    r2 = (1 - ((P - Yte) ** 2).sum(1) / sst).mean(-1)
    rmse = ((P - Yte) ** 2).sum(-1).mean(-1).sqrt()
    return r2, rmse, alphas[best]


def _paths(Xtr, Ytr, Xte, alphas):
    """Ridge predictions on Xte for every alpha and every cell at once, (A,C,m,k); unpenalised
    intercept; solved through the eigendecomposition of the (D,D) Gram (fp64 retry on failure)."""
    mu = Xtr.mean(1, keepdim=True)
    sd = Xtr.std(1, keepdim=True, unbiased=False).clamp_min(1e-8)
    A, B = (Xtr - mu) / sd, (Xte - mu) / sd
    ym = Ytr.mean(1, keepdim=True)
    G = A.transpose(1, 2) @ A
    try:
        s, V = torch.linalg.eigh(G)
    except Exception:
        s, V = torch.linalg.eigh(G.double())
        s, V = s.to(A.dtype), V.to(A.dtype)
    VtAtY = V.transpose(1, 2) @ (A.transpose(1, 2) @ (Ytr - ym))        # (C,D,k)
    BV = B @ V                                                          # (C,m,D)
    d = 1.0 / (s.clamp_min(0).unsqueeze(0) + alphas[:, None, None])     # (A,C,D)
    return torch.einsum("cmd,acdk->acmk", BV, d.unsqueeze(-1) * VtAtY.unsqueeze(0)) + ym


def _fold_normalization(encoder, mean, std):
    """Fold x -> (x/255 - mean)/std into the patch-embed conv in place, so the encoder accepts
    raw uint8-valued floats. Done in the weights' current (fp32) dtype."""
    conv = encoder.embeddings.patch_embeddings.projection
    w = conv.weight.data                                                # (D,3,p,p)
    m = torch.as_tensor(mean, dtype=w.dtype, device=w.device).view(1, -1, 1, 1)
    s = torch.as_tensor(std, dtype=w.dtype, device=w.device).view(1, -1, 1, 1)
    shift = (w * (m / s)).sum(dim=(1, 2, 3))
    conv.weight.data = w / (255.0 * s)
    conv.bias.data = conv.bias.data - shift


class ProbeSuiteCallback(Callback):
    """Probe the frozen suite every `every_n_steps` optimiser steps (rank 0 only).

    pack:          directory holding frames.npy (n,H,W,3) uint8, cells.npz and index.json.
    batch:         rows staged and encoded per step.
    log_per_cell:  also log every cell's R^2 under `<prefix>_cell/<name>`.
    dtype:         shadow-encoder dtype (fp16 | bf16 | fp32).
    arm_every:     fit the arm groups (ARM_GROUPS) on every k-th firing only.
    shm:           read the pack from /dev/shm/<pack name> when a copy is staged there.

    The hook snapshots the weights on the training thread and runs encode + fit on a daemon
    thread; results are logged on a later hook call against the `<prefix>/step` axis."""

    def __init__(self, pack, every_n_steps=2000, batch=512, prefix="suite",
                 log_per_cell=False, dtype="fp16", arm_every=4, shm=True):
        super().__init__()
        self.pack = Path(pack)
        self.every_n_steps = int(every_n_steps)
        self.batch = int(batch)
        self.dtype = dict(fp16=torch.float16, bf16=torch.bfloat16,
                          fp32=torch.float32)[str(dtype)]
        self.prefix = prefix
        self.log_per_cell = log_per_cell
        self.arm_every = max(1, int(arm_every))
        self.shm = bool(shm)
        self._loaded = False
        self._shadow = None
        self._last = -1
        self._n_fired = 0
        self._thread = None
        self._pending = None            # (step, logs) awaiting a hook call to log
        self._lock = threading.Lock()
        self._axis_defined = False
        # empty (not unset): `_emit` reads it on the training thread even when `_load` failed
        self.cells = {}

    # ---------------------------------------------------------------- data
    def _pack_dir(self):
        """The pack to read: a /dev/shm copy if staged, else the configured path."""
        if not self.shm:
            return self.pack
        shm = Path("/dev/shm") / self.pack.name
        return shm if (shm / "frames.npy").exists() else self.pack

    def _load(self, device):
        """Memory-map the pack's frames, load every cell's rows and labels onto `device`, and
        allocate the pinned staging buffer."""
        src = self._pack_dir()
        idx = json.loads((src / "index.json").read_text())
        self.frames = np.load(src / "frames.npy", mmap_mode="r")
        if self.frames.ndim != 4:
            raise ValueError(f"pack {src}: frames.npy must be (n,H,W,3), got {self.frames.shape}")
        z = np.load(src / "cells.npz")
        self.cells = {}
        for g, info in idx["groups"].items():
            self.cells[g] = dict(
                rows=torch.as_tensor(z[f"{g}.rows"], dtype=torch.long),
                Y=torch.as_tensor(z[f"{g}.Y"], dtype=torch.float32, device=device),
                names=info["names"], n_train=info.get("n_train", idx.get("n_train")),
                is_ood=torch.as_tensor([s == "OOD" for s in info["split"]]))
        self.alphas = torch.as_tensor(ALPHAS, device=device, dtype=torch.float32)
        shp = (self.batch,) + self.frames.shape[1:]
        self._host = torch.empty(shp, dtype=torch.uint8)
        # pinning needs a CUDA context; on a CPU-only host the buffer is plain memory
        if torch.device(device).type == "cuda":
            self._host = self._host.pin_memory()
        self._loaded = True

    def shadow(self, model):
        """Snapshot `model.encoder` + `model.projector` into an eager, frozen copy in `self.dtype`
        with the input normalisation folded into the patch-embed conv. Returns fn: (b,3,H,W)
        raw-pixel-valued frames -> (b, D) emb = projector(CLS). Must be called on the training
        thread; the copy is rebuilt at every firing."""
        live = torch.nn.ModuleDict({"e": _uncompiled(model.encoder),
                                    "p": _uncompiled(model.projector)})
        sh = copy.deepcopy(live)
        _decompile_region(sh)
        _fold_normalization(sh["e"], IMAGENET_MEAN, IMAGENET_STD)
        sh = sh.to(self.dtype).eval()
        for p in sh.parameters():
            p.requires_grad_(False)
        self._shadow = sh

        def fn(x):
            cls = sh["e"](x, interpolate_pos_encoding=True).last_hidden_state[:, 0]
            return sh["p"](cls)
        return fn

    @torch.inference_mode()
    def _encode(self, encode_fn, device, rows=None):
        """(n_rows, D) float32 embeddings for `rows` (sorted row indices; default: the whole
        pack), staged through the pinned buffer as raw uint8."""
        n = len(self.frames) if rows is None else len(rows)
        out = []
        for i, c in enumerate(range(0, n, self.batch)):
            sl = slice(c, min(c + self.batch, n))
            if rows is None:
                chunk = self.frames[sl]
            else:
                r = rows[sl]
                # a contiguous run is read as one slice rather than gathered row by row
                chunk = (self.frames[r[0]:r[-1] + 1]
                         if len(r) and len(r) == r[-1] - r[0] + 1 else self.frames[r])
            nb = len(chunk)
            self._host[:nb].copy_(torch.from_numpy(np.ascontiguousarray(chunk)))
            # blocking copy: one staging buffer, refilled only after the transfer completes
            x = self._host[:nb].to(device)
            x = x.permute(0, 3, 1, 2)                                   # HWC -> CHW
            out.append(encode_fn(x.to(self.dtype)).float())
        return torch.cat(out)

    # ---------------------------------------------------------------- probe
    @torch.inference_mode()
    def run(self, encode_fn, device, retries=2, fit_arm=True):
        """Encode the rows the fitted groups reference and fit every cell; returns the log dict
        (`<prefix>/skipped` = 1.0 when the features were non-finite after `retries` re-encodes).
        fit_arm=False skips ARM_GROUPS."""
        if not self._loaded:
            self._load(device)
        groups = {g: c for g, c in self.cells.items()
                  if fit_arm or g not in ARM_GROUPS}
        if not groups:
            return {}
        need = torch.unique(torch.cat([c["rows"].reshape(-1) for c in groups.values()]))
        full = len(need) == len(self.frames)
        rows = None if full else need.numpy()
        remap = None if full else torch.full((len(self.frames),), -1, dtype=torch.long)
        if remap is not None:
            remap[need] = torch.arange(len(need))

        t0 = time.time()
        for _ in range(retries + 1):
            F = self._encode(encode_fn, device, rows=rows)
            n_bad = int((~torch.isfinite(F).all(1)).sum())
            if n_bad == 0:
                break
        if n_bad:
            print(f"[{self.prefix}] refusing to report: {n_bad} non-finite feature rows "
                  f"after {retries + 1} attempts", flush=True)
            return {f"{self.prefix}/skipped": 1.0}
        t_enc = time.time() - t0
        logs = {f"{self.prefix}/skipped": 0.0}

        # collapse monitors: eff_rank and mean_cos over the obj-group rows at every firing
        # (`obj`), and over all rows on a full firing (`all`); deterministic 2000-row sample
        def _collapse(Fp, pop):
            Fc = Fp - Fp.mean(0, keepdim=True)
            ev = torch.linalg.eigvalsh(((Fc.T @ Fc) / max(1, Fp.shape[0] - 1)).double()).clamp_min(0)
            logs[f"{self.prefix}/{pop}/eff_rank"] = float(ev.sum() ** 2 / ((ev ** 2).sum() + 1e-12))
            gen = torch.Generator(device="cpu").manual_seed(0)
            gi = torch.randperm(Fp.shape[0], generator=gen)[:2000].to(Fp.device)
            Fn = torch.nn.functional.normalize(Fp[gi], dim=1)
            G = Fn @ Fn.T
            n = G.shape[0]
            logs[f"{self.prefix}/{pop}/mean_cos"] = float((G.sum() - G.trace()) / (n * (n - 1)))
        obj_g = next((c for g, c in groups.items() if "obj" in g), None)
        if obj_g is not None:
            ro = obj_g["rows"].reshape(-1).unique()
            ro = ro if remap is None else remap[ro]
            _collapse(F[ro.to(device)], "obj")
        else:
            _collapse(F, "obj")
        if rows is None and obj_g is not None:
            _collapse(F, "all")
        for g, c in groups.items():
            gd = DISPLAY.get(g, g)
            r = c["rows"] if remap is None else remap[c["rows"]]
            X = F[r.to(device)]
            r2, rmse, alpha = _batched_ridge(X, c["Y"], c["n_train"], self.alphas)
            r2, rmse = r2.cpu().numpy(), rmse.cpu().numpy()
            for dom, m in (("id", ~c["is_ood"].numpy()), ("ood", c["is_ood"].numpy())):
                if not m.any():
                    continue
                v = r2[m]
                p = np.percentile(v, [10, 50, 90])
                logs[f"{self.prefix}/{gd}/{dom}/p50"] = float(p[1])
                logs[f"{self.prefix}/{gd}/{dom}/p10"] = float(p[0])
                logs[f"{self.prefix}/{gd}/{dom}/p90"] = float(p[2])
                logs[f"{self.prefix}/{gd}/{dom}/mean"] = float(v.mean())
                logs[f"{self.prefix}/{gd}/{dom}/rmse"] = float(np.median(rmse[m]))
            if self.log_per_cell:
                for nm, v in zip(c["names"], r2):
                    logs[f"{self.prefix}_cell/{nm}"] = float(v)
        logs[f"{self.prefix}/seconds"] = time.time() - t0
        logs[f"{self.prefix}/seconds_encode"] = t_enc
        return logs

    # ---------------------------------------------------------------- hook
    def _emit(self, trainer, step, logs):
        """Log `logs` against `<prefix>/step` (results land after global_step has moved on)."""
        if not logs:
            return
        logs = dict(logs)
        logs[f"{self.prefix}/step"] = float(step)
        for lgr in (trainer.loggers or []):
            exp = getattr(lgr, "experiment", None)
            if exp is not None and hasattr(exp, "define_metric"):
                if not self._axis_defined:
                    exp.define_metric(f"{self.prefix}/step")
                    exp.define_metric(f"{self.prefix}/*", step_metric=f"{self.prefix}/step")
                    self._axis_defined = True
                exp.log(logs)
            else:
                lgr.log_metrics(logs, step=step)
        head = " ".join(f"{DISPLAY.get(g, g)}={logs[k]:+.3f}" for g in self.cells
                        if (k := f"{self.prefix}/{DISPLAY.get(g, g)}/id/p50") in logs)
        print(f"[{self.prefix}] step {step} {head or 'SKIPPED'} "
              f"eff_rank={logs.get(f'{self.prefix}/obj/eff_rank', float('nan')):.1f} "
              f"({logs.get(f'{self.prefix}/seconds', 0):.1f}s, "
              f"encode {logs.get(f'{self.prefix}/seconds_encode', 0):.1f}s)", flush=True)

    def _drain(self, trainer):
        with self._lock:
            pending, self._pending = self._pending, None
        if pending is not None:
            self._emit(trainer, *pending)

    def on_train_batch_end(self, trainer, pl_module, *a, **k):
        if not trainer.is_global_zero:
            return
        self._drain(trainer)
        step = int(trainer.global_step)
        if step == self._last or step % self.every_n_steps:
            return
        if self._thread is not None and self._thread.is_alive():
            print(f"[{self.prefix}] step {step}: previous probe still running, skipping",
                  flush=True)
            return
        self._last = step
        model = getattr(pl_module, "model", pl_module)
        device = next(model.parameters()).device
        fn = self.shadow(model)                       # must be on the training thread
        self._n_fired += 1
        fit_arm = (self._n_fired % self.arm_every) == 1 or self.arm_every == 1

        def work():
            try:
                logs = self.run(fn, device, fit_arm=fit_arm)
            except Exception as e:                    # never let a diagnostic kill training
                print(f"[{self.prefix}] step {step} failed: {type(e).__name__}: {e}", flush=True)
                logs = {f"{self.prefix}/skipped": 1.0}
            with self._lock:
                self._pending = (step, logs)

        self._thread = threading.Thread(target=work, daemon=True,
                                        name=f"{self.prefix}-probe")
        self._thread.start()

    def on_train_end(self, trainer, pl_module):
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=300)
        if trainer.is_global_zero:
            self._drain(trainer)
