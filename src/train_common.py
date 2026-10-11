"""Shared training scaffolding for train_lewm.py / train_spec.py: data, model, optimizer,
callbacks, trainer. Each entry script passes its own `forward_fn` and `build_loss_modules`."""
import json
import os
import re
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import numpy as np
import torch
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict

from optim import MuonAdamW
from throughput import enable_fused_attention, enable_region_encoder
from utils import (RowReadHDF5Dataset, get_column_normalizer,
                   get_uint8_pixels_pipeline, load_env, AuxStateCallback, SaveCkptCallback,
                   PickleableHDF5Wrapper, StepLossLogger, LightningCkptPointer,
                   find_resume_ckpt)

from utils import classify_columns

load_env()


class _WorldSizeGuard(Callback):
    """cfg.trainer.devices is read as the world size (it sizes the LR schedule), so the process
    group must match it."""

    def __init__(self, devices):
        self.want = int(devices) if isinstance(devices, int) or str(devices).isdigit() else None

    def on_train_start(self, trainer, pl_module):
        import torch.distributed as _d
        if self.want is None:      # "auto" or an explicit device list: nothing to compare against
            print("==> world size guard INACTIVE (trainer.devices is not an int)", flush=True)
            return
        got = _d.get_world_size() if (_d.is_available() and _d.is_initialized()) else 1
        assert got == self.want, (
            f"trainer.devices={self.want} but the process group has {got} rank(s). devices is read "
            f"as the world size (LR schedule, sampler length, gathered pair width), so this run "
            f"would train a DIFFERENT objective than the config describes. If one rank is intended, "
            f"set trainer.devices=1. If not: lightning defers launching to SLURM unless "
            f"SLURM_JOB_NAME is 'bash'/'interactive', and `srun <script>` names the step after the "
            f"script -- export SLURM_JOB_NAME=bash before python.")
        print(f"==> world size guard: {got} rank(s) == trainer.devices", flush=True)


class _ValMetricLogger(Callback):
    """Append validation metrics to <run_dir>/val_log.csv at each validation end (rank 0)."""

    def __init__(self, run_dir, keys):
        super().__init__()
        self.run_dir = Path(run_dir)
        self.keys = keys

    def on_validation_epoch_end(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        cm = trainer.callback_metrics

        def g(k):
            v = cm.get(k, None)
            try:
                return float(v)
            except Exception:
                return float("nan")
        path = self.run_dir / "val_log.csv"
        try:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            new = not path.exists()
            with open(path, "a") as f:
                if new:
                    f.write("step,epoch," + ",".join(k.split("/")[-1] for k in self.keys) + "\n")
                f.write(f"{int(trainer.global_step)},{int(trainer.current_epoch)}," +
                        ",".join(f"{g(k):.6f}" for k in self.keys) + "\n")
        except Exception as e:
            print(f"[val_log] write failed: {e}", flush=True)


# spt bug workaround: on_train_start calls len(self.optimizers()), but Lightning returns a
# single optimizer (not a list) when only one is configured.
_spt_optimizers = spt.Module.optimizers


def _spt_optimizers_as_list(self, *args, **kwargs):
    opts = _spt_optimizers(self, *args, **kwargs)
    return opts if isinstance(opts, (list, tuple)) else [opts]


spt.Module.optimizers = _spt_optimizers_as_list

# spt bug workaround: WandbCheckpoint.on_load_checkpoint deletes the live wandb run and re-inits
# with the checkpoint's stored id; the stored id is dropped so the logger's own id stands.
from stable_pretraining.callbacks import checkpoint_sklearn as _spt_ckpt

_spt_wandb_on_load = _spt_ckpt.WandbCheckpoint.on_load_checkpoint


def _spt_wandb_on_load_keep_run(self, trainer, pl_module, checkpoint):
    checkpoint.pop("wandb", None)
    return _spt_wandb_on_load(self, trainer, pl_module, checkpoint)


_spt_ckpt.WandbCheckpoint.on_load_checkpoint = _spt_wandb_on_load_keep_run


# Two per-step host<->device syncs inside Lightning, patched out.
#
# 1. `LightningModule.log(name, <python number>)` builds the value with a blocking H2D copy; a fill
#    kernel carries the scalar as an argument instead.
import numbers as _numbers
from lightning.pytorch.core.module import LightningModule as _LM
_lm_to_tensor = _LM._LightningModule__to_tensor


def _lm_to_tensor_nosync(self, value, name):
    if isinstance(value, _numbers.Number):
        value = torch.full((), value, device=self.device, dtype=torch.get_default_dtype())
    return _lm_to_tensor(self, value, name)


_LM._LightningModule__to_tensor = _lm_to_tensor_nosync

# 2. Under DDP the training loop opens every batch with a blocking sigterm broadcast; the patch
#    keeps the semantics with a lag of a step or two (non-blocking copy, read once its event lands).
from lightning.pytorch.loops.training_epoch_loop import _TrainingEpochLoop as _TEL
import lightning.pytorch.loops.training_epoch_loop as _tel_mod
assert hasattr(_TEL, "_broadcast_sigterm_tensor"), "lightning changed; re-check the sigterm patch"


def _broadcast_sigterm_lagged(self):
    # every rank issues the broadcast EVERY step (a collective skipped on one rank deadlocks the
    # others); the flag comes back through a non_blocking copy into pinned host memory behind a
    # CUDA event, and completed copies are read on later steps by polling the event -- no `.item()`,
    # so the host never waits on the stream here
    import collections
    q = getattr(self, "_sigterm_inflight", None)
    if q is None:
        q = self._sigterm_inflight = collections.deque()
    while q and q[0][0].query():
        _, host = q.popleft()
        if int(host[0]) == 1:
            import contextlib
            with contextlib.suppress(Exception):
                torch.distributed.barrier()
            raise _tel_mod.SIGTERMException()
    dev = self.trainer.strategy.root_device
    flag = torch.full((1,), 1 if getattr(self.trainer, "received_sigterm", False) else 0,
                      dtype=torch.long, device=dev)
    torch.distributed.broadcast(flag, src=0)
    host = torch.empty(1, dtype=torch.long).pin_memory()
    host.copy_(flag, non_blocking=True)
    ev = torch.cuda.Event()
    ev.record()
    q.append((ev, host))


_TEL._broadcast_sigterm_tensor = _broadcast_sigterm_lagged

torch.set_float32_matmul_precision("high")

# one growable reservation per stream; an exported value still wins
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def log_step_metrics(self, output, emb, stage):
    """Shared per-step logging: emb norm/DC/eff-rank diagnostics + every *loss* key (+ the
    quasimetric_* and gram_corr_* series) pushed to the Lightning logger."""
    per_sample_norm = emb.detach().norm(dim=-1)
    output["emb_norm"] = per_sample_norm.mean()
    output["emb_norm_std"] = per_sample_norm.std()
    with torch.no_grad():
        # log the shared DC vector apart from the centered deviation
        e = emb.detach().reshape(-1, emb.shape[-1]).float()
        dev = e - e.mean(0)
        dev_n = dev.norm(dim=-1)
        output["emb_dc_norm"] = e.mean(0).norm()
        output["emb_dev_norm"] = dev_n.mean()
        output["emb_dc_dev_ratio"] = e.mean(0).norm() / (dev_n.mean() + 1e-9)
        if e.shape[0] > 2:
            # participation ratio of the centered covariance = (tr C)^2 / ||C||_F^2
            C = (dev.t() @ dev) / e.shape[0]
            tr = C.diagonal().sum()
            output["emb_eff_rank"] = tr * tr / ((C * C).sum() + 1e-12)
    losses_dict = {f"{stage}/{k}": v.detach()
                   for k, v in output.items() if "loss" in k}
    losses_dict.update({f"{stage}/{k}": v.detach()
                        for k, v in output.items() if k.startswith(("quasimetric_", "gram_corr_"))})
    losses_dict[f"{stage}/emb_norm"] = output["emb_norm"]
    losses_dict[f"{stage}/emb_norm_std"] = output["emb_norm_std"]
    # diagnostics matched by name (the dict above is built from "loss" and the qrl_/gram_corr_ prefixes)
    for _k in ("emb_dc_norm", "emb_dev_norm", "emb_dc_dev_ratio", "emb_eff_rank",
               "gram_std_within", "kern_std_within", "kern_std_min_frac",
               "gram_std_cross_pred", "kern_std_cross_pred",
               "spec_mse", "spec_target_rank", "pred_cos", "pred_anchor_cos"):
        if _k in output:
            losses_dict[f"{stage}/{_k}"] = output[_k].detach()
    losses_dict[f"{stage}/batch"] = torch.full(
        (), float(int(getattr(self, "global_step", 0) or 0)), device=emb.device)
    # one collective for the whole dict, not one per metric
    self.log_dict(reduce_metrics_once(losses_dict), on_step=True, sync_dist=False)


def reduce_metrics_once(metrics):
    """Cross-rank mean of a dict of scalar tensors through a single all-reduce (NCCL AVG, what
    Lightning's `sync_dist=True` with the default `reduce_fx="mean"` does per value). Returns the
    dict unchanged outside a process group of more than one rank. The dict must hold the same keys
    in the same order on every rank -- the invariant the per-value sync already required, since
    each of its collectives has to match across ranks too."""
    dist = torch.distributed
    if not (dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1):
        return metrics
    keys = list(metrics)
    if not keys:
        return metrics
    dev = next(v.device for v in metrics.values() if isinstance(v, torch.Tensor) and v.is_cuda) \
        if torch.cuda.is_available() else torch.device("cpu")
    vals = []
    for k in keys:
        v = metrics[k]
        v = v.detach() if isinstance(v, torch.Tensor) else torch.tensor(float(v))
        assert v.numel() == 1, f"{k}: logged metric must be a scalar, got {tuple(v.shape)}"
        vals.append(v.reshape(()).to(dev, torch.float32))
    stacked = torch.stack(vals)
    op = dist.ReduceOp.AVG if dist.get_backend() != "gloo" else dist.ReduceOp.SUM
    dist.all_reduce(stacked, op=op)
    if op == dist.ReduceOp.SUM:
        stacked /= dist.get_world_size()
    return dict(zip(keys, stacked.unbind(0)))


def batches_done(pl_module):
    """Batch count: Lightning's global_step advances once per optimizer step, so with N optimizer
    groups it runs N x the batch count. Schedule gates in a forward count batches."""
    try:
        opts = pl_module.optimizers()
        n = len(opts) if isinstance(opts, (list, tuple)) else 1
    except Exception:
        n = 1
    return int(getattr(pl_module, "global_step", 0) or 0) // max(1, n)


def compute_schedule(*, n_train, world_size, batch_size, max_epochs, warmup_epochs,
                     limit_train_batches=None, steps_per_epoch=None):
    """(steps_per_epoch, total_steps, warmup_steps) in optimizer steps. spt steps the scheduler
    every optimizer step; an int limit_train_batches caps the schedule's epoch length, an explicit
    steps_per_epoch replaces it (a corpus ablation keeping another corpus's schedule)."""
    if steps_per_epoch is None:
        steps_per_epoch = max(1, n_train // world_size // batch_size)
        lim = limit_train_batches
        if isinstance(lim, int) and not isinstance(lim, bool) and lim > 0:
            steps_per_epoch = max(1, min(steps_per_epoch, lim))
    total_steps = steps_per_epoch * int(max_epochs)
    warmup_steps = max(1, min(total_steps - 1, round(float(warmup_epochs) * steps_per_epoch)))
    return steps_per_epoch, total_steps, warmup_steps


def assert_no_action_bottleneck(world_model):
    """Refuse to start a run whose action encoder rank-squeezes the action block. Checked here
    rather than in Embedder so eval can still reload historical checkpoints."""
    pe = getattr(getattr(world_model, "action_encoder", None), "patch_embed", None)
    if pe is None:
        return
    n_in, n_out = int(pe.in_channels), int(pe.out_channels)
    assert n_out >= n_in, (
        f"action encoder bottleneck: patch_embed maps {n_in} -> {n_out}, discarding {n_in - n_out} "
        f"of every action block's {n_in} dimensions. Set model.action_encoder.smoothed_dim to null "
        f"(= input_dim) or >= {n_in}.")


def _assert_all_parameterised_modules_are_optimised(world_model, optimizers):
    """Fail at startup when an attached module owns trainable params no optimizer group selects
    (spt matches groups by regex on qualified module names). Matching mirrors spt's exactly."""
    compiled = [re.compile(g["modules"]) for g in optimizers.values() if g.get("modules")]
    grouped, orphans = {}, {}
    for qual_name, mod in world_model.named_modules():
        if not qual_name or "callbacks_modules" in qual_name or "callbacks_metrics" in qual_name:
            continue
        parent = qual_name.rsplit(".", 1)[0] if "." in qual_name else None
        in_group = grouped.get(parent, False)
        if any(rx.match(qual_name) for rx in compiled):
            in_group = True
        grouped[qual_name] = in_group
        if not in_group:
            n = sum(p.numel() for _, p in mod.named_parameters(recurse=False) if p.requires_grad)
            if n:
                orphans[qual_name.split(".")[0]] = orphans.get(qual_name.split(".")[0], 0) + n
    if orphans:
        listed = ", ".join(f"{k} ({v:,} params)" for k, v in sorted(orphans.items()))
        raise ValueError(
            f"optim_modules={[p.pattern for p in compiled]} selects no optimizer for: {listed}. "
            f"Widen `optim_modules` in the config to name them, e.g. "
            f"optim_modules: \"(model|quasimetric_head)\"."
        )


def _apply_throughput_flags(cfg, world_model):
    """Encoder/predictor speed flags; none of them changes a parameter or a state-dict key."""
    # fused attention must run before any compile: `.fused` is read at trace time
    if cfg.get("fused_predictor_attn", False):
        n_fa = enable_fused_attention(world_model.predictor, True)
        print(f"==> fused predictor attention: enabled on {n_fa} CrossAttnHeads")
    enc_opt = str(cfg.get("encoder_opt", "") or "none").lower()
    assert enc_opt in ("none", "region"), f"encoder_opt={enc_opt!r}; expected none | region"
    if enc_opt == "region":
        prec = str(cfg.trainer.get("precision", "32"))
        assert "bf16" in prec, f"encoder_opt=region needs bf16 precision, got {prec!r}"
        n_rg = enable_region_encoder(world_model.encoder)
        print(f"==> encoder_opt=region: re-associated {n_rg} ViT blocks into one compiled region")
    if cfg.get("compile_predictor", False):
        # compile the bound forward, not the module: torch.compile(module) renames every
        # state-dict key to _orig_mod.*, breaking resume and eval loading
        world_model.predictor.forward = torch.compile(world_model.predictor.forward, dynamic=False)
        print("==> compiled predictor forward (dynamic=False)")


def _build_callbacks(cfg, world_model, dataset, run_id, run_dir, ckpt_dir_pointer, aux_names):
    """Every Lightning callback the recipe asks for: checkpointing, the step/val loss logs, and
    each probe its config gate enables."""
    callbacks = [SaveCkptCallback(
        run_name=f"{cfg.output_model_name}/{run_id}", cfg=cfg.model, epoch_interval=1),
        LightningCkptPointer(pointer_path=ckpt_dir_pointer)]
    if aux_names:
        callbacks.append(AuxStateCallback(
            run_name=f"{cfg.output_model_name}/{run_id}", module_names=aux_names,
            epoch_interval=1))

    # periodic full-state checkpoint (weights, optimizer, LR schedule, step counter), the one a
    # relaunch resumes from; first in the list so the pointer records its dir
    _fc = int(cfg.get("full_ckpt_every_n_steps", 0) or 0)
    if _fc > 0:
        from lightning.pytorch.callbacks import ModelCheckpoint
        callbacks.insert(0, ModelCheckpoint(
            dirpath=str(Path(swm.data.utils.get_cache_dir(), "checkpoints", cfg.output_model_name, run_id, "lightning")),
            filename="{epoch}-{step}", every_n_train_steps=_fc, save_top_k=1,
            save_on_train_epoch_end=True, enable_version_counter=False))

    _ss = cfg.get("step_snapshots", None)
    if _ss:
        from utils import StepSnapshotCallback
        callbacks.append(StepSnapshotCallback(
            out_dir=Path(swm.data.utils.get_cache_dir(), "checkpoints", cfg.output_model_name,
                         run_id, "step_snaps"),
            start=_ss.start, stop=_ss.stop, every=_ss.every, module_names=aux_names,
            step_offset=int(getattr(getattr(world_model, "model", world_model), "_step_offset", 0))))

    callbacks.append(StepLossLogger(run_dir=run_dir, every=50))
    callbacks.append(_WorldSizeGuard(cfg.trainer.get("devices", 1)))
    callbacks.append(_ValMetricLogger(
        run_dir, ["validate/loss", "validate/pred_loss", "validate/pred_cos",
                  "validate/sigreg_loss", "validate/spec_loss", "validate/emb_norm"]))

    probe_cfg = cfg.get("probe", None)
    if probe_cfg and probe_cfg.get("enabled", False):
        from diagnostics.probe_callback import InlineProbeCallback
        callbacks.append(InlineProbeCallback(
            dataset=probe_cfg.get("dataset") or str(dataset.h5_path),
            pixels_key=probe_cfg.get("pixels_key", "pixels"),
            targets=list(probe_cfg.targets),
            run_name=f"{cfg.output_model_name}/{run_id}",
            run_dir=run_dir,
            every_n_epochs=probe_cfg.get("every_n_epochs", 1),
            within_episode=probe_cfg.get("within_episode", False),
            n_frames=probe_cfg.get("n_frames", 1500),
            img_size=cfg.img_size,
            reps=list(probe_cfg.get("reps", ["emb", "cls"])),
            ridge_alpha=probe_cfg.get("ridge_alpha", "cv"),
            n_splits=probe_cfg.get("n_splits", 5),
            mlp=probe_cfg.get("mlp", False),
            excess=probe_cfg.get("excess", False),
            excess_covariate=probe_cfg.get("excess_covariate", None),
            seed=probe_cfg.get("seed", 0),
            batch=probe_cfg.get("batch", 512),
        ))

    suite_cfg = cfg.get("suite_probe", None)
    if suite_cfg and suite_cfg.get("enabled", False):
        from diagnostics.probe_suite_callback import ProbeSuiteCallback
        callbacks.append(ProbeSuiteCallback(
            pack=suite_cfg.pack,
            every_n_steps=suite_cfg.get("every_n_steps", 2000),
            batch=suite_cfg.get("batch", 512),
            log_per_cell=suite_cfg.get("log_per_cell", False),
            dtype=suite_cfg.get("dtype", "fp16"),
        ))
    return callbacks


def run(cfg, forward_fn, build_loss_modules, *, gpu_norm):
    """Drive a training run. `forward_fn(self, batch, stage, cfg)` is the recipe's per-step
    forward; `build_loss_modules(cfg, gpu_norm) -> dict` returns its extra spt.Module attrs.
    `gpu_norm` (always true) says that the forward ImageNet-normalises the uint8 pixels itself."""

    # ---- dataset ----
    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    dataset_name = dataset_cfg.pop("name")
    # `raw_columns` is a normalization instruction, not a loader argument
    dataset_cfg.pop("raw_columns", None)
    cache_dir = os.environ.get("LOCAL_DATASET_DIR", None)

    dataset = RowReadHDF5Dataset(
        dataset_name, transform=None, cache_dir=cache_dir, **dataset_cfg)
    if "calvin" in str(dataset_name):
        # the three block slots of scene_obs must be in canonical red, blue, pink order
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data" / "calvin"))
        from roster import assert_canonical as _assert_roster
        _assert_roster(dataset.h5_path, context="train_common")

    # the loader ships resized uint8 frames; the forward normalises them on the GPU (gpu_aug)
    assert gpu_norm, "every entry point normalises pixels on the GPU"
    transforms = [get_uint8_pixels_pipeline(source='pixels', target='pixels', img_size=cfg.img_size)]

    # `data.dataset.raw_columns`: columns this recipe reads in their own physical units
    _raw_cols = list(cfg.data.dataset.get("raw_columns", []) or [])
    for _c in _raw_cols:
        assert _c in list(cfg.data.dataset.keys_to_load), (
            f"data.dataset.raw_columns names '{_c}', which is not in keys_to_load")
    _norm_cols, _served_raw = classify_columns(cfg.data.dataset.keys_to_load, _raw_cols)
    print(f"[normalize] z-scored: {_norm_cols}  |  served raw: {_served_raw}"
          + (f"  (per-recipe raw_columns: {_raw_cols})" if _raw_cols else ""))

    with open_dict(cfg):
        for col in _norm_cols:
            transforms.append(get_column_normalizer(dataset, col, col))
        cfg.model.action_encoder.input_dim = cfg.data.dataset.frameskip * \
            dataset.get_dim("action")

    dataset.transform = spt.data.transforms.Compose(*transforms)

    # the validation split has its own seed so that a seed sweep (cfg.seed) changes the init and the
    # data order but holds out the same episodes; it defaults to cfg.seed
    split_seed = int(cfg.get("split_seed", cfg.seed))
    rnd_gen = torch.Generator().manual_seed(split_seed)
    if cfg.get("val_split_by_episode", False):
        # whole episodes held out, so a solution keyed on episode identity cannot score on val
        ci = np.asarray(getattr(dataset, "clip_indices"), dtype=np.int64)
        assert ci.ndim == 2 and ci.shape[1] >= 2, (
            f"val_split_by_episode needs an (n_clips, >=2) clip table whose first column is the "
            f"episode; got {ci.shape} from {type(dataset).__name__}")
        assert len(ci) == len(dataset), (
            f"val_split_by_episode: the clip table has {len(ci)} rows but the dataset serves "
            f"{len(dataset)} indices ({type(dataset).__name__})")
        eps = ci[:, 0]
        uniq = np.unique(eps)
        perm = torch.randperm(len(uniq), generator=rnd_gen).numpy()
        n_val_ep = max(1, int(round(len(uniq) * (1 - cfg.train_split))))
        val_eps = set(uniq[perm[:n_val_ep]].tolist())
        is_val = np.fromiter((e in val_eps for e in eps), bool, len(eps))
        train_set = torch.utils.data.Subset(dataset, np.flatnonzero(~is_val).tolist())
        val_set = torch.utils.data.Subset(dataset, np.flatnonzero(is_val).tolist())
        print(f"==> val split by EPISODE: {len(uniq) - n_val_ep} fit / {n_val_ep} val episodes "
              f"({len(train_set)} / {len(val_set)} clips) [seeded perm, seed {split_seed}]")
    else:
        train_set, val_set = spt.data.random_split(
            dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen)
    train_set = PickleableHDF5Wrapper(train_set)
    val_set = PickleableHDF5Wrapper(val_set)

    # forkserver: forked workers COW-break the training heap; forkserver forks from a clean server
    _mp_ctx = "forkserver" if int(dict(cfg.loader).get("num_workers", 0)) > 0 else None
    train = torch.utils.data.DataLoader(
        train_set, **cfg.loader, shuffle=True, drop_last=True, generator=rnd_gen,
        multiprocessing_context=_mp_ctx)
    # the val loop is mostly a probe trigger; val_num_workers=0 drops its persistent workers
    _vnw = int(cfg.get("val_num_workers", dict(cfg.loader).get("num_workers", 0)))
    _val_kw = {**dict(cfg.loader), "num_workers": _vnw}
    if _vnw == 0:
        _val_kw["persistent_workers"] = False
        _val_kw.pop("prefetch_factor", None)
    val = torch.utils.data.DataLoader(val_set, **_val_kw, shuffle=False, drop_last=False)

    # weight init is drawn here, from cfg.seed
    pl.seed_everything(int(cfg.seed), workers=True)

    # ---- model ----
    world_model = hydra.utils.instantiate(cfg.model)
    assert_no_action_bottleneck(world_model)

    # Resume, in order of preference: full state from this run_id's own Lightning checkpoint;
    # +init_weights=<path> (weights only, a new lineage); auto_resume from the furthest
    # weights_epoch_*.pt / step_snaps/step_*.pt of this run_id (weights only).
    _rid = cfg.get("run_id") or ""
    ckpt_dir_pointer = Path(swm.data.utils.get_cache_dir(), "checkpoints",
                            cfg.output_model_name, _rid, "lightning_ckpt_dirs")
    ckpt_path = find_resume_ckpt(
        ckpt_dir_pointer, legacy_glob_root=".",
        run_name=f"{cfg.output_model_name}/{_rid}") \
        if _rid and cfg.get("auto_resume", True) and cfg.get("init_weights", None) is None else None
    _resume_step = 0
    if ckpt_path is not None:
        print(f"==> FULL RESUME from {ckpt_path} "
              f"(optimizer, LR schedule and step counter restored)")
        _m = re.search(r"step=(\d+)", Path(ckpt_path).name)
        _resume_step = int(_m.group(1)) if _m else int(torch.load(ckpt_path, map_location="cpu").get("global_step", 0))

    init_weights = cfg.get("init_weights", None)
    warm_restart = init_weights is not None
    _auto_resumed = False
    _meta_dir = Path(swm.data.utils.get_cache_dir(), "checkpoints", cfg.output_model_name, _rid)
    _meta_dir.mkdir(parents=True, exist_ok=True)
    # the horizon the predictor actually sees: min(max_pred_horizon, N - H) over the multistart
    _N_meta = int(cfg.data.dataset.get("num_steps", 0) or 0)
    _mph_cfg = int(cfg.wm.get("max_pred_horizon", cfg.wm.get("num_preds", 0)) or 0)
    _mph_eff = max(0, min(_mph_cfg, _N_meta - int(cfg.wm.get("history_size", 0) or 0)))
    # tmp + rename: every DDP rank writes the same bytes, and renaming is atomic
    _meta_tmp = _meta_dir / f".train_meta.json.{os.getpid()}.tmp"
    _meta_tmp.write_text(json.dumps({
        "run_id": _rid,
        "dataset": str(cfg.data.dataset.get("name", "")),
        "frameskip": int(cfg.data.dataset.get("frameskip", 1)),
        "num_steps": int(cfg.data.dataset.get("num_steps", 0) or 0),
        "history_size": int(cfg.wm.get("history_size", 0) or 0),
        "num_preds": int(cfg.wm.get("num_preds", 0) or 0),
        "max_pred_horizon": _mph_eff,
        "max_pred_horizon_config": _mph_cfg,
    }, indent=1))
    _meta_tmp.replace(_meta_dir / "train_meta.json")
    if init_weights is None and ckpt_path is None and cfg.get("auto_resume", True):
        _snap_dir = Path(swm.data.utils.get_cache_dir(), "checkpoints", cfg.output_model_name, _rid)
        _snaps = sorted(_snap_dir.glob("weights_epoch_*.pt"),
                        key=lambda p: int(p.stem.split("_")[-1])) if _rid else []
        _steps = sorted((_snap_dir / "step_snaps").glob("step_*.pt"),
                        key=lambda p: int(p.stem.split("_")[-1])) if _rid else []
        # the furthest step snapshot wins; an epoch snapshot only when no step snapshot exists
        _cands = _steps[-1:] or _snaps[-1:]
        if _cands:
            init_weights = str(_cands[0])
            warm_restart = True
            _auto_resumed = True
            print(f"==> AUTO WARM-RESUME from {init_weights} (latest snapshot for run_id={_rid}; "
                  f"optimizer/LR reset, representation preserved). +auto_resume=false to disable.")
    snap_heads = None
    if _resume_step:
        world_model._step_offset = _resume_step
        print(f"==> step offset {_resume_step} (cumulative step axis preserved across the resume)")
    if init_weights:
        sd = torch.load(init_weights, map_location="cpu")
        if isinstance(sd, dict):
            snap_heads = sd.get("heads", None) or None
            if _auto_resumed:
                world_model._step_offset = int(sd.get("step", 0) or 0)
                if world_model._step_offset:
                    print(f"==> step offset {world_model._step_offset} "
                          f"(cumulative step axis preserved across the restart)")
                else:
                    print("==> WARNING: the resumed snapshot carries no step; step-gated names "
                          "and schedules restart at 0", flush=True)
        sd = sd.get("model", sd) if isinstance(sd, dict) else sd
        sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
        missing, unexpected = world_model.load_state_dict(sd, strict=False)
        print(f"==> init_weights: loaded {init_weights} "
              f"(missing={len(missing)}, unexpected={len(unexpected)})")

    _apply_throughput_flags(cfg, world_model)
    # ---- LR schedule + optimizer ----
    world_size = max(1, int(cfg.trainer.get("devices", 1)))
    steps_per_epoch, total_steps, warmup_steps = compute_schedule(
        n_train=len(train_set), world_size=world_size, batch_size=cfg.loader.batch_size,
        max_epochs=cfg.trainer.max_epochs,
        warmup_epochs=float(cfg.get("scheduler", {}).get("warmup_epochs", 1.0)),
        limit_train_batches=cfg.trainer.get("limit_train_batches", None),
        steps_per_epoch=cfg.get("scheduler", {}).get("steps_per_epoch", None))
    print(f"==> LR schedule: {steps_per_epoch} steps/epoch x {cfg.trainer.max_epochs} "
          f"epochs = {total_steps} steps; linear warmup over {warmup_steps} steps "
          f"then cosine anneal to 0.")
    # the schedule is sized from max_epochs; trainer.max_steps stops the run without shortening it
    _ms = int(cfg.trainer.get("max_steps", -1) or -1)
    if 0 < _ms < total_steps:
        import math as _math
        _f = max(_ms - warmup_steps, 0) / max(total_steps - warmup_steps, 1)
        print(f"==> NOTE: trainer.max_steps={_ms} stops the run after {_f:.1%} of the cosine, at "
              f"~{0.5 * (1 + _math.cos(_math.pi * _f)):.2f}x peak LR -- the schedule is sized from "
              f"max_epochs={cfg.trainer.max_epochs}, not from max_steps. Set "
              f"trainer.limit_train_batches={_ms} with max_epochs=1 to anneal inside the budget.")
    if cfg.trainer.get("max_time", None):
        print(f"==> NOTE: trainer.max_time={cfg.trainer.get('max_time')} bounds the run by wall "
              f"clock; the cosine is still {total_steps} steps long.")

    opt_cfg = OmegaConf.to_container(cfg.optimizer, resolve=True)
    # embedding tables are 2-D but not matrices to orthogonalise
    _no_muon = {id(p) for m in world_model.modules() if isinstance(m, torch.nn.Embedding)
                for p in m.parameters()}
    if opt_cfg.get("type") == "MuonAdamW":
        from functools import partial as _partial
        # whitelisted, so an unrecognised key is refused rather than silently dropped
        _known = {"type", "lr", "weight_decay", "momentum", "adjust_lr_fn", "muon_lr", "adamw_lr",
                  "nonmatrix_lr", "betas", "eps", "muon_graph"}
        _unknown = sorted(set(opt_cfg) - _known)
        assert not _unknown, (
            f"optimizer keys {_unknown} are not forwarded to MuonAdamW and would be silently "
            f"ignored; add them to the partial in train_common (and to _known) or remove them")
        opt_factory = _partial(
            MuonAdamW,
            lr=float(opt_cfg.get("lr", 1e-3)),
            weight_decay=float(opt_cfg.get("weight_decay", 0.0)),
            momentum=float(opt_cfg.get("momentum", 0.95)),
            adjust_lr_fn=opt_cfg.get("adjust_lr_fn", "match_rms_adamw"),
            muon_lr=opt_cfg.get("muon_lr", None),
            adamw_lr=opt_cfg.get("adamw_lr", None),
            nonmatrix_lr=opt_cfg.get("nonmatrix_lr", None),
            muon_graph=bool(opt_cfg.get("muon_graph", True)),
            no_muon=_no_muon,
            **({"betas": tuple(opt_cfg["betas"])} if "betas" in opt_cfg else {}),
            **({"eps": float(opt_cfg["eps"])} if "eps" in opt_cfg else {}),
        )
    else:
        # keys meaningful only to the MuonAdamW facade must not reach plain torch optimizers
        opt_factory = {k: v for k, v in dict(cfg.optimizer).items()
                       if k not in ("muon_lr", "adamw_lr", "nonmatrix_lr", "adjust_lr_fn")}

    optimizers = {
        'model_opt': {
            "modules": cfg.get("optim_modules", 'model'),
            "optimizer": opt_factory,
            "scheduler": {
                "type": "LinearWarmupCosineAnnealingLR",
                "warmup_steps": warmup_steps,
                "max_steps": total_steps,
            },
            "interval": "step",
        },
    }

    data_module = spt.data.DataModule(train=train, val=val)
    fwd = lambda *args, **kwargs: forward_fn(cfg=cfg, *args, **kwargs)
    module_kwargs = dict(model=world_model, forward=fwd, optim=optimizers)
    loss_mods = build_loss_modules(cfg, gpu_norm)
    module_kwargs.update(loss_mods)
    world_model = spt.Module(**module_kwargs)
    _assert_all_parameterised_modules_are_optimised(world_model, optimizers)

    # attached loss modules with state are not in weights_epoch_*.pt; their sidecar makes warm
    # restarts full restarts
    aux_names = [k for k, v in loss_mods.items()
                 if isinstance(v, torch.nn.Module)
                 and (any(True for _ in v.parameters()) or any(True for _ in v.buffers()))]
    if snap_heads:
        _restored = []
        for _n, _s in snap_heads.items():
            if hasattr(world_model, _n):
                getattr(world_model, _n).load_state_dict(_s)
                _restored.append(_n)
        print(f"==> step-snap heads: restored {_restored} from {Path(init_weights).name}")
    elif init_weights and Path(init_weights).name.startswith("weights_epoch_"):
        _sc = Path(init_weights).with_name(
            Path(init_weights).name.replace("weights_epoch_", "heads_epoch_"))
        if _sc.exists():
            _hsd = torch.load(_sc, map_location="cpu")
            _restored = []
            for _n, _s in _hsd.items():
                if hasattr(world_model, _n):
                    getattr(world_model, _n).load_state_dict(_s)
                    _restored.append(_n)
            print(f"==> heads sidecar: restored {_restored} from {_sc.name}")
        elif aux_names:
            print(f"==> heads sidecar MISSING next to {Path(init_weights).name}: attached "
                  f"loss modules ({', '.join(aux_names)}) start from init.")

    # ---- training ----
    run_id = cfg.get("run_id") or ""
    run_dir = Path(swm.data.utils.get_cache_dir(), run_id)

    # logger=False (not None) when wandb is off: Lightning's default CSVLogger crashes on
    # dynamic mid-epoch validate/* keys
    logger = False
    if cfg.wandb.enabled:
        import wandb
        wb_cfg = dict(cfg.wandb.config)
        init_timeout = wb_cfg.pop("init_timeout", 300)
        # a warm restart resets the step axis, so it must be a new wandb run grouped under run_id
        is_slurm_requeue = int(os.environ.get("SLURM_RESTART_COUNT", "0") or "0") >= 1
        if warm_restart and not is_slurm_requeue:
            import time
            _base = str(wb_cfg.get("id") or run_id)
            wb_cfg["id"] = f"{_base}-r{int(time.time())}"
            wb_cfg["name"] = _base
            wb_cfg["group"] = _base
            wb_cfg["resume"] = "never"
            for _sidecar in (Path.cwd() / "wandb_resume.json", run_dir / "wandb_resume.json"):
                try:
                    _sidecar.unlink()
                except FileNotFoundError:
                    pass
            print(f"==> WANDB warm-restart: fresh run id '{wb_cfg['id']}' (group={_base})")
        logger = WandbLogger(**wb_cfg, settings=wandb.Settings(init_timeout=float(init_timeout)))
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    callbacks = _build_callbacks(cfg, world_model, dataset, run_id, run_dir,
                                 ckpt_dir_pointer, aux_names)
    trainer_kwargs = dict(cfg.trainer)
    # spt runs manual optimization and never divides the loss by the accumulation count
    _acc = int(trainer_kwargs.get("accumulate_grad_batches", 1) or 1)
    assert _acc == 1, (
        f"trainer.accumulate_grad_batches={_acc}: nothing in this tree divides the loss by it, so "
        f"the run would train at {_acc}x optimizer.lr. Raise loader.batch_size instead.")
    if world_size > 1 and "strategy" not in trainer_kwargs:
        from lightning.pytorch.strategies import DDPStrategy
        _find_unused = bool(cfg.get("ddp_find_unused_parameters", False))
        trainer_kwargs["strategy"] = DDPStrategy(find_unused_parameters=_find_unused)
        print(f"[ddp] DDPStrategy find_unused_parameters={_find_unused}", flush=True)
    trainer = pl.Trainer(
        **trainer_kwargs,
        callbacks=callbacks,
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
    )

    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        ckpt_path=ckpt_path,
    )
    # freeze the static heap before the loop, so the cyclic GC's full collections stay cheap
    import gc
    gc.collect()
    gc.freeze()
    manager()
