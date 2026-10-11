"""State-recoverability probe callback: encode a fixed frame subset on validation
epoch end and fit ridge (+ optional MLP) probes to ground-truth targets, logging
held-out R^2 under probe{,_a10,_within,_lin,_mlp,_gap,_excess,_alpha}/."""
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from lightning.pytorch.callbacks import Callback

sys.path.insert(0, str(Path(__file__).resolve().parent))
import probe_state_recoverability as P  # noqa: E402
from probe_heads import grouped_fit  # noqa: E402


def _demean_by_group(A, groups):
    """A minus the mean of its rows sharing a group id."""
    A = np.asarray(A, dtype=np.float64); A = A[:, None] if A.ndim == 1 else A
    g = np.asarray(groups); uniq, inv = np.unique(g, return_inverse=True)
    sums = np.zeros((len(uniq), A.shape[1])); np.add.at(sums, inv, A)
    counts = np.bincount(inv, minlength=len(uniq))[:, None]
    return A - (sums / counts)[inv]


_SUPPORTED_REPS = ("emb", "cls")


def _batch_step(trainer):
    """Batch index for the logged step: global_step divided by the optimizer count, plus the
    step offset of the snapshot a restart resumed from (model._step_offset)."""
    try:
        n_opt = max(1, len(trainer.optimizers))
    except Exception:
        n_opt = 1
    off = int(getattr(getattr(trainer.lightning_module, "model", None), "_step_offset", 0) or 0)
    return int(trainer.global_step) // n_opt + off


class InlineProbeCallback(Callback):
    """Fires on `on_validation_epoch_end` (rank 0 only) every `every_n_epochs` epochs.

    dataset:    h5 path (absolute, or a name resolved under $STABLEWM_HOME); must carry `ep_idx`.
    targets:    "name=col[a:b]" / "name=col[i]" specs (probe_state_recoverability.parse_target).
    reps:       subset of ("emb", "cls").
    ridge_alpha: must be "cv" (alpha per split by grouped inner CV, plus an alpha-10 twin).
    excess:     also log R^2(target | [covariate, rep]) - R^2(target | covariate) per target,
                with `excess_covariate` naming one of the targets.
    within_episode: also log the grouped fit on per-episode-demeaned features and targets.
    mlp:        also fit the batched ridge+MLP head (probe_mlp_torch.r2_ridge_mlp_multi)."""

    def __init__(
        self,
        dataset,
        targets,
        run_name,
        run_dir,
        every_n_epochs=1,
        n_frames=1500,
        pixels_key="pixels",
        img_size=224,
        reps=("emb", "cls"),
        ridge_alpha="cv",
        within_episode=False,
        batch=128,
        seed=0,
        mlp=False,
        mlp_hidden=256,
        mlp_steps=3000,
        mlp_splits=3,
        mlp_max_rows=5000,
        n_splits=5,
        excess=False,
        excess_covariate=None,
    ):
        super().__init__()
        bad = [r for r in reps if r not in _SUPPORTED_REPS]
        if bad:
            raise ValueError(f"inline probe reps {bad} unsupported; use {_SUPPORTED_REPS}")
        if ridge_alpha != "cv":
            raise ValueError(f"ridge_alpha must be 'cv', got {ridge_alpha!r}")
        self.dataset = dataset
        self.target_specs = list(targets)
        self.run_name = run_name
        self.run_dir = Path(run_dir)
        self.every_n_epochs = max(1, int(every_n_epochs))
        self.n_frames = int(n_frames)
        self.pixels_key = pixels_key
        self.img_size = int(img_size)
        self.reps = list(reps)
        self.within_episode = bool(within_episode)
        self.ridge_alpha = "cv"
        self.batch = int(batch)
        self.seed = int(seed)
        self.mlp = bool(mlp)
        self.mlp_hidden = int(mlp_hidden)
        self.mlp_steps = int(mlp_steps)
        self.mlp_splits = int(mlp_splits)
        self.mlp_max_rows = int(mlp_max_rows)
        self.n_splits = int(n_splits)
        self.excess = bool(excess)
        self.excess_covariate = excess_covariate
        assert (not self.excess) or excess_covariate, "excess needs excess_covariate"
        self._frames = None          # (N, C, H, W) float16 CPU, ImageNet-preprocessed
        self._targets = None         # {name: (Y float32, mask bool)}
        self._groups = None          # (N,) episode id per frame
        self._data_path = None

    def _ensure_loaded(self):
        """Sample the fixed frame subset once (seeded) and read frames, targets and episode ids."""
        if self._frames is not None:
            return
        ds = Path(self.dataset).expanduser()
        self._data_path = ds if ds.is_absolute() else P.resolve_dataset(self.dataset)
        tf = P.img_transform(self.img_size)
        targets = [P.parse_target(t) for t in self.target_specs]
        with h5py.File(self._data_path, "r") as f:
            if self.pixels_key not in f:
                raise KeyError(f"probe pixels_key {self.pixels_key!r} not in {self._data_path}; "
                               f"have: {sorted(f.keys())}")
            if "ep_idx" not in f:
                raise KeyError(f"probe dataset {self._data_path} has no ep_idx column")
            n_total = f[self.pixels_key].shape[0]
            n = min(self.n_frames, n_total)
            rng = np.random.default_rng(self.seed)
            rows = np.sort(rng.choice(n_total, size=n, replace=False))
            self._groups = f["ep_idx"][rows].astype(np.int64)
            frames = f[self.pixels_key][rows]
            x = torch.stack([tf(fr) for fr in frames]).half()
            tgt = {}
            for name, col, idx in targets:
                Y = P.extract(f, col, idx, rows).astype(np.float32)
                if np.issubdtype(Y.dtype, np.floating):
                    mask = ~np.isnan(Y).any(axis=1)
                else:
                    mask = np.ones(len(Y), bool)
                tgt[name] = (Y, mask)
        self._frames = x
        self._targets = tgt
        self._target_order = [t[0] for t in targets]
        print(f"[probe] loaded {len(x)} fixed frames from {self._data_path.name} "
              f"({x.element_size() * x.nelement() / 1e6:.0f} MB host); "
              f"targets={self._target_order}", flush=True)

    @torch.no_grad()
    def _encode(self, model, device):
        """{rep: (N, D) float32} for the requested reps: `cls` is the ViT CLS token, `emb` is
        projector(cls)."""
        cls_l, emb_l = [], []
        want = set(self.reps)
        for c in range(0, len(self._frames), self.batch):
            x = self._frames[c:c + self.batch].to(device).float()
            hs = model.encoder(x, interpolate_pos_encoding=True).last_hidden_state
            cls = hs[:, 0]
            if "cls" in want:
                cls_l.append(cls.float().cpu().numpy())
            if "emb" in want:
                emb_l.append(model.projector(cls).float().cpu().numpy())
        out = {}
        if cls_l:   out["cls"] = np.concatenate(cls_l)
        if emb_l:   out["emb"] = np.concatenate(emb_l)
        return out

    def on_validation_epoch_end(self, trainer, pl_module):
        # rank 0 only; metrics go to the loggers directly (self.log would be a collective)
        if not trainer.is_global_zero:
            return
        if (trainer.current_epoch + 1) % self.every_n_epochs != 0:
            return
        self._run_probe(trainer, pl_module, tag=f"epoch {trainer.current_epoch + 1}")

    def _run_probe(self, trainer, pl_module, tag=""):
        self._ensure_loaded()
        was_training = pl_module.model.training
        pl_module.model.eval()
        reps = self._encode(pl_module.model, pl_module.device)
        if was_training:
            pl_module.model.train()

        metrics, per_target = {}, {t: [] for t in self._target_order}
        blocks, alphas = [], {r: [] for r in self.reps}
        for rname in self.reps:
            for tname in self._target_order:
                X = reps[rname]
                Y, mask = self._targets[tname]
                res = grouped_fit(X[mask], Y[mask], self._groups[mask],
                                  ridge_alpha=self.ridge_alpha,
                                  n_splits=self.n_splits, seed0=self.seed)
                r2 = res["r2"]
                metrics[f"probe_a10/{rname}/{tname}"] = res.get("r2_a10", float("nan"))
                if np.isfinite(res.get("alpha", float("nan"))):
                    alphas[rname].append(res["alpha"])
                metrics[f"probe/{rname}/{tname}"] = r2
                per_target[tname].append(r2)
                if self.within_episode:
                    Xw, Yw = _demean_by_group(X[mask], self._groups[mask]), _demean_by_group(Y[mask], self._groups[mask])
                    resw = grouped_fit(Xw, Yw, self._groups[mask], ridge_alpha=self.ridge_alpha,
                                       n_splits=self.n_splits, seed0=self.seed)
                    metrics[f"probe_within/{rname}/{tname}"] = resw["r2"]
                if self.mlp:
                    blocks.append(((rname, tname), X[mask], Y[mask], self._groups[mask]))
        for rname, a in alphas.items():
            if a:
                metrics[f"probe_alpha/{rname}/p50"] = float(np.median(a))

        # excess_r2 shares groups and seeds with the plain fits, so paired fits share splits
        if self.excess:
            Acov, amask = self._targets[self.excess_covariate]
            for tname in self._target_order:
                if tname == self.excess_covariate:
                    continue
                Y, tmask = self._targets[tname]
                m = amask & tmask
                g = self._groups[m]
                r_cov = grouped_fit(Acov[m], Y[m], g, ridge_alpha=self.ridge_alpha,
                                    n_splits=self.n_splits, seed0=self.seed)["r2"]
                for rname in self.reps:
                    both = np.concatenate([Acov[m], reps[rname][m]], axis=1)
                    r_both = grouped_fit(both, Y[m], g, ridge_alpha=self.ridge_alpha,
                                         n_splits=self.n_splits, seed0=self.seed)["r2"]
                    metrics[f"probe_excess/{rname}/{tname}"] = r_both - r_cov
            if "emb" in self.reps:
                eline = "  ".join(f"{t}={metrics[f'probe_excess/emb/{t}']:+.3f}"
                                  for t in self._target_order if t != self.excess_covariate)
                print(f"[probe-excess] {tag} emb beyond-{self.excess_covariate}:  {eline}", flush=True)

        if self.mlp:
            import time as _time
            from diagnostics.probe_mlp_torch import r2_ridge_mlp_multi
            _t0 = _time.time()
            pairs = r2_ridge_mlp_multi(blocks, n_splits=self.mlp_splits, device=pl_module.device,
                                       max_rows=self.mlp_max_rows, h=self.mlp_hidden,
                                       steps=self.mlp_steps)
            metrics["probe_mlp/fit_seconds"] = float(_time.time() - _t0)
            for (rname, tname), (lin, mlp) in pairs.items():
                # probe_lin is this head's own ridge, scored on the same splits as probe_mlp
                metrics[f"probe_lin/{rname}/{tname}"] = lin
                metrics[f"probe_mlp/{rname}/{tname}"] = mlp
                metrics[f"probe_gap/{rname}/{tname}"] = mlp - lin
                if mlp < lin - 0.01:
                    print(f"[probe] !! {rname}/{tname}: mlp {mlp:+.3f} < ridge {lin:+.3f} -- the "
                          f"MLP head underfit (raise mlp_hidden/mlp_steps)", flush=True)
        for tname, vals in per_target.items():
            metrics[f"probe/best/{tname}"] = max(vals)

        # collapse diagnostics on emb: participation ratio of the centered covariance spectrum
        # and the mean off-diagonal cosine over a fixed 2000-row sample
        if "emb" in self.reps:
            try:
                Xe = np.asarray(reps["emb"], dtype=np.float32)
                ev = np.linalg.svd(Xe - Xe.mean(0), full_matrices=False, compute_uv=False) ** 2
                metrics["probe/emb_eff_rank"] = float(ev.sum() ** 2 / ((ev ** 2).sum() + 1e-12))
                Xn = Xe / (np.linalg.norm(Xe, axis=1, keepdims=True) + 1e-8)
                gi = np.random.default_rng(0).choice(len(Xn), size=min(2000, len(Xn)), replace=False)
                G = Xn[gi] @ Xn[gi].T
                m = G.shape[0]
                metrics["probe/emb_mean_cos"] = float((G.sum() - np.trace(G)) / (m * (m - 1)))
                print(f"[probe] {tag} emb "
                      f"eff_rank={metrics['probe/emb_eff_rank']:.1f} "
                      f"mean_cos={metrics['probe/emb_mean_cos']:.3f}", flush=True)
            except Exception as e:  # never let a diagnostic kill training
                print(f"[probe] rank diag failed: {e}", flush=True)

        # every logger: spt's RegistryLogger sits at index 0, so trainer.logger != wandb
        for lgr in (trainer.loggers or []):
            lgr.log_metrics(metrics, step=_batch_step(trainer))
        epoch1 = trainer.current_epoch + 1
        line = "  ".join(f"{t}={metrics[f'probe/best/{t}']:.3f}" for t in self._target_order)
        print(f"[probe] {tag} best-rep R^2:  {line}", flush=True)
        if "emb" in self.reps:
            eline = "  ".join(f"{t}={metrics[f'probe/emb/{t}']:.3f}" for t in self._target_order)
            allpass = all(metrics[f"probe/emb/{t}"] >= 0.90 for t in self._target_order)
            print(f"[probe] {tag} EMB linear R^2:   {eline}"
                  f"{'   <<< ALL>=0.90' if allpass else ''}", flush=True)
        try:
            csv_path = self.run_dir / "probe_emb_log.csv"
            new = not csv_path.exists()
            with open(csv_path, "a") as fh:
                if new:
                    fh.write("epoch,step,rep," + ",".join(self._target_order) + "\n")
                for rname in self.reps:
                    fh.write(f"{epoch1},{_batch_step(trainer)},{rname}," +
                             ",".join(f"{metrics[f'probe/{rname}/{t}']:.4f}"
                                      for t in self._target_order) + "\n")
        except Exception as e:  # never let logging kill training
            print(f"[probe] CSV log failed: {e}", flush=True)

        if self.mlp:
            for rname in self.reps:
                pline = "  ".join(
                    f"{t}={metrics[f'probe_lin/{rname}/{t}']:.3f}/"
                    f"{metrics[f'probe_mlp/{rname}/{t}']:.3f}"
                    f"({metrics[f'probe_gap/{rname}/{t}']:+.3f})" for t in self._target_order)
                print(f"[probe-mlp] {tag} {rname} ridge/mlp(gap):  {pline}"
                      f"{'  [' + format(metrics['probe_mlp/fit_seconds'], '.1f') + 's]' if rname == self.reps[0] else ''}",
                      flush=True)
            try:
                csv_path = self.run_dir / "probe_mlp_log.csv"
                new = not csv_path.exists()
                with open(csv_path, "a") as fh:
                    if new:
                        cols = [f"{t}_{k}" for t in self._target_order for k in ("lin", "mlp", "gap")]
                        fh.write("epoch,step,rep," + ",".join(cols) + "\n")
                    for rname in self.reps:
                        vals = [f"{metrics[f'probe_{k}/{rname}/{t}']:.4f}"
                                for t in self._target_order for k in ("lin", "mlp", "gap")]
                        fh.write(f"{epoch1},{_batch_step(trainer)},{rname}," + ",".join(vals) + "\n")
            except Exception as e:
                print(f"[probe] MLP CSV log failed: {e}", flush=True)
