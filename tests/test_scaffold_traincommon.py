"""train_common's scaffolding: the LR schedule, the normalisation ledger, get_column_normalizer,
the episode-level val split, the logging allowlist, sigreg_forward's slices and GPUPixelNormalize."""
import re
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

FAIL = []


def check(name, cond, msg=""):
    print(("  PASS " if cond else "  FAIL ") + name + (f"  -- {msg}" if msg else ""))
    if not cond:
        FAIL.append(name)


def main():
    from train_common import compute_schedule, log_step_metrics, batches_done
    from utils import classify_columns, get_column_normalizer, ZScoreNormalizer, GPUPixelNormalize

    # ---- 1. LR schedule ----------------------------------------------------
    sp, tot, wu = compute_schedule(n_train=1000, world_size=1, batch_size=10,
                                   max_epochs=1, warmup_epochs=1.0)
    check("max_epochs=1 with common.yaml's warmup_epochs 1.0: warmup IS the whole run",
          wu == tot - 1, f"steps/epoch {sp}, total {tot}, warmup {wu}")
    sp, tot, wu = compute_schedule(n_train=1000, world_size=1, batch_size=10,
                                   max_epochs=1, warmup_epochs=0.05)
    check("warmup_epochs 0.05 on a 1-epoch run leaves 95% of the run annealing",
          wu == 5 and tot == 100, f"total {tot} warmup {wu}")
    sp, tot, wu = compute_schedule(n_train=100000, world_size=4, batch_size=10,
                                   max_epochs=3, warmup_epochs=0.5, limit_train_batches=200)
    check("an int limit_train_batches is the schedule's real epoch length",
          sp == 200 and tot == 600 and wu == 100, f"{sp}/{tot}/{wu}")
    sp, tot, wu = compute_schedule(n_train=100000, world_size=4, batch_size=10,
                                   max_epochs=3, warmup_epochs=0.5, limit_train_batches=1.0)
    check("a FLOAT limit_train_batches (a fraction) is ignored by the schedule",
          sp == 2500, f"steps/epoch {sp} -- a float fraction < 1 would shorten the real epoch "
                      f"without shortening the schedule")
    # max_steps is not an input at all
    import inspect
    check("compute_schedule takes no max_steps: trainer.max_steps cannot size the cosine",
          "max_steps" not in inspect.signature(compute_schedule).parameters)

    # ---- 2. normalization ledger ------------------------------------------
    n, r = classify_columns(["pixels", "action", "qpos", "ep_idx", "step_idx"], [])
    check("classify_columns: ids raw, features z-scored",
          n == ["action", "qpos"] and r == ["ep_idx", "step_idx"], f"{n} | {r}")
    raised = False
    try:
        classify_columns(["action", "not_a_known_column"], [])
    except AssertionError:
        raised = True
    check("an unclassified column REFUSES the run", raised)

    # every supported recipe's keys_to_load must classify
    import yaml
    bad = []
    for f in sorted((ROOT / "config" / "train").glob("*.yaml")):
        try:
            y = yaml.safe_load(f.read_text())
        except Exception:
            continue
        d = ((y or {}).get("data") or {}).get("dataset") or {}
        ks = d.get("keys_to_load")
        if not ks:
            continue
        try:
            classify_columns(ks, d.get("raw_columns", []))
        except AssertionError as e:
            bad.append((f.name, str(e).split("\n")[0]))
    known = set()
    for name, msg in bad:
        col = re.search(r"names '(\w+)'", msg)
        if (name, col.group(1) if col else None) in known:
            print(f"    KNOWN DEFECT {name}: {msg}")
    unknown = [(a, b) for a, b in bad
               if (a, (re.search(r"names '(\w+)'", b) or [None, None])[1]) not in known]
    check("every config/train/*.yaml that names keys_to_load classifies all of them (known defects listed)",
          not unknown, "; ".join(f"{a}: {b}" for a, b in unknown[:5]))

    # ---- 3. get_column_normalizer: population and constant columns ---------
    class _DS:
        def __init__(self, a):
            self.a = a
        def get_col_data(self, c):
            return self.a
    a = np.zeros((100, 3), np.float32)
    a[:, 0] = np.arange(100)
    a[:, 1] = 7.0                                   # constant column
    a[50, 2] = np.nan                               # NaN row (whole row dropped)
    wrapped = get_column_normalizer(_DS(a), "action", "action", tile=1)
    got = wrapped({"action": torch.tensor([[0.0, 8.0, 0.0]])})["action"]
    # reference stats, computed the way the function documents
    d = torch.from_numpy(a)
    d = d[~torch.isnan(d).any(dim=1)]
    mu, sd = d.mean(0), d.std(0).clamp_min(1e-6)
    check("get_column_normalizer drops the NaN ROW and clamps a constant column to std 1e-6",
          abs(float(sd[1]) - 1e-6) < 1e-12 and not bool(torch.isnan(mu).any()),
          f"mean {mu.tolist()} std {sd.tolist()}")
    ref = (torch.tensor([[0.0, 8.0, 0.0]]) - mu) / sd
    check("the wrapped transform reproduces those stats exactly",
          torch.allclose(got, ref, atol=0, rtol=0), f"{got.tolist()} vs {ref.tolist()}")
    check("a constant column becomes a 1e6-gain channel once anything moves off its train value",
          abs(float(got[0, 1])) > 1e5,
          f"raw +1.0 above the train constant becomes z={float(got[0,1]):.3g}; "
          f"eval's sklearn StandardScaler uses scale_=1.0 for the same column")
    # tile: lerobot stacks `frameskip` action rows into one block, so the stats must tile
    wt = get_column_normalizer(_DS(a), "action", "action", tile=2)
    g2 = wt({"action": torch.tensor([[0.0, 8.0, 0.0, 0.0, 8.0, 0.0]])})["action"]
    check("tile=frameskip repeats the per-column stats across the stacked action block",
          torch.allclose(g2[0, :3], g2[0, 3:], atol=0, rtol=0), f"{g2.tolist()}")

    # ---- 4. episode split reads the clip table -----------------------------
    src_tc = (ROOT / "src" / "train_common.py").read_text()
    blk = src_tc[src_tc.index("if cfg.get(\"val_split_by_episode\""):]
    blk = blk[:blk.index("train_set, val_set = spt.data.random_split")]
    check("val_split_by_episode no longer reshapes the clip table to a fixed width",
          ".reshape(-1, 3)" not in blk)
    for width, tag in ((3, "lerobot (N,3)"), (2, "hdf5 (N,2)")):
        ci_list = [tuple([e] + [0] * (width - 1)) for e in range(9)]
        # the OLD expression, kept so the regression is visible
        try:
            old = np.unique(np.asarray(ci_list, dtype=np.int64).reshape(-1, 3)[:, 0]).tolist()
        except ValueError as e:
            old = f"ValueError: {e}"
        ci = np.asarray(ci_list, dtype=np.int64)           # the NEW expression
        ok = ci.ndim == 2 and ci.shape[1] >= 2 and np.array_equal(np.unique(ci[:, 0]), np.arange(9))
        check(f"the width-agnostic split recovers all 9 episodes on the {tag} clip table",
              ok, f"reshape(-1,3) would have given {old}")

    check("train_common's val_split_by_episode now asserts table rows == served indices",
          "len(ci) == len(dataset)" in blk)

    # ---- 5. logging allowlist ---------------------------------------------
    src = (ROOT / "src" / "train_common.py").read_text()
    allow = set(re.findall(r'"([a-z0-9_]+)"', src[src.index("for _k in ("):src.index("):\n        if _k in output")]))
    prefixes = ("quasimetric_", "gram_corr_")
    explicit = allow | {"pred_cos", "pred_cos_head", "pred_anchor_cos", "emb_norm", "emb_norm_std"}

    def covered(k):
        return ("loss" in k) or k.startswith(prefixes) or k in explicit

    emitted = {}
    for name in ("train_lewm.py", "train_spec.py"):
        p = ROOT / "src" / name
        if not p.exists():
            continue
        txt = p.read_text()
        ks = set(re.findall(r'output\[\s*"([a-zA-Z0-9_]+)"\s*\]\s*=', txt))
        ks |= set(re.findall(r'out\[\s*"([a-zA-Z0-9_]+)"\s*\]\s*=', txt))
        emitted[name] = ks
    dropped = {n: sorted(k for k in ks if not covered(k) and k not in
                         ("emb", "act_emb", "trunk", "tokens", "pixels", "action"))
               for n, ks in emitted.items()}
    n_drop = sum(len(v) for v in dropped.values())
    print(f"    [allowlist] emitted-but-dropped monitor keys: "
          + "; ".join(f"{n}: {v}" for n, v in dropped.items() if v))
    check("no monitor key written by a forward is dropped at the log boundary",
          n_drop == 0, f"{n_drop} dropped")

    # ---- 6. sigreg_forward's slices ---------------------------------------
    import train_lewm
    from omegaconf import OmegaConf

    class _Model(nn.Module):
        """encode() returns a LABELLED emb: emb[b,t,0] = t. predict() returns its context
        unchanged, so a wrong target slice shows up as a nonzero pred_loss."""
        def __init__(self, D=4):
            super().__init__()
            self.D = D
            self.seen = {}
        def encode(self, info):
            B, T = info["pixels"].shape[:2]
            e = torch.zeros(B, T, self.D)
            e[..., 0] = torch.arange(T, dtype=torch.float32)
            info["emb"] = e
            info["act_emb"] = torch.arange(T, dtype=torch.float32)[None, :, None].expand(B, T, self.D).clone()
            return info
        def predict(self, ctx, act):
            self.seen["ctx"] = ctx.clone()
            self.seen["act"] = act.clone()
            out = ctx.clone()
            out[..., 0] += 1.0                   # a perfect next-step predictor on this labelling
            return out

    class _Self:
        def __init__(self):
            self.model = _Model()
            self.sigreg = lambda x: torch.zeros(())
            self.gpu_aug = lambda px: (px, None)
            self.global_step = 0
            self.logged = {}
        def log_dict(self, d, **kw):
            self.logged.update(d)

    cfg = OmegaConf.create({"wm": {"history_size": 3, "num_preds": 1},
                            "loss": {"sigreg": {"weight": 0.0}, "pred_weight": 1.0}})
    s = _Self()
    N = 6
    batch = {"pixels": torch.zeros(2, N, 3, 4, 4), "action": torch.zeros(2, N, 3)}
    out = train_lewm.sigreg_forward(s, batch, "fit", cfg)
    check("sigreg_forward: ctx = emb[:, :history_size], target = emb[:, num_preds:H+num_preds]",
          float(out["pred_loss"]) == 0.0,
          f"pred_loss {float(out['pred_loss']):.4g} (0 means target == ctx+1, i.e. one step ahead)")
    check("sigreg_forward: the predictor's context is the causal PREFIX 0..H-1",
          torch.equal(s.model.seen["ctx"][0, :, 0], torch.tensor([0.0, 1.0, 2.0])),
          f"{s.model.seen['ctx'][0, :, 0].tolist()}")
    check("sigreg_forward: the conditioning actions are the SAME frames as the context",
          torch.equal(s.model.seen["act"][0, :, 0], torch.tensor([0.0, 1.0, 2.0])),
          f"{s.model.seen['act'][0, :, 0].tolist()}")
    # history_size reaching the last frame (whose action block is zero-padded) must be refused
    cfg2 = OmegaConf.create({"wm": {"history_size": N, "num_preds": 1},
                             "loss": {"sigreg": {"weight": 0.0}, "pred_weight": 1.0}})
    raised = False
    try:
        train_lewm.sigreg_forward(_Self(), {"pixels": torch.zeros(2, N, 3, 4, 4),
                                              "action": torch.zeros(2, N, 3)}, "fit", cfg2)
    except AssertionError:
        raised = True
    check("sigreg_forward refuses a history that reaches the zero-padded final action block",
          raised)

    # the logged keys of that step
    print(f"    [sigreg logged] {sorted(k.split('/')[-1] for k in s.logged)}")

    # ---- 7. GPUPixelNormalize ---------------------------------------------
    g = GPUPixelNormalize()
    u8 = torch.randint(0, 256, (1, 2, 3, 4, 4), dtype=torch.uint8)
    y, _ = g(u8)
    from stable_pretraining import data as dt
    mean = torch.tensor(dt.dataset_stats.ImageNet["mean"]).view(1, 1, 3, 1, 1)
    std = torch.tensor(dt.dataset_stats.ImageNet["std"]).view(1, 1, 3, 1, 1)
    ref = (u8.float() / 255.0 - mean) / std
    check("GPUPixelNormalize == (u8/255 - ImageNet mean)/std, channel-first",
          torch.allclose(y, ref, atol=1e-6), f"max |d| {float((y-ref).abs().max()):.2e}")
    raised = False
    try:
        g(u8.float())
    except AssertionError:
        raised = True
    check("GPUPixelNormalize refuses already-normalized float pixels (the double-normalize trap)",
          raised)

    print("\n" + ("ALL PASS" if not FAIL else f"FAILURES: {FAIL}"))
    return 1 if FAIL else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(2)
