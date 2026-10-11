"""Fingerprint a training recipe: the composed config plus every scalar its forward produces.

Runs the REAL entry point on the REAL config for a few steps, monkeypatching only the metric
sink, so no part of the objective is simulated or re-implemented. A refactor is behaviour-
preserving when the fingerprints match to within the run-to-run noise floor, which
`--repeat 2` measures on the unchanged tree.

    python tests/regression_baseline.py --recipe tdwm_scene --out results/_regr/before
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

CAPTURE = []


def _patch():
    import torch
    import train_common
    orig = train_common.log_step_metrics

    def patched(self, output, emb, stage):
        row = {"stage": stage}
        for k, v in output.items():
            if torch.is_tensor(v) and v.numel() == 1:
                row[k] = float(v.detach().float())
            elif isinstance(v, (int, float)):
                row[k] = float(v)
        CAPTURE.append(row)
        return orig(self, output, emb, stage)

    train_common.log_step_metrics = patched


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", required=True, help="config name under config/train")
    ap.add_argument("--entry", default="train_spec", choices=["train_spec", "train_lewm"])
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--devices", type=int, default=1)
    ap.add_argument("--out", required=True, help="directory for <name>.json")
    ap.add_argument("--name", default=None, help="capture name (default: the recipe)")
    ap.add_argument("--extra", nargs="*", default=[], help="further hydra overrides")
    ap.add_argument("--rng-shim", action="store_true",
                    help="replay the CPU-RNG draws of the removed zero-weight temporal-regression head "
                         "(two Linear(192, 192) inits) before the loss modules are built, so a quasimetric "
                         "teacher's head is initialised as in the pre-cleanup baseline")
    args = ap.parse_args()

    ov = [
        f"--config-name={args.recipe}",
        # ++ throughout: a recipe that does not already carry the key must still get it
        f"++trainer.limit_train_batches={args.steps}",
        "++trainer.max_epochs=1",
        f"++trainer.devices={args.devices}",
        "++wandb.enabled=false",
        f"++run_id=_regr_{args.recipe}",
        "++auto_resume=false",
        # workers off for a fast, orphan-free start. prefetch_factor and persistent_workers are
        # only legal with workers, so they go too -- train_common does the same for its val loader
        "++loader.num_workers=0",
        "~loader.prefetch_factor",
        "++loader.persistent_workers=false",
        # the probes are separately validated instruments and cost minutes; a fingerprint is
        # about the forward
        "++probe.enabled=false", "++kin_probe.enabled=false",
        "++suite_probe.enabled=false", "++battery.enabled=false", "++pose_probe.enabled=false",
        *args.extra,
    ]
    entry_path = str(ROOT / "src" / f"{args.entry}.py")
    sys.argv = [entry_path, *ov]
    os.chdir(ROOT)

    _patch()
    if args.rng_shim:
        import torch.nn as nn
        import train_common
        _run = train_common.run

        def _run_shimmed(cfg, forward_fn, build_loss_modules, **kw):
            def _build(cfg_, gpu_norm):
                for _ in range(2):
                    nn.Linear(192, 192)
                return build_loss_modules(cfg_, gpu_norm)
            return _run(cfg, forward_fn, _build, **kw)

        train_common.run = _run_shimmed
    status, err = "ok", None
    try:
        # run_name="__main__" so hydra resolves the decorator's relative config_path against the
        # entry point's own directory; importing it resolves against THIS file instead and the
        # search path becomes a nonexistent `config.train` module
        import runpy
        runpy.run_path(entry_path, run_name="__main__")
    except SystemExit as e:
        if int(e.code or 0):
            status, err = "error", f"SystemExit {e.code}"
    except BaseException as e:                      # record, never mask: a recipe that cannot
        status, err = "error", f"{type(e).__name__}: {e}"   # start is a finding, not a gap
    if status == "ok" and not CAPTURE:
        # hydra swallows its own failures into a clean exit, so silence is the failure mode that
        # would otherwise be recorded as a pass
        status, err = "error", "no forward call was made"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.name or args.recipe}.json").write_text(json.dumps(
        {"recipe": args.recipe, "entry": args.entry, "devices": args.devices,
         "status": status, "error": err, "steps": CAPTURE}, indent=1))
    print(f"[regr] {args.recipe}: {status} ({len(CAPTURE)} forward calls)"
          + (f" {err}" if err else ""), flush=True)


if __name__ == "__main__":
    main()
