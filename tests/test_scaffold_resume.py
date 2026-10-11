"""Do step snapshots stay on the run's CUMULATIVE step axis across a resume?

Two resume paths reach train_common: the FULL Lightning resume (preferred, `ckpt_path`) and the
weights-only AUTO WARM-RESUME (the fallback, which passes `step_offset` into
StepSnapshotCallback's constructor). Only the second one carries an offset. This drives a real
2-parameter Lightning run through the full-resume path and reads the names the callback writes.

    .venv/bin/python tests/adv_scaffold_resume.py       (from the repo root)
"""
import sys
import tempfile
import traceback
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

FAIL = []


def check(name, cond, msg=""):
    print(("  PASS " if cond else "  FAIL ") + name + (f"  -- {msg}" if msg else ""))
    if not cond:
        FAIL.append(name)


class _D(torch.utils.data.Dataset):
    def __len__(self):
        return 24
    def __getitem__(self, i):
        return {"x": torch.ones(2), "y": torch.ones(())}


def build(out_dir, tmp, max_epochs, ckpt_path=None):
    import lightning as pl
    import stable_pretraining as spt
    import train_common  # noqa: F401
    from utils import StepSnapshotCallback

    torch.manual_seed(0)
    model = nn.Linear(2, 1, bias=False)

    def fwd(self, batch, stage):
        out = dict(batch)
        out["loss"] = (self.model(batch["x"]).squeeze(-1) - batch["y"]).pow(2).mean()
        return out

    mod = spt.Module(model=model, forward=fwd, optim={
        "model_opt": {"modules": "model", "optimizer": {"type": "SGD", "lr": 0.01},
                      "scheduler": {"type": "ConstantLR"}, "interval": "step"}})
    cb = StepSnapshotCallback(out_dir=out_dir, start=1, stop=1000, every=2)
    ckpt = pl.pytorch.callbacks.ModelCheckpoint(dirpath=str(tmp), save_last=True, every_n_epochs=1)
    tr = pl.Trainer(accelerator="cpu", devices=1, max_epochs=max_epochs, logger=False,
                    enable_progress_bar=False, enable_model_summary=False,
                    num_sanity_val_steps=0, callbacks=[cb, ckpt],
                    default_root_dir=str(tmp))
    dl = torch.utils.data.DataLoader(_D(), batch_size=4, shuffle=False)
    tr.fit(mod, train_dataloaders=dl, ckpt_path=ckpt_path)
    return tr, cb, ckpt


def main():
    tmp = Path(tempfile.mkdtemp(prefix="advresume_"))
    snaps = tmp / "step_snaps"

    tr1, cb1, ck1 = build(snaps, tmp, max_epochs=1)
    leg1 = sorted(int(p.stem.split("_")[-1]) for p in snaps.glob("step_*.pt"))
    print(f"    leg 1: global_step {tr1.global_step}, snapshots {leg1}")
    check("leg 1 numbers its snapshots by batch count", leg1 == [2, 4, 6], f"{leg1}")

    last = tmp / "last.ckpt"
    check("a Lightning checkpoint was written", last.exists(), f"{sorted(p.name for p in tmp.glob('*.ckpt'))}")

    tr2, cb2, _ = build(snaps, tmp, max_epochs=2, ckpt_path=str(last))
    leg2 = sorted(int(p.stem.split("_")[-1]) for p in snaps.glob("step_*.pt"))
    print(f"    leg 2: resumed global_step {tr2.global_step}, "
          f"callback offset {cb2.step_offset}, snapshots now {leg2}")
    check("a FULL Lightning resume keeps step snapshots on the CUMULATIVE axis "
          "(leg 2 must not overwrite leg 1's names)",
          max(leg2) > max(leg1),
          f"leg 2 wrote {leg2}; without an offset it re-writes {leg1} over leg 1's files")
    for p in snaps.glob("step_*.pt"):
        blob = torch.load(p, map_location="cpu", weights_only=False)
        if int(blob["step"]) != int(p.stem.split("_")[-1]):
            check("blob['step'] matches the file name", False, f"{p.name} carries {blob['step']}")
            break
    else:
        check("blob['step'] matches the file name for every snapshot", True)

    print("\n" + ("ALL PASS" if not FAIL else f"FAILURES: {FAIL}"))
    return 1 if FAIL else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(2)
