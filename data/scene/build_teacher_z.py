"""Tabulate a frozen QRL teacher's z coordinates over a scene HDF5 dataset.

Scene pixels live in one HDF5 column, so this reads them directly.

    python data/scene/build_teacher_z.py --config <run>/.hydra/config.yaml \
        --ckpt lewm/scene-qrlfaithful-1/step_snaps/step_2000.pt --out $SCENE_STAGE_DIR/teacher_z.npy

Sharding is by frame range across GPUs: shard 0 CREATES the memmap and the rest --resume into
it, because four processes racing to create one .npy each truncate the others' writes. The array is (n_frames, z_dim) fp16; at 1.3M frames x 384 that is
1.0 GB, small enough to stage node-local and mmap.

WHY POST-HEAD z AND NOT emb. The consumer (`_factor_block_kernel`, form=qrl_z) calls
`all_pairs_dist_from_z`, so the head must already be applied; storing emb would leave the head free
to drift between build and training, and the distance would no longer be the teacher's own.

READS ARE SEQUENTIAL AND THAT IS LOAD-BEARING. `pixels` is a blosc-compressed HDF5 column of
224x224x3 uint8; a shuffled gather decompresses one chunk per row. Each shard walks its own
contiguous range, so every chunk is touched once.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from utils import get_uint8_pixels_pipeline, load_env  # noqa: E402

load_env()

import h5py            # noqa: E402
import hdf5plugin      # noqa: E402  -- registers the blosc filter the pixels column needs
import numpy as np     # noqa: E402
import torch           # noqa: E402
from hydra.utils import instantiate  # noqa: E402
from omegaconf import OmegaConf      # noqa: E402


def load_teacher(config_path, ckpt_path, device="cuda"):
    """(model, quasimetric head) rebuilt from the run's own hydra config, weights strict-loaded.

    strict=True throughout: a head built with the wrong z_dim/group_size, or an encoder whose
    architecture drifted from the config, must fail here rather than produce coordinates that are
    not the teacher's.
    """
    from losses import QRLHead

    cfg = OmegaConf.load(config_path)
    blob = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    msd = blob.get("model", None)
    assert msd is not None, f"{ckpt_path} carries no `model` state dict"
    # action_encoder.input_dim is `???` in a saved config: train_common fills it at runtime from
    # the dataset (frameskip x action width), so the config alone cannot rebuild the model. The
    # CHECKPOINT is the authority -- the first embed matrix's fan-in is that width by construction,
    # and taking it from there makes the rebuilt module match the weights it is about to load
    # instead of matching whatever dataset happens to be mounted.
    if OmegaConf.is_missing(cfg.model.action_encoder, "input_dim"):
        w = msd.get("action_encoder.embed.0.weight", None)
        assert w is not None, ("config leaves action_encoder.input_dim unset and the checkpoint "
                               "has no action_encoder.embed.0.weight to infer it from")
        cfg.model.action_encoder.input_dim = int(w.shape[1])
    model = instantiate(cfg.model, _convert_="all")
    model.load_state_dict(msd, strict=True)
    model = model.to(device).eval().requires_grad_(False)

    heads = blob.get("heads", {}) or {}
    hsd = heads.get("quasimetric_head", heads.get("qrl_head", None))       # qrl_head: teachers trained before the rename
    assert hsd, f"{ckpt_path} carries no quasimetric head; a teacher without its head defines no kernel"
    hkw = OmegaConf.to_container((cfg.get("quasimetric_head") or cfg.qrl_kernel).head, resolve=True)
    hkw.setdefault("window_size", 1)
    head = QRLHead(**hkw)
    head.load_state_dict(hsd, strict=True)
    head = head.to(device).eval().requires_grad_(False)
    print(f"[teacher] {Path(ckpt_path).name} @ step {blob.get('step', '?')}: "
          f"z_dim {hkw.get('z_dim')} group_size {hkw.get('group_size')}", flush=True)
    return model, head


@torch.no_grad()
def encode_range(model, head, px_u8, gpu_norm, tf_u8, bf16=True):
    """(B,H,W,C) uint8 numpy -> (B, z_dim) float32 on device.

    THE LOADER'S OWN TRANSFORM, not a reimplementation: `tf_u8` is the very
    `get_uint8_pixels_pipeline(img_size=...)` train_common builds, so the resize here cannot drift
    from the one training applies. It is not cosmetic -- CALVIN ships 200x200 frames against a
    224 model input, and encoding them unresized would tabulate z for an image the encoder never
    sees. (On a corpus already at img_size the resize is an identity.)

    Then the uint8 -> GPU-normalize path, matching `gpu_normalize_pixels` in training: the CPU
    float preprocessor resizes AFTER converting to float while this one stays in uint8, and the two
    differ at ~1e-2 relative. Training uses this one, so the cache must be built with it.
    """
    # the loader's order (utils.RowReadHDF5Dataset._load_slice): permute HWC -> CHW as a tensor,
    # then the transform. ToImage on a NUMPY (T,H,W,C) array returns (T,C,W,H) -- H and W swapped.
    px = torch.from_numpy(np.ascontiguousarray(px_u8)).permute(0, 3, 1, 2)
    px = tf_u8({"pixels": px})["pixels"]
    px = px.cuda(non_blocking=True).unsqueeze(1).contiguous()   # (B, T=1, C, H, W) uint8
    px, _ = gpu_norm(px)                                        # -> ImageNet-normalized float
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
        emb = model.encode({"pixels": px}, window_size=1)["emb"][:, -1]
        return head.encode_z(emb).float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="the run's .hydra/config.yaml")
    ap.add_argument("--ckpt", required=True,
                    help="step snapshot, relative to $STABLEWM_HOME/checkpoints or absolute")
    ap.add_argument("--h5", default=None,
                    help="scene dataset (default: $SCENE_STAGE_DIR/scene_play_explore.h5)")
    ap.add_argument("--out", required=True, help="output .npy (fp16, (n_frames, z_dim))")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=0, help="0 = to the last frame")
    ap.add_argument("--resume", action="store_true", help="write into an existing .npy")
    ap.add_argument("--verify", type=int, default=256, help="rows to re-encode and compare")
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--img-size", type=int, default=224,
                    help="the model's input size; frames are resized to it by the LOADER's own "
                         "pipeline (CALVIN ships 200x200 against a 224 model)")
    ap.add_argument("--no-bf16", action="store_true")
    args = ap.parse_args()

    from utils import GPUPixelNormalize

    root = os.environ["STABLEWM_HOME"]
    ckpt = args.ckpt if os.path.isabs(args.ckpt) else os.path.join(root, "checkpoints", args.ckpt)
    h5p = args.h5 or os.path.join(os.environ["SCENE_STAGE_DIR"],
                                  "scene_play_explore.h5")
    model, head = load_teacher(args.config, ckpt)
    gpu_norm = GPUPixelNormalize().cuda()
    tf_u8 = get_uint8_pixels_pipeline(source="pixels", target="pixels", img_size=args.img_size)

    with h5py.File(h5p, "r", swmr=True) as f:
        n, zdim = int(f["pixels"].shape[0]), int(head.z_dim if hasattr(head, "z_dim") else 0)
        if not zdim:                       # probe it, rather than trusting an attribute name
            zdim = int(encode_range(model, head, f["pixels"][0:1], gpu_norm, tf_u8,
                                    not args.no_bf16).shape[-1])
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        if args.resume or args.verify_only:
            assert os.path.exists(args.out), f"--resume/--verify-only but no {args.out}"
            arr = np.lib.format.open_memmap(args.out, mode="r+")
            assert arr.shape == (n, zdim), f"existing store is {arr.shape}, this build wants {(n, zdim)}"
        else:
            print(f"[qmz] {n} frames x {zdim}d fp16 = {n * zdim * 2 / 1e9:.2f} GB -> {args.out}",
                  flush=True)
            arr = np.lib.format.open_memmap(args.out, mode="w+", dtype=np.float16, shape=(n, zdim))

        lo, hi = max(0, args.start), min(n, args.end or n)
        if not args.verify_only:
            print(f"[qmz] shard [{lo}, {hi}) of {n}", flush=True)
            t0 = time.time()
            for a in range(lo, hi, args.batch):
                b = min(a + args.batch, hi)
                z = encode_range(model, head, f["pixels"][a:b], gpu_norm, tf_u8, not args.no_bf16)
                arr[a:b] = z.cpu().numpy().astype(np.float16)
                if ((a - lo) // args.batch) % 50 == 0 and a > lo:
                    rate = (a - lo) / max(time.time() - t0, 1e-9)
                    print(f"[qmz] {a}/{hi}  {rate:.0f} frames/s  "
                          f"eta {(hi - a) / max(rate, 1) / 60:.1f} min", flush=True)
            arr.flush()
            print(f"[qmz] shard [{lo}, {hi}) done in {(time.time() - t0) / 60:.1f} min", flush=True)

        if (lo, hi) == (0, n) or args.verify_only:
            meta = {"ckpt": args.ckpt, "config": str(args.config), "h5": h5p,
                    "n_frames": n, "z_dim": zdim, "dtype": "float16",
                    "autocast_bf16": not args.no_bf16, "readout": "qrl_z"}
            with open(f"{os.path.splitext(args.out)[0]}.meta.json", "w") as m:
                json.dump(meta, m, indent=2)
            print("[qmz] wrote meta.json", flush=True)
        else:
            print("[qmz] shard build: run --verify-only when every shard is done, to write "
                  "meta.json and check the whole array", flush=True)

        if args.verify:
            # fp16 storage error, a stale checkpoint, the wrong readout and index misalignment are
            # what this catches; it re-encodes through the SAME uint8 path, because comparing
            # against the float-CPU preprocessor fails at ~1e-2 for a reason that is not a bug here.
            rng = np.random.default_rng(0)
            # SAMPLE THE RANGE THIS PROCESS WROTE, not the whole array. A shard build leaves the
            # rest of the memmap at zero, and drawing globally compares live rows against those
            # zeros -- relative 1.0, a failure that says nothing about the shard. --verify-only
            # runs after every shard, and is the pass that checks the array end to end.
            _v0, _v1 = (0, n) if args.verify_only else (lo, hi)
            rows = sorted(int(x) for x in rng.choice(np.arange(_v0, _v1),
                                                     size=min(args.verify, _v1 - _v0),
                                                     replace=False))
            live = np.concatenate([
                encode_range(model, head, f["pixels"][r:r + 1], gpu_norm, tf_u8,
                             not args.no_bf16).cpu().numpy()
                for r in rows]).astype(np.float32)
            cached = np.asarray(arr[rows], dtype=np.float32)
            d = np.abs(live - cached)
            rel = d.max() / max(np.abs(live).max(), 1e-9)
            print(f"[qmz] verify {len(rows)} rows: max|delta| {d.max():.3e}  relative {rel:.3e}  "
                  f"mean {d.mean():.3e}", flush=True)
            assert rel < 1e-2, (
                f"cached z disagrees with a live forward (relative {rel:.3e}). The store was built "
                f"under a different precision or transform than the consumer will use; training on "
                f"it would feed the kernel distances the teacher does not produce.")
            print("[qmz] verified against a live forward", flush=True)


if __name__ == "__main__":
    main()
