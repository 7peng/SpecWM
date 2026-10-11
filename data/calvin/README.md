# CALVIN (ABC->D)

CALVIN teleoperated tabletop-manipulation play data (Franka arm, four scene variants).
The ABC_D split trains on scenes A/B/C and validates on scene D. Each timestep is one
`episode_<idx>.npz` with `rgb_static` (200x200x3), `rgb_gripper` (84x84x3), depth maps,
7-DOF absolute + relative actions, robot/scene state, and language annotations; episode
boundaries live in `ep_start_end_ids.npy`.

Data lands under `$CALVIN_ROOT` (root `.env.example`), extracted to
`$CALVIN_ROOT/task_ABC_D/{training,validation}/`.

```bash
data/calvin/download_calvin.sh              # resumable download + sha256 check + extract
python data/calvin/verify_calvin.py         # frame/episode counts per sub-split
```

`SPLIT=D_D` / `SPLIT=ABCD_D` select the other splits. The download log for the staged
copy is `$CALVIN_ROOT/download_ABC_D.log`.

Sizes: `task_ABC_D.zip` is 555,309,812,705 bytes (~517 GiB), 1,894,067 frames extracted;
download + extract needs ~1.1 TB free (the script refuses to start below `NEED_GB`,
default 1100). The zip is deleted after extraction. Checksums come from the official
`http://calvin.cs.uni-freiburg.de/dataset/sha256sum.txt`.

## Extraction is verified, not assumed

Info-ZIP `unzip 6.00` mishandles this ZIP64 archive: it wrote 1,255 of ~2M frames and
exited 0, after which the script stamped `.extracted` and deleted the 517 GiB zip. The
script now extracts with `7z` and refuses to mark the split done or delete the archive
until the frame count clears 100k and `ep_start_end_ids.npy`, `statistics.yaml` and
`scene_info.npy` are all present.

Download uses `aria2c -x16` when available (measured 108 MB/s vs 14.7 MB/s single-stream
on this host) and falls back to `wget -c`. Size verification uses `du --apparent-size`,
because `stat -c%s` reports the sparse extent while parallel range writes are in flight.

## The block roster is per-scene (fixed 2026-09-04)

`scene_obs[6:24]` is three blocks x (xyz, xyz-Euler), and calvin_env writes them in each scene
config's own `movable_objects` order: pink/blue/red in A, red/blue/pink in B and D,
blue/red/pink in C. A reader that calls slot 0 "red" therefore reads pink in scene A and blue
in scene C, which is two thirds of the training corpus.

`roster.py` holds the permutation. `build_calvin_h5.py` applies it at build time and stamps
`scene_obs.attrs["roster"] = "red,blue,pink"`; `train_common.py` and
`experiments/state_kernel/corpora.py` refuse a pack without that stamp. Migrate an older pack
in place -- 172 MB of a 216 GB file, pixels untouched, idempotent:

```bash
python data/calvin/canonicalize_scene_obs.py --h5 $CALVIN_STAGE_DIR/calvin_abc_d.h5 --check
python data/calvin/canonicalize_scene_obs.py --h5 $CALVIN_STAGE_DIR/calvin_abc_d.h5
```

The gate is the language annotations: the block colour a task names must be the block whose xyz
moves most over that window. Simulator order scores 0.571 pooled (scenes A and C at chance
0.34); canonical scores 0.991.

## Probe suite (needs the simulator)

`make_calvin_suite.py` builds the CALVIN object-recoverability suite: independently sampled
scene configurations (blocks re-placed uniformly and
resampled until contact-free, articulated state drawn independently), arm frozen at the
canonical pose, so consecutive samples share nothing and within-episode correlation
cannot score. Output matches the OGBench suite layout, so `build_probe_pack.py` and the
probe callback consume it unchanged.

```bash
$CALVIN_VENV/bin/python data/calvin/make_calvin_suite.py \
    --scenes A B C --split id  --frames 1500 --out $CALVIN_SUITE_DIR
$CALVIN_VENV/bin/python data/calvin/make_calvin_suite.py \
    --scenes D     --split ood --frames 1500 --out $CALVIN_SUITE_DIR
```

Camera and renderer follow the released dataset, not defaults: `cameras=static_and_gripper`
(the two views CALVIN ships) and `use_egl=true`, which the release's own
`.hydra/config.yaml` records. The CPU TINY_RENDERER shades differently -- mean pixel 123
vs the real data's 139, EGL 136 -- which would put suite frames off the training
distribution. `CALVIN_USE_EGL=0` forces the CPU path for hosts without EGL.

One scene per process: PyBullet reuses client 0, so building a second env in the same
process makes the first env's `__del__` disconnect it.
