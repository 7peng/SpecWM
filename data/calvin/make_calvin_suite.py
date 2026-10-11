"""Generate the CALVIN probe suite: i.i.d. object placements in each CALVIN scene.

Each sample is an INDEPENDENTLY sampled scene configuration, not a frame of a demonstration: the three blocks are re-placed uniformly on
a randomly chosen surface (`initial_pos: any`) and resampled until no pair is in contact
(`PlayTableScene.reset_movable_objects`), and the articulated state -- slider, drawer,
button, switch, lightbulb -- is drawn independently. That is what makes the probe a test of
object recoverability rather than of episode memorisation: consecutive samples share
nothing, so within-episode correlation cannot score.

The arm is FROZEN at the scene's canonical joint pose, so the arm is not a confound and every
frame of a scene differs only in object/articulated state. After placement every block is
raised by an independent height in [Z_LO, Z_HI] above where it rests, as the OGBench suite
does, so the z label varies; frames are written as PNG.

Scenes A/B/C/D are the suite's cells: different table layouts and textures, one fixed camera
each.

Output layout, consumed by `data/probe_suite/build_probe_pack.py` and the probe callback:

    <out>/<split>/S<scene>/frames/{fi:06d}.jpg
    <out>/<split>/S<scene>/labels.json   {"meta": ..., "frames": [...]}

Usage (needs the calvin venv, not the training venv):
    $CALVIN_VENV/bin/python make_calvin_suite.py \
        --scenes A B C --frames 400 --out $CALVIN_ROOT/probe_suite_calvin
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

CALVIN_SRC = Path(os.environ["CALVIN_ENV_SRC"])
sys.path.insert(0, str(CALVIN_SRC))

RES = 200                 # the static camera's native size; kept, not upsampled
USE_EGL = os.environ.get("CALVIN_USE_EGL", "1") == "1"
JPEG_Q = 95               # gripper view only; the probed static view is written lossless
Z_LO, Z_HI = 0.01, 0.09   # hover above the resting pose, the OGBench suite constants
# scene_obs layout (24): slider, drawer, button, switch, lightbulb, green light, then
# 3 movable blocks x (x, y, z, euler_x, euler_y, euler_z).
N_ARTIC, N_OBJ, OBJ_DIM = 6, 3, 6
ARTIC_NAMES = ["slider", "drawer", "button", "switch", "lightbulb", "green_light"]


def build_env(scene, data_path):
    """A rendering env for one CALVIN scene, static camera only, no GUI, no VR."""
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    from omegaconf import open_dict

    # config_data_collection wires env<-scene/robot/cameras with the interpolations intact;
    # composing it whole (rather than assembling the env node by hand) is what keeps
    # ${data_path} and ${robot.euler_obs} resolvable inside the scene config.
    with initialize_config_dir(config_dir=str(CALVIN_SRC / "conf"), version_base=None):
        cfg = compose(config_name="config_data_collection",
                      overrides=[f"scene=calvin_scene_{scene}", "cameras=static_and_gripper",
                                 f"data_path={data_path}", "use_vr=false", "record=false"])
    with open_dict(cfg):
        # EGL, matching the flag the PUBLISHED dataset was generated under
        # (.hydra/config.yaml in the release ships use_egl: true). The CPU
        # TINY_RENDERER shades differently -- measured mean pixel 110 vs the
        # real data's 148 -- which would put suite frames off the training
        # distribution. EGL_DEVICE picks an idle GPU.
        cfg.env.use_egl = USE_EGL
        cfg.env.show_gui = False
        cfg.env.use_scene_info = True
    return instantiate(cfg.env, _recursive_=False)


def sample_articulated(rng, env):
    """Independent articulated state. Ranges come from the scene's own joint limits, so a
    sampled value is always reachable."""
    out = []
    for door in env.scene.doors:
        lo, hi = door.limits if hasattr(door, "limits") else (0.0, 0.25)
        out.append(float(rng.uniform(lo, hi)))
    while len(out) < 2:                                   # slider + drawer
        out.append(0.0)
    out.append(float(rng.uniform(0, 1) < 0.5))            # button
    out.append(float(rng.uniform(0, 1) < 0.5))            # switch
    out += [0.0, 0.0]                                     # lights follow button/switch
    return np.array(out[:N_ARTIC], dtype=np.float64)


def cam_matrices(cam):
    """(view, projection) as 4x4, plus the look_from/look_at the labels record."""
    return (np.asarray(cam.viewMatrix, np.float64).reshape(4, 4, order="F"),
            np.asarray(cam.projectionMatrix, np.float64).reshape(4, 4, order="F"))


def seg_px(cam, uids):
    """Visible pixels per pybullet body id, from the static camera's segmentation buffer.

    This is the visibility gate build_probe_pack.py applies at MIN_PX, the same criterion the
    OGBench suite uses. The low 24 bits of a segmentation pixel are the body id."""
    import pybullet as p
    img = p.getCameraImage(width=cam.width, height=cam.height, viewMatrix=cam.viewMatrix,
                           projectionMatrix=cam.projectionMatrix, physicsClientId=cam.cid)
    seg = np.asarray(img[4]).reshape(cam.height, cam.width) & ((1 << 24) - 1)
    return {u: int((seg == u).sum()) for u in uids}


def world_to_cam(pos_w, view):
    """Object position in CAMERA coordinates, the `in_cam` label of the OGBench suite."""
    p = np.append(np.asarray(pos_w, np.float64), 1.0)
    return (view @ p)[:3]


def render_scene(args):
    rng = np.random.default_rng(args.seed)
    out_root = Path(args.out) / args.split
    for scene in args.scenes:
        kdir = out_root / f"S{scene}"
        if (kdir / "labels.json").exists() and args.skip_existing:
            print(json.dumps(dict(scene=scene, skipped_existing=True)), flush=True)
            continue
        (kdir / "frames").mkdir(parents=True, exist_ok=True)
        env = build_env(scene, str(CALVIN_SRC / "data"))
        cams = {c.name: c for c in env.cameras}
        cam = cams["static"]                      # view 0: the frame labels are expressed in
        view, proj = cam_matrices(cam)
        (kdir / "frames_gripper").mkdir(parents=True, exist_ok=True)
        import cv2

        # EXACT COUNT by rejection: sample a configuration, read the blocks' visibility from the
        # segmentation buffer, and only then render and keep it -- so `--frames` is the number
        # of frames that PASS the gate, not a budget to overshoot. Rejected samples still draw
        # from rng, so the suite is deterministic under the seed; the acceptance rate is
        # recorded in meta as a property of the scene.
        uids = [env.scene.movable_objects[k].uid for k in range(N_OBJ)]
        names = [env.scene.movable_objects[k].name for k in range(N_OBJ)]
        # SINGLE-OBJECT VARIANT. The roster order differs per scene (data/calvin/roster.py), so
        # the block to keep is named, not indexed. The other two are teleported under the floor
        # after the placement is set, which removes them from the image without touching the
        # kept block's sampled pose. It exists to separate "cannot localise a small object" from
        # "cannot say which of three identical shapes is which": with one block in frame there
        # is nothing to bind. A one-block frame is OFF the training distribution -- every arm
        # this probes was trained on three -- so it bounds the binding effect rather than
        # measuring it in situ.
        keep_k = names.index(args.keep) if args.keep else None
        hide = [k for k in range(N_OBJ) if k != keep_k] if args.keep else []
        gate_uids = [uids[keep_k]] if args.keep else uids
        recs, fi, tried = [], 0, 0
        while fi < args.frames:
            tried += 1
            env.scene.reset(None)                      # contact-free block placements
            artic = sample_articulated(rng, env)
            obs_now = env.scene.get_obs()
            scene_obs = np.concatenate([artic, obs_now[N_ARTIC:]])
            env.reset(robot_obs=None, scene_obs=scene_obs)
            import pybullet as _pb
            for k in range(N_OBJ):
                pos, orn = _pb.getBasePositionAndOrientation(uids[k], physicsClientId=cam.cid)
                _pb.resetBasePositionAndOrientation(
                    uids[k], [pos[0], pos[1], pos[2] + rng.uniform(Z_LO, Z_HI)], orn, physicsClientId=cam.cid)
            if hide:
                for k in hide:
                    _pb.resetBasePositionAndOrientation(
                        uids[k], [0.0, 0.0, -5.0], [0, 0, 0, 1], physicsClientId=cam.cid)
            vis = seg_px(cam, gate_uids)
            if min(vis.values()) < args.min_px:
                continue
            obs = env.get_obs()
            rgb = obs["rgb_obs"]["rgb_static"]
            cv2.imwrite(str(kdir / "frames" / f"{fi:06d}.png"), rgb[:, :, ::-1])
            g = obs["rgb_obs"]["rgb_gripper"]
            cv2.imwrite(str(kdir / "frames_gripper" / f"{fi:06d}.jpg"),
                        g[:, :, ::-1], [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_Q])
            so = np.asarray(obs["scene_obs"], np.float64)
            objs = {}
            for j, k in enumerate([keep_k] if args.keep else range(N_OBJ)):
                base = N_ARTIC + k * OBJ_DIM
                w = so[base:base + 3]
                objs[f"object_{j if args.keep else k}"] = dict(
                    category=env.scene.movable_objects[k].name,
                    asset=Path(str(env.scene.movable_objects[k].file)).stem,
                    world=w.tolist(),
                    in_cam=world_to_cam(w, view).tolist(),
                    euler=so[base + 3:base + 6].tolist(),
                    px=vis[env.scene.movable_objects[k].uid],
                )
            recs.append(dict(frame=fi, layout=ord(scene) - 65, style=ord(scene) - 65,
                             cam_pos=list(cam.look_from), cam_look_at=list(cam.look_at),
                             cam_mat=view.reshape(-1).tolist(), fovy=float(cam.fov),
                             artic={n: float(v) for n, v in zip(ARTIC_NAMES, so[:N_ARTIC])},
                             objects=objs))
            fi += 1
        meta = dict(env="calvin", scene=scene, camera="static", H=RES, W=RES,
                    n_sampled=tried, acceptance=fi / tried, min_px=int(args.min_px),
                    n_frames=len(recs), n_objects=1 if args.keep else N_OBJ, window=1,
                    single=args.keep,
                    cameras=["static", "gripper"], jpeg_quality=JPEG_Q, ext="png", hover=[Z_LO, Z_HI],
                    note="independent object placements; arm frozen at canonical pose"
                         + ("; single-object variant, other blocks hidden below the floor"
                            if args.keep else ""))
        (kdir / "labels.json").write_text(json.dumps(dict(meta=meta, frames=recs)))
        print(json.dumps(dict(scene=scene, frames=len(recs))), flush=True)
        # close AND drop the reference before the next scene builds its env. pybullet hands out
        # the same client id after a disconnect, so if this env is still alive when the next one
        # connects, its __del__ fires later and disconnects the NEW env's client -- the next
        # scene then dies on its first reset with "Not connected to physics server".
        env.close()
        del env, cams, cam


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", default=["A", "B", "C", "D"])
    ap.add_argument("--split", default="id")
    ap.add_argument("--frames", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-px", type=int, default=10)
    ap.add_argument("--out", default=os.environ.get("CALVIN_SUITE_DIR",
                                                    os.path.join(os.environ.get("STABLEWM_HOME", os.path.expanduser("~/.stable-wm")), "probe_suite_calvin")))
    ap.add_argument("--keep", default=None,
                    choices=("block_red", "block_blue", "block_pink"),
                    help="render ONE block and hide the other two (identity-free control)")
    ap.add_argument("--skip-existing", action="store_true")
    render_scene(ap.parse_args())
