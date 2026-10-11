"""Generate the OGBench probe suite on OGBScene and OGBCube.

One cell is one env.reset(): it fixes the arm pose and (on scene) the drawer, window and
buttons, then every frame teleports the cube to an independently sampled valid position and
yaw. Placement validity per frame: inside the camera frustum, not interpenetrating another
body, unoccluded along the camera ray, and at least MIN_PX visible pixels. Because cube
position is drawn independently of arm pose, camera pose and cell, none of those carries any
information about it by construction.

The frustum, occlusion and interpenetration tests and their constants (MIN_PX, Z_LO/Z_HI,
PEN, FRUSTUM_MARGIN) are imported from `data/probe_suite/suite_geometry.py`. The env layer is
OGBench's: cell = one placement seed under the default scene, one movable cube
(`object_joint_0`, qpos[14:17]), teleported by a qpos write + mj_forward on
`env.unwrapped._model / ._data`.

Output, consumed by `data/probe_suite/build_probe_pack.py`:

    <out>/<split>/<cell>/frames/{fi:06d}.png
    <out>/<split>/<cell>/labels.json    {"meta": ..., "frames": [...]}

    python data/ogbench/make_ogbench_suite.py --env scene --split id  --cells 12 --frames 2667
    python data/ogbench/make_ogbench_suite.py --env scene --split ood --cells 3  --frames 2667

Cells differ only in their placement RNG, so several cells are independent repeats of the same
measurement (error bars), not a generalisation test: nothing about the scene is held out, only
cube positions are unseen.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

# validity tests on raw mujoco model/data, shared with the pack builder's constants
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "probe_suite"))
from suite_geometry import (  # noqa: E402
    MIN_PX, Z_HI, Z_LO,
    camera_blocked, in_frustum, obj_in_cam, penetrating, unoccluded,
)

ENVS = {"scene": "swm/OGBScene-v0", "cube": "swm/OGBCube-v0"}
RES_OGB = 224                 # the training corpus resolution; rendered directly, never resampled
OFFSAMPLES = 4                # MSAA, as the corpus builders (data/scene, data/cube) set it
CAM = "front_pixels"          # the camera the training renders use; labels are in ITS frame
CUBE_QADR = 14                # object_joint_0 (FREE): qpos[14:17] xyz, qpos[17:21] quat


def cam_frame_ogb(u, cam=CAM):
    """(pos, mat, fovy) of the suite camera from the env's model/data."""
    m, d = u._model, u._data
    cid = m.cam(cam).id
    return d.cam_xpos[cid].copy(), d.cam_xmat[cid].reshape(3, 3).copy(), float(m.cam_fovy[cid])


def cube_ids(u):
    """(body id, geom ids) of the movable cube, for the occlusion and penetration checks."""
    import mujoco
    m = u._model
    jid = m.joint("object_joint_0").id
    bid = int(m.jnt_bodyid[jid])
    gids = {int(g) for g in range(m.ngeom) if int(m.geom_bodyid[g]) == bid}
    return bid, gids


def visible_px(seg_r, d, gids, cam=CAM):
    """Visible pixel count of the cube from the segmentation render: recorded as a covariate
    and used as the occlusion floor at MIN_PX."""
    seg_r.update_scene(d, camera=cam)
    seg = seg_r.render()                       # (H, W, 2): [geom id, object type]
    ids = seg[..., 0]
    return int(np.isin(ids, list(gids)).sum())


def run_cell(env_id, cell, n_frames, cdir, rng):
    import cv2
    import mujoco
    import stable_worldmodel  # noqa: F401  (registers swm/* ids)

    # The env exactly as the corpus builders construct it: visualize_info off (no goal marker or
    # success colouring reaches the pixels), the corpus resolution, MSAA 4. Default camera, colours
    # and lighting otherwise; the only thing that varies is where the cube is.
    if "Scene" in env_id:
        from stable_worldmodel.envs.ogbench.scene_env import SceneEnv
        env = SceneEnv(ob_type="states", width=RES_OGB, height=RES_OGB, visualize_info=False)
    else:
        from stable_worldmodel.envs.ogbench.cube_env import CubeEnv
        env = CubeEnv(env_type="single", ob_type="states", width=RES_OGB, height=RES_OGB, visualize_info=False)
    u = env.unwrapped
    env.reset(seed=int(cell))
    u.model.vis.quality.offsamples = OFFSAMPLES
    if u._renderer is not None:
        u._renderer.close(); u._renderer = None
    m, d = u._model, u._data
    bid, gids = cube_ids(u)
    seg_r = mujoco.Renderer(m, height=RES_OGB, width=RES_OGB)
    seg_r.enable_segmentation_rendering()
    cpos, cmat, fovy = cam_frame_ogb(u)
    assert not camera_blocked(m, d, cpos, cmat), f"cell {cell}: camera sees nothing"

    # frozen for the whole cell: arm pose, and on scene the drawer/window/button state
    q0 = d.qpos.copy()
    lo, hi = np.asarray(u._object_sampling_bounds, float)
    surf = float(q0[CUBE_QADR + 2])          # the cube's resting height defines the surface

    (cdir / "frames").mkdir(parents=True, exist_ok=True)
    recs, fi, tries = [], 0, 0
    t0 = time.time()
    while len(recs) < n_frames:
        tries += 1
        if tries > 60 * n_frames:
            raise RuntimeError(f"cell {cell}: placement infeasible ({len(recs)}/{n_frames})")
        p = np.array([rng.uniform(lo[0], hi[0]), rng.uniform(lo[1], hi[1]),
                      surf + rng.uniform(Z_LO, Z_HI)])
        if not in_frustum(p, cpos, cmat, fovy)[0]:
            continue
        # teleport through the sim; a full reset would re-run the env's own state machinery
        yaw = rng.uniform(-np.pi, np.pi)
        d.qpos[CUBE_QADR:CUBE_QADR + 3] = p
        d.qpos[CUBE_QADR + 3:CUBE_QADR + 7] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)
        if penetrating(d, gids) or not unoccluded(m, d, bid, gids, cpos):
            continue
        px = visible_px(seg_r, d, gids)
        if px < MIN_PX:
            continue
        rgb = np.asarray(u.render(camera=CAM), np.uint8)
        cv2.imwrite(str(cdir / "frames" / f"{fi:06d}.png"), rgb[:, :, ::-1])
        wp = d.xpos[bid].copy()
        recs.append(dict(frame=fi, cell=int(cell), cam_pos=cpos.tolist(),
                         cam_mat=cmat.reshape(-1).tolist(), fovy=fovy,
                         arm_qpos=q0[:14].tolist(), yaw=float(yaw),
                         objects={"object_0": dict(category="cube", asset="cube", slot=0,
                                                   world=wp.tolist(),
                                                   in_cam=obj_in_cam(wp, cpos, cmat)[0].tolist(),
                                                   px=int(px))}))
        fi += 1
    dt = time.time() - t0
    meta = dict(cell=int(cell), env=env_id, mode="object", n_frames=len(recs), n_objects=1,
                tries=tries, seconds=round(dt, 1), feasibility=round(len(recs) / max(tries, 1), 3),
                objects={"object_0": dict(category="cube", asset="cube")},
                cam_pos=cpos.tolist(), fovy=fovy, cameras=[CAM], res=RES_OGB, ext="png",
                offsamples=OFFSAMPLES, visualize_info=False, hover=[Z_LO, Z_HI],
                note="corpus render config; i.i.d. cube placements hovering Z_LO..Z_HI above the surface; arm and camera frozen")
    seg_r.close()
    (cdir / "labels.json").write_text(json.dumps(dict(meta=meta, frames=recs)))
    print(json.dumps(dict(cell=int(cell), kept=len(recs), tries=tries,
                          feas=meta["feasibility"], sec=round(dt, 1))), flush=True)
    env.close()
    return len(recs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", choices=sorted(ENVS), default="scene")
    ap.add_argument("--split", choices=("id", "ood"), default="id")
    ap.add_argument("--cells", type=int, default=12)
    ap.add_argument("--first-cell", type=int, default=0, help="index of the first cell to render, so cells can run as separate processes")
    ap.add_argument("--frames", type=int, default=2667,
                    help="rows per cell; the reference pack was built at 2667")
    ap.add_argument("--out", default=os.environ.get(
        "OGBENCH_SUITE_DIR", os.path.join(os.environ.get("STABLEWM_HOME", "."), "probe_suite_ogbench")))
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()
    # disjoint seed ranges per split
    base = 100 if args.split == "id" else 0
    root = Path(args.out) / args.env / args.split
    for i in range(args.first_cell, args.first_cell + args.cells):
        cell = base + i
        cdir = root / f"C{cell:03d}"
        if args.skip_existing and (cdir / "labels.json").exists():
            print(json.dumps(dict(cell=cell, skipped_existing=True)), flush=True)
            continue
        run_cell(ENVS[args.env], cell, args.frames, cdir,
                 np.random.default_rng(10_000 + cell))


if __name__ == "__main__":
    main()
