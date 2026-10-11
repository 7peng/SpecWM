# OGBench probe suite (scene, cube)

The object-recoverability probe suite on the OGBench `swm/OGBScene-v0` and
`swm/OGBCube-v0` environments. Each sample is an independently placed cube in an otherwise
fixed scene, so the probe measures whether an encoder represents *where the object is* rather
than which episode it is looking at.

```bash
MUJOCO_GL=egl python data/ogbench/make_ogbench_suite.py --env scene --split id --cells 1 --frames 2667
MUJOCO_GL=egl python data/ogbench/make_ogbench_suite.py --env cube  --split id --cells 1 --frames 2667
python data/probe_suite/build_probe_pack.py --suite $OGBENCH_SUITE_DIR/scene --out $OGBENCH_PACK_DIR/scene --rows 2667
```

`MUJOCO_EGL_DEVICE_ID=<gpu>` is needed to render on a GPU other than 0.

## Validity tests

The frustum test, the raycast occlusion check, the interpenetration check, `obj_in_cam` and the
constants (`RES=256`, `MIN_PX=10`, `Z_LO/Z_HI`, `PEN`, `FRUSTUM_MARGIN`) live in
`data/probe_suite/suite_geometry.py` and act on raw MuJoCo model/data. The cube is teleported
by writing `qpos[14:17]` (`object_joint_0`) and calling `mj_forward`.

Output is `<suite>/<split>/<cell>/frames/*.png` + `labels.json`, which `build_probe_pack.py`
consumes.

## What is held fixed, and what varies

Default camera, default colours, default lighting; the arm is frozen at its reset pose, and on
`scene` so are the drawer, window and buttons. Only the cube's position and yaw vary, drawn
i.i.d. per frame. Position is therefore independent of arm pose, camera pose and cell by
construction, so proprioception and scene identity carry no information about the label.

## What a reader should know

- **No OOD split.** OGBench has a single scene, so nothing about the scene is held out and only
  cube positions are unseen. These numbers measure in-distribution recoverability.
- **Cells are repeats.** They differ only in placement RNG, so several cells give error bars,
  not a generalisation estimate.
- **Effective support differs by environment.** Placements are drawn uniformly over the env's
  `_object_sampling_bounds` and kept if they pass the frustum and occlusion tests, which yields a
  uniform distribution over the *visible* region. On `scene` that is the whole box (feasibility
  ~1.0); on `cube` the frustum admits only x in about [0.30, 0.40] of a [0.30, 0.55] box
  (feasibility ~0.23). Absolute scores are therefore not comparable between the two environments.
- **Cubes float.** `Z_LO..Z_HI` places the object 1-9 cm above the surface, so the target is
  3D. The cube rests at z about 0.020.

Self-test: an 8x8-pooled-pixel ridge recovers `obj_in_cam` at R^2 = 0.96 on an 800-frame cell,
while the arm and camera are frozen and carry nothing -- the trivial baselines are a check on the
generator rather than a correction applied afterwards.
