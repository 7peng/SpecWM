"""Placement geometry shared by the OGBench probe suite.

The frustum test, the raycast occlusion and camera-clearance checks and the interpenetration
test that `data/ogbench/make_ogbench_suite.py` applies to every candidate object placement,
with the constants they read. Everything here takes raw MuJoCo `MjModel` / `MjData` handles
and camera pose, so it is independent of the environment wrapper. Not run on its own:

    from suite_geometry import in_frustum, unoccluded, penetrating, camera_blocked, obj_in_cam
"""
import numpy as np

RES = 256
MIN_PX = 10                  # minimum visible pixels for a placement to count
Z_LO, Z_HI = 0.01, 0.09      # object height above the surface (m)
PEN = 0.008                  # allowed interpenetration (m)
FRUSTUM_MARGIN = 0.9
CAM_CLEAR = 0.35             # metres the camera must see past along its axis to be usable


def in_frustum(P, cpos, cmat, fovy):
    """(N,) bool: which of the (N,3) world points P lie inside the camera frustum, shrunk by
    FRUSTUM_MARGIN and at least 0.25 m in front of the camera. No simulation required."""
    rel = (np.atleast_2d(P) - cpos) @ cmat
    z = -rel[:, 2]
    t = np.tan(np.radians(fovy) / 2) * FRUSTUM_MARGIN
    ok = z > 0.25
    zs = np.where(ok, z, 1.0)
    return ok & (np.abs(rel[:, 0] / zs) < t) & (np.abs(rel[:, 1] / zs) < t)


def obj_in_cam(P, cpos, cmat):
    """(N,3) camera-frame coordinates of the (N,3) world points P."""
    return (np.atleast_2d(P) - cpos) @ cmat


def ray_dist(m, d, src, tgt):
    """(distance to the first geom along src->tgt or inf, |tgt - src|)."""
    import mujoco
    vec = np.asarray(tgt, float) - src
    L = float(np.linalg.norm(vec))
    if L < 1e-6:
        return float("inf"), L
    g = np.zeros(1, dtype=np.int32)
    dist = mujoco.mj_ray(m, d, src, vec / L, None, 1, -1, g)
    return (float(dist) if dist >= 0 else float("inf")), L


def camera_blocked(m, d, cpos, cmat):
    """True if the median clearance along the camera axis and four tilted rays is below
    CAM_CLEAR, i.e. the camera sits inside or hard against geometry."""
    fwd = -cmat[:, 2]
    up, right = cmat[:, 1], cmat[:, 0]
    dists = []
    for v in (fwd, fwd + 0.3 * up, fwd - 0.3 * up, fwd + 0.3 * right, fwd - 0.3 * right):
        v = v / np.linalg.norm(v)
        dists.append(ray_dist(m, d, cpos, cpos + 10.0 * v)[0])
    return float(np.median(dists)) < CAM_CLEAR


def penetrating(d, gids):
    """True if any active contact between a geom in `gids` and one outside it is deeper than
    PEN."""
    for i in range(d.ncon):
        c = d.contact[i]
        if (c.geom1 in gids) != (c.geom2 in gids) and c.dist < -PEN:
            return True
    return False


def unoccluded(m, d, bid, gids, cpos):
    """True if a ray from the camera to the centre of body `bid` or to one of four 2 cm lateral
    offsets first hits a geom of that body."""
    import mujoco
    for off in ([0, 0, 0], [0.02, 0, 0], [-0.02, 0, 0], [0, 0.02, 0], [0, -0.02, 0]):
        tgt = d.xpos[bid] + np.array(off)
        vec = tgt - cpos
        L = float(np.linalg.norm(vec))
        if L < 1e-6:
            continue
        g = np.zeros(1, dtype=np.int32)
        mujoco.mj_ray(m, d, cpos, vec / L, None, 1, -1, g)
        if g[0] in gids:
            return True
    return False
