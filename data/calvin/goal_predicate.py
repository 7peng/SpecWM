"""The CALVIN goal predicate: is a canonical scene_obs at the goal?

ONE definition, shared by the task-set builder (which keeps only windows whose goal is NOT already
met at the start) and the eval env (which terminates the episode when it is met), so a task that is
non-trivial by construction is scored by the same rule that made it non-trivial -- the analogue of
data/scene/make_scene_offset_tasks.py sharing SceneEnv's 0.04 m success tolerance.

Every factor is checked, moved or not, as OGBench scene checks every object: a policy that reaches
the goal drawer while knocking a block off the table has not reached the goal. Robot pose is not a
factor. Block orientation is not a factor either (the blocks are near-cubes; Euler angles wrap and
flip at contact, and no scene-style predicate scores them).

TOLERANCES (continuous factors 15% of the factor's range, binary exact). The
range is the 1st-99th percentile span of the canonical scene_obs over calvin_abc_d_noised (1.79M
frames): slider 0-0.304 -> 0.046, drawer 0-0.220 -> 0.033, switch 0-0.088 -> 0.013. The button's
depth is 0-0.011 at p99 (0.041 max) and reads as a press/no-press in the data, so it takes 15% of
its full range, 0.006. Lightbulb and LED are 0/1 and must match exactly. Block position uses one
absolute 0.05 m radius (the blocks are ~5 cm; scene's cubes use 0.04 m) -- a percentage of a
position range across the whole table would be meaningless.
"""
import numpy as np

SLIDER, DRAWER, BUTTON, SWITCH, LIGHTBULB, LED = 0, 1, 2, 3, 4, 5
BLOCK_XYZ = [slice(6, 9), slice(12, 15), slice(18, 21)]     # red, blue, pink (canonical order)
TOL = {"slider": 0.046, "drawer": 0.033, "button": 0.006, "switch": 0.013, "block_xyz": 0.05}
FACTORS = ["slider", "drawer", "button", "switch", "lightbulb", "led", "red", "blue", "pink"]


def factor_errors(cur, goal):
    """Per-factor distance of `cur` from `goal` (both canonical (24,) scene_obs), in each factor's
    own units: |delta| for the articulated joints and lights, Euclidean xyz for the blocks."""
    cur, goal = np.asarray(cur, np.float64).reshape(-1), np.asarray(goal, np.float64).reshape(-1)
    assert cur.shape == (24,) and goal.shape == (24,), (cur.shape, goal.shape)
    e = {"slider": abs(cur[SLIDER] - goal[SLIDER]), "drawer": abs(cur[DRAWER] - goal[DRAWER]),
         "button": abs(cur[BUTTON] - goal[BUTTON]), "switch": abs(cur[SWITCH] - goal[SWITCH]),
         "lightbulb": abs(round(cur[LIGHTBULB]) - round(goal[LIGHTBULB])),
         "led": abs(round(cur[LED]) - round(goal[LED]))}
    for name, sl in zip(("red", "blue", "pink"), BLOCK_XYZ):
        e[name] = float(np.linalg.norm(cur[sl] - goal[sl]))
    return e


def factor_ok(cur, goal, tol=TOL):
    e = factor_errors(cur, goal)
    ok = {"slider": e["slider"] <= tol["slider"], "drawer": e["drawer"] <= tol["drawer"],
          "button": e["button"] <= tol["button"], "switch": e["switch"] <= tol["switch"],
          "lightbulb": e["lightbulb"] == 0, "led": e["led"] == 0}
    for name in ("red", "blue", "pink"):
        ok[name] = e[name] <= tol["block_xyz"]
    return ok


def at_goal(cur, goal, tol=TOL):
    return all(factor_ok(cur, goal, tol).values())


def moved_factors(start, goal, tol=TOL):
    """The factors the window changes beyond tolerance -- what a task is 'about'."""
    return [k for k, v in factor_ok(start, goal, tol).items() if not v]
