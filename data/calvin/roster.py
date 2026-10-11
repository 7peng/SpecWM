"""The CALVIN block roster, and the permutation that makes it the same in every scene.

THE PROBLEM. `scene_obs[6:24]` is three movable blocks x (xyz, xyz-Euler), and every reader in
this repo -- the probe targets, the K* factor specs, the state-kernel battery -- names those
slots red, blue, pink. The simulator does not: `PlayTableScene` builds `self.movable_objects`
by iterating its scene config's `movable_objects` mapping, and writes `get_obs` in that order,
so the slot order is whatever the scene's yaml lists. calvin_env ships a different order in
three of the four scenes:

    scene A   pink  blue  red
    scene B   red   blue  pink
    scene C   blue  red   pink
    scene D   red   blue  pink

CONFIRMED FROM THE DATA, not only from the configs. Over the task_ABC_D language annotations
that name exactly one block colour, the slot whose xyz moves most between the window's first
and last frame is the slot this table says holds that colour, on 10,933 of 11,034 windows
(99.1%; the remainder are windows where the arm displaced a different block on the way).
`canonicalize_scene_obs.py --check` recomputes it.

WHAT IT BREAKS. A reader that takes scene_obs[6:9] as "red" reads red only in scenes B and D --
in A it reads pink and in C it reads blue. A per-frame probe target is then a mixture of three
physical blocks, and a state kernel comparing two frames from different scenes compares
different objects.

THE FIX is applied at the source: `build_calvin_h5.py` permutes each frame's three block
groups into the canonical red, blue, pink order before writing, and stamps
`scene_obs.attrs["roster"]`. `canonicalize_scene_obs.py` migrates a pack built before this.
Downstream code should assert the attribute rather than permute again.
"""

CANONICAL = ("red", "blue", "pink")

# movable_objects order in $CALVIN_ENV_SRC/conf/scene/calvin_scene_<X>.yaml == the order
# PlayTableScene.get_obs writes the blocks into scene_obs[6:24].
SCENE_ROSTER = {
    "A": ("pink", "blue", "red"),
    "B": ("red", "blue", "pink"),
    "C": ("blue", "red", "pink"),
    "D": ("red", "blue", "pink"),
}

# scene_id column values, matching the enumeration build_calvin_h5.py writes.
SCENE_IDS = {"A": 0, "B": 1, "C": 2, "D": 3}

BLOCK_START = 6          # scene_obs[0:6] is the articulated state, identical in every scene
BLOCK_DIM = 6            # xyz + xyz-Euler


def source_slots(scene):
    """Source slot index for each canonical position, e.g. scene A -> (2, 1, 0)."""
    roster = SCENE_ROSTER[scene]
    return tuple(roster.index(c) for c in CANONICAL)


def column_permutation(scene):
    """Index array over the 24 scene_obs columns that puts the blocks in canonical order."""
    cols = list(range(BLOCK_START))
    for src in source_slots(scene):
        a = BLOCK_START + BLOCK_DIM * src
        cols.extend(range(a, a + BLOCK_DIM))
    return cols


def canonicalize(scene_obs, scene_id):
    """(N, 24) scene_obs + (N,) scene_id -> a copy with the blocks in red, blue, pink order.

    Frames whose scene already lists the canonical order are copied unchanged. Every frame must
    carry a known scene id: an unlabelled frame cannot be permuted and is not silently kept.
    """
    import numpy as np

    scene_obs = np.asarray(scene_obs)
    scene_id = np.asarray(scene_id).reshape(-1)
    if scene_obs.shape[0] != scene_id.shape[0]:
        raise ValueError(f"scene_obs has {scene_obs.shape[0]} rows, scene_id {scene_id.shape[0]}")
    if scene_obs.shape[1] != BLOCK_START + BLOCK_DIM * len(CANONICAL):
        raise ValueError(f"scene_obs must be 24-column, got {scene_obs.shape[1]}")
    known = np.isin(scene_id, list(SCENE_IDS.values()))
    if not known.all():
        raise ValueError(f"{(~known).sum()} frames carry a scene id outside {sorted(SCENE_IDS.values())}")

    out = np.array(scene_obs, copy=True)
    for scene, sid in SCENE_IDS.items():
        cols = column_permutation(scene)
        if cols == list(range(scene_obs.shape[1])):
            continue
        rows = scene_id == sid
        if rows.any():
            out[rows] = scene_obs[rows][:, cols]
    return out


def roster_attr():
    """The value stamped on scene_obs once its blocks are canonical."""
    return ",".join(CANONICAL)


def assert_canonical(path, context=""):
    """Raise unless the pack at `path` carries canonical scene_obs blocks.

    A pack built before 2026-09-04 does not, and every reader that names a block slot -- probe
    targets, K* factor specs, the state-kernel battery -- is then reading a different physical
    block in two of the three training scenes. That is silent and it is not recoverable from
    the numbers afterwards, so it is a hard failure with the one-line fix in the message.
    """
    import h5py

    with h5py.File(path, "r") as f:
        if "scene_obs" not in f:
            return False
        stamped = f["scene_obs"].attrs.get("roster")
    if stamped == roster_attr():
        return True
    where = f" ({context})" if context else ""
    raise RuntimeError(
        f"CALVIN pack {path}{where} has scene_obs roster {stamped!r}, not {roster_attr()!r}: its "
        "three block slots are in the simulator's per-scene order, so slot 0 is pink in scene A "
        "and blue in scene C. Fix it in place (172 MB, ~30 s, pixels untouched):\n"
        f"    python data/calvin/canonicalize_scene_obs.py --h5 {path}")


def verify_against_configs(calvin_env_src):
    """Re-read the scene yamls and check SCENE_ROSTER still describes them.

    The table is a claim about another repository's config files, so it is checked rather than
    trusted whenever that checkout is reachable.
    """
    import re
    from pathlib import Path

    conf = Path(calvin_env_src) / "conf" / "scene"
    for scene, expected in SCENE_ROSTER.items():
        text = (conf / f"calvin_scene_{scene}.yaml").read_text()
        body = text.split("movable_objects:", 1)[1]
        found = tuple(m.group(1).split("block_")[1]
                      for m in re.finditer(r"^\s{4}(block_\w+):", body, re.M))
        if found != expected:
            raise AssertionError(
                f"calvin_scene_{scene}.yaml lists {found}, SCENE_ROSTER says {expected}")
    return True
