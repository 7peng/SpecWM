"""One reader for CEM cells: results/<run>-<tag>/O<offset>.txt in this checkout (or another results root), else the
collaborator's handoff table data/handoff_<date>/cem_scores.csv (run_id, checkpoint tag, success_rate_pct)."""
import re
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
_HANDOFF = None


def handoff():
    global _HANDOFF
    if _HANDOFF is None:
        parts = [pd.read_csv(p) for p in sorted((HERE / "data").glob("handoff_*/cem_scores.csv"))]
        _HANDOFF = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["run_id", "checkpoint", "goal_offset", "success_rate_pct"])
    return _HANDOFF


def cem_cell(run, tag, offset, roots=(HERE.parent,)):
    """Success rate (%) of `run` at checkpoint `tag` ('s20000' or 'final') and goal offset `offset`, or NaN."""
    for root in roots:
        p = Path(root) / f"{run}-{tag}" / f"O{offset}.txt"
        m = re.findall(r"'success_rate': ([0-9.]+)", p.read_text()) if p.exists() else []
        if m: return float(m[-1])
    h = handoff(); r = h[(h.run_id == run) & (h.checkpoint == tag) & (h.goal_offset == offset)]
    return float(r.success_rate_pct.iloc[-1]) if len(r) else np.nan
