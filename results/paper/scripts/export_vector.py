"""Run a figure script unchanged and write a PDF and an SVG next to every PNG it saves, into --dir.

    python results/paper/scripts/export_vector.py --dir results/paper/vector -- <script.py> <script args>
"""
import runpy, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure

i = sys.argv.index("--"); out = Path(sys.argv[sys.argv.index("--dir") + 1]); out.mkdir(parents=True, exist_ok=True)
script, args = sys.argv[i + 1], sys.argv[i + 2:]
matplotlib.rcParams["svg.fonttype"] = "none"      # keep text as text in the SVG
matplotlib.rcParams["pdf.fonttype"] = 42          # embed TrueType, editable in Illustrator / Inkscape
_save = Figure.savefig


def savefig(self, fname, *a, **k):
    _save(self, fname, *a, **k)
    stem = Path(str(fname)).stem
    for ext in ("pdf", "svg"):
        _save(self, out / f"{stem}.{ext}", *a, **{**k, "format": ext})
        print(f"[vector] {out / (stem + '.' + ext)}", flush=True)


Figure.savefig = savefig
sys.argv = [script] + args
runpy.run_path(script, run_name="__main__")
