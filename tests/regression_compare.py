"""Compare two fingerprint directories written by regression_baseline.py.

Every scalar is compared per (forward call, key). `--noise` points at a second capture of the
UNCHANGED tree, which sets the per-key tolerance: a difference only counts when it exceeds what
the same code produces against itself.

    python tests/regression_compare.py --before results/_regr/before \
        --after results/_regr/after --noise results/_regr/before2
"""
import argparse
import json
from pathlib import Path


def load(d):
    return {p.stem: json.loads(p.read_text()) for p in sorted(Path(d).glob("*.json"))}


def deltas(a, b):
    """Max |difference| per key across matched forward calls, plus structural mismatches."""
    out, notes = {}, []
    sa, sb = a.get("steps", []), b.get("steps", [])
    if len(sa) != len(sb):
        notes.append(f"forward-call count {len(sa)} -> {len(sb)}")
    for i, (ra, rb) in enumerate(zip(sa, sb)):
        for k in set(ra) | set(rb):
            if k == "stage":
                continue
            if k not in ra:
                notes.append(f"key ADDED at call {i}: {k}")
                continue
            if k not in rb:
                notes.append(f"key REMOVED at call {i}: {k}")
                continue
            d = abs(float(ra[k]) - float(rb[k]))
            out[k] = max(out.get(k, 0.0), d)
    return out, notes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", required=True)
    ap.add_argument("--after", required=True)
    ap.add_argument("--noise", default=None, help="second capture of the unchanged tree")
    ap.add_argument("--floor", type=float, default=1e-6, help="tolerance when --noise is absent")
    ap.add_argument("--drop", default="", help="comma-separated monitor keys whose removal is accepted")
    ap.add_argument("--rename", default="", help="comma-separated old=new metric-key prefixes applied to the "
                                                "'before' capture (a rename is not a difference)")
    ap.add_argument("--recipes", default="", help="comma-separated old=new recipe names: compare before/<old> "
                                                 "with after/<new>")
    args = ap.parse_args()
    dropped = {k for k in args.drop.split(",") if k}
    renames = [r.split("=", 1) for r in args.rename.split(",") if r]
    recipe_map = dict(r.split("=", 1) for r in args.recipes.split(",") if r)

    def _rekey(k):
        for old, new in renames:
            if k.startswith(old):
                return new + k[len(old):]
        return k

    B, A = load(args.before), load(args.after)
    # several captures of the unchanged tree (comma-separated) give a per-key floor that is the
    # largest run-to-run difference seen among them
    NZ = [load(d) for d in args.noise.split(",")] if args.noise else []
    for cap in [B, *NZ]:
        for name in list(cap):
            for row in cap[name].get("steps", []):
                for k in list(row):
                    if _rekey(k) != k:
                        row[_rekey(k)] = row.pop(k)
            if name in recipe_map:
                cap[recipe_map[name]] = cap.pop(name)
    rc = 0
    for name in sorted(set(B) | set(A)):
        if name not in A or name not in B:
            print(f"{name}: MISSING from {'after' if name not in A else 'before'}")
            rc = 1
            continue
        if B[name]["status"] != A[name]["status"]:
            print(f"{name}: status {B[name]['status']} -> {A[name]['status']}  {A[name]['error']}")
            rc = 1
            continue
        if B[name]["status"] != "ok":
            print(f"{name}: SKIPPED both sides ({B[name]['error']})")
            continue
        d, notes = deltas(B[name], A[name])
        notes = [n for n in notes if not (n.startswith("key REMOVED") and n.split(": ")[-1] in dropped)]
        tol = {}
        for nz in NZ:
            if name in nz and nz[name]["status"] == "ok":
                t, _ = deltas(B[name], nz[name])
                tol = {k: max(tol.get(k, 0.0), v) for k, v in {**tol, **t}.items()}
        bad = {k: v for k, v in d.items()
               if v > max(args.floor, 2.0 * tol.get(k, 0.0))}
        if not bad and not notes:
            print(f"{name}: MATCH ({len(d)} keys, {len(B[name]['steps'])} calls)")
            continue
        rc = 1
        print(f"{name}: DIFF")
        for n in notes:
            print(f"    {n}")
        for k, v in sorted(bad.items(), key=lambda kv: -kv[1])[:12]:
            print(f"    {k:28} max|d| {v:.3e}   noise {tol.get(k, 0.0):.3e}")
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
