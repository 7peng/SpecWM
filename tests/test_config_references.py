"""Every recipe named in paper/RUNS.tsv and tests/paper_recipes.txt exists under config/train
with an existing entry point, every bare `defaults:` entry of a recipe resolves, and no file under
src/, scripts/, tests/ or data/ launches a config that does not exist."""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "config" / "train"
EVAL = ROOT / "config" / "eval"
SEARCH = ("src", "scripts", "tests", "data")
SUFFIXES = (".py", ".sh", ".sbatch", ".yaml")
FLAG = re.compile(r"--config-name[= ]([\w.]+)")
CALL = re.compile(r"""config_name=["']([\w.]+)["']""")
DEFAULT_ENTRY = re.compile(r"^\s*-\s*([\w.]+)\s*(?:#.*)?$")


def paper_rows():
    rows = []
    for line in (ROOT / "paper" / "RUNS.tsv").read_text().splitlines()[1:]:
        if line.strip():
            f = line.split("\t")
            rows.append(("paper/RUNS.tsv", f[3], f[4]))
    for line in (ROOT / "tests" / "paper_recipes.txt").read_text().splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            f = s.split()
            rows.append(("tests/paper_recipes.txt", f[0], f[1]))
    return rows


def test_paper_recipes_and_entry_points_exist():
    rows = paper_rows()
    assert rows
    missing = sorted({f"{src}: {entry} {recipe}" for src, entry, recipe in rows
                      if not (TRAIN / f"{recipe}.yaml").is_file() or not (ROOT / "src" / f"{entry}.py").is_file()})
    assert not missing, "\n".join(missing)


def test_defaults_entries_resolve():
    live = {p.stem for p in TRAIN.glob("*.yaml")}
    broken = []
    for p in sorted(TRAIN.glob("*.yaml")):
        inside = False
        for i, line in enumerate(p.read_text().split("\n"), 1):
            if re.match(r"^defaults:\s*$", line):
                inside = True
                continue
            if inside and line and not line[0].isspace():
                inside = False
            m = DEFAULT_ENTRY.match(line) if inside else None
            if m and m.group(1) != "_self_" and m.group(1) not in live:
                broken.append(f"config/train/{p.name}:{i} defaults on '{m.group(1)}'")
    assert not broken, "\n".join(broken)


def test_no_launch_names_a_missing_config():
    live = {p.stem for p in TRAIN.glob("*.yaml")} | {p.stem for p in EVAL.glob("*.yaml")}
    broken = []
    for sub in SEARCH:
        for f in sorted((ROOT / sub).rglob("*")):
            if f.suffix not in SUFFIXES or not f.is_file():
                continue
            try:
                text = f.read_text()
            except (UnicodeDecodeError, OSError):
                continue
            # a `config_name=` call composes from our tree only when the file names it
            ours = "config/train" in text or "config/eval" in text or "config_path=" in text
            for i, line in enumerate(text.split("\n"), 1):
                if line.lstrip().startswith("#"):
                    continue
                names = FLAG.findall(line) + (CALL.findall(line) if ours else [])
                for name in names:
                    if name not in live:
                        broken.append(f"{f.relative_to(ROOT)}:{i} names config '{name}'")
    assert not broken, "\n".join(broken)


if __name__ == "__main__":
    test_paper_recipes_and_entry_points_exist()
    test_defaults_entries_resolve()
    test_no_launch_names_a_missing_config()
    print("test_config_references: OK")
