"""eval.py inlines the ImageNet statistics rather than importing stable_pretraining: the inlined
table must equal spt's, and the import must stay out."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def test_inlined_imagenet_stats_match_stable_pretraining():
    from eval import IMAGENET_STATS

    spt = pytest.importorskip("stable_pretraining",
                              reason="eval.py no longer depends on spt; skip where it is absent")
    ref = dict(spt.data.dataset_stats.ImageNet)
    assert set(IMAGENET_STATS) == set(ref), (
        f"key mismatch: inlined {sorted(IMAGENET_STATS)} vs spt {sorted(ref)}")
    for k in ref:
        assert [float(v) for v in IMAGENET_STATS[k]] == [float(v) for v in ref[k]], (
            f"{k}: inlined {IMAGENET_STATS[k]} != spt {ref[k]} -- eval would normalize frames "
            f"differently from training, which reads as a worse policy rather than as a bug")


def test_eval_does_not_import_stable_pretraining():
    src = (Path(__file__).resolve().parents[1] / "src" / "eval.py").read_text()
    assert "import stable_pretraining" not in src, (
        "eval.py imports stable_pretraining again; it costs ~10s of startup per task and the only "
        "thing it was ever used for is IMAGENET_STATS")


if __name__ == "__main__":
    test_inlined_imagenet_stats_match_stable_pretraining()
    test_eval_does_not_import_stable_pretraining()
    print("test_eval_norm: OK")
