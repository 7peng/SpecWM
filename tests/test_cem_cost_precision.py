"""The planner scores in fp32: no autocast / half / bf16 on the eval path, fp32 ranks the elite
population the solver converges to (1 - cos ~ 6e-4, spread ~1e-5) as fp64 does while bf16 would
not, and `DeltaCosineCostModel` returns the parameter dtype."""
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
os.environ.setdefault("MUJOCO_GL", "egl")

import torch                                                                    # noqa: E402
from torch import nn                                                            # noqa: E402

from eval import DeltaCosineCostModel                                           # noqa: E402
from jepa import JEPA                                                           # noqa: E402

D, S, K = 192, 300, 30


def test_eval_path_has_no_reduced_precision():
    bad = []
    for name in ("eval.py", "jepa.py"):
        src = open(os.path.join(REPO, "src", name)).read()
        for m in re.finditer(r"autocast|\.half\(\)|bfloat16|float16", src):
            bad.append(f"{name}:{src[: m.start()].count(chr(10)) + 1} {m.group(0)}")
    assert not bad, f"reduced precision on the planner path: {bad}"


def _cos_cost(pred, goal):
    p = torch.nn.functional.normalize(pred, dim=-1)
    g = torch.nn.functional.normalize(goal, dim=-1)
    return 1.0 - (p * g.unsqueeze(0)).sum(-1)


def test_fp32_ranks_the_elite_population_correctly():
    g64 = torch.Generator().manual_seed(0)
    goal = torch.randn(D, generator=g64, dtype=torch.float64)
    goal = goal / goal.norm()
    c = torch.rand(S, generator=g64, dtype=torch.float64) * 2e-5 + 5.9e-4
    perp = torch.randn(S, D, generator=g64, dtype=torch.float64)
    perp = perp - (perp @ goal)[:, None] * goal[None]
    perp = perp / perp.norm(dim=-1, keepdim=True)
    cosv = 1.0 - c
    pred64 = cosv[:, None] * goal[None] + torch.sqrt(1 - cosv ** 2)[:, None] * perp
    ref = _cos_cost(pred64, goal)
    top_ref = set(torch.topk(ref, K, largest=False).indices.tolist())
    got32 = _cos_cost(pred64.float(), goal.float()).double()
    agree32 = len(top_ref & set(torch.topk(got32, K, largest=False).indices.tolist()))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        got_bf = _cos_cost(pred64.bfloat16(), goal.bfloat16()).double()
    agree_bf = len(top_ref & set(torch.topk(got_bf, K, largest=False).indices.tolist()))
    assert agree32 >= K - 2, f"fp32 disagrees with fp64 on {K - agree32} of {K} elites"
    assert agree_bf < K - 2, "the bf16 contrast no longer holds"


def test_cost_model_runs_in_the_parameter_dtype():
    class Enc(nn.Module):
        class Cfg:
            hidden_size = 8

        def __init__(self):
            super().__init__()
            self.config = self.Cfg()
            self.lin = nn.Linear(8, 8)

        def forward(self, pixels, interpolate_pos_encoding=False, **kw):
            h = self.lin(pixels.reshape(pixels.shape[0], -1)[:, :8])
            return type("O", (), {"last_hidden_state": torch.stack([h, h], 1)})()

    class Pred(nn.Module):
        def __init__(self):
            super().__init__()
            self.state_pos = nn.Parameter(torch.zeros(1, 3, 8))

        def forward(self, ctx, bridge, hor, mask=None):
            return ctx[:, -1] + bridge.mean(1)

    m = JEPA(encoder=Enc(), predictor=Pred(), action_encoder=nn.Linear(12, 8),
             projector=nn.Identity(), pred_proj=nn.Identity(), embed_dim=8,
             emb_norm=False, pred_norm=False).eval()
    cm = DeltaCosineCostModel(m)
    assert next(cm.parameters()).dtype == torch.float32
    info = {"pixels": torch.rand(1, 4, 3, 3, 4, 4), "goal": torch.rand(1, 4, 1, 3, 4, 4)}
    with torch.inference_mode():
        c = cm.get_cost(info, torch.randn(1, 4, 3, 12))
    assert c.dtype == torch.float32 and c.shape == (1, 4)


if __name__ == "__main__":
    test_eval_path_has_no_reduced_precision()
    test_fp32_ranks_the_elite_population_correctly()
    test_cost_model_runs_in_the_parameter_dtype()
    print("test_cem_cost_precision: OK")
