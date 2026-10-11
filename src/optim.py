"""MuonAdamW: single-optimizer facade -- Muon on 2D matrix weights, AdamW on the rest."""
import collections

import torch
from torch.optim._muon import _adjust_lr, _zeropower_via_newtonschulz, muon as _muon_fn


class GraphedMuon(torch.optim.Muon):
    """torch.optim.Muon with the Newton-Schulz orthogonalisation of every matrix replayed from one CUDA
    graph (torch's own `_zeropower_via_newtonschulz`, captured over static buffers on the first step and
    again whenever the set of matrices holding a gradient changes). Momentum and parameter updates go
    through the foreach kernels; the learning rate is read eagerly each step, so schedulers are unaffected;
    the state is torch.optim.Muon's, so checkpoints load either way. Parameters on a CPU device take
    torch's eager path."""

    def __init__(self, params, **kw):
        super().__init__(params, **kw)
        self._captures = {}

    def _capture(self, group, params):
        coeffs, steps, eps = group["ns_coefficients"], group["ns_steps"], group["eps"]
        ins = [torch.zeros_like(p) for p in params]
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):      # warmup on a side stream, as capture requires
                for x in ins:
                    _zeropower_via_newtonschulz(x, coeffs, steps, eps)
        torch.cuda.current_stream().wait_stream(s)
        graph = torch.cuda.CUDAGraph()
        # thread_local: the DataLoader's pin-memory thread makes CUDA calls while the capture is open
        with torch.cuda.graph(graph, capture_error_mode="thread_local"):
            # fp32, contiguous in the parameter's own layout: what `param.add_(bf16_update)`
            # computes elementwise, and what the foreach fast path needs
            outs = [_zeropower_via_newtonschulz(x, coeffs, steps, eps).float().contiguous() for x in ins]
        by_ratio = collections.defaultdict(list)
        for i, p in enumerate(params):
            by_ratio[_adjust_lr(1.0, group["adjust_lr_fn"], p.shape)].append(i)
        print(f"[GraphedMuon] captured NS for {len(params)} matrices "
              f"({len(by_ratio)} lr-adjust classes)", flush=True)
        return dict(ids=tuple(id(p) for p in params), ins=ins, outs=outs, graph=graph, by_ratio=dict(by_ratio))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for gi, group in enumerate(self.param_groups):
            params, grads, bufs = [], [], []
            has_complex = self._init_group(group, params, grads, bufs)
            if not params:
                continue
            lr, wd, mom = group["lr"], group["weight_decay"], group["momentum"]
            if not params[0].is_cuda:
                _muon_fn(params, grads, bufs, lr=lr, weight_decay=wd, momentum=mom,
                         nesterov=group["nesterov"], ns_coefficients=group["ns_coefficients"],
                         ns_steps=group["ns_steps"], eps=group["eps"],
                         adjust_lr_fn=group["adjust_lr_fn"], has_complex=has_complex)
                continue
            torch._foreach_lerp_(bufs, grads, 1 - mom)
            upd = torch._foreach_lerp(grads, bufs, mom) if group["nesterov"] else bufs
            cap = self._captures.get(gi)
            if cap is None or cap["ids"] != tuple(id(p) for p in params):
                cap = self._captures[gi] = self._capture(group, params)
            torch._foreach_copy_(cap["ins"], upd)
            cap["graph"].replay()
            torch._foreach_mul_(params, 1 - lr * wd)
            for ratio, idx in cap["by_ratio"].items():
                torch._foreach_add_([params[i] for i in idx], [cap["outs"][i] for i in idx],
                                    alpha=-_adjust_lr(lr, group["adjust_lr_fn"], params[idx[0]].shape))
        return loss


class MuonAdamW(torch.optim.Optimizer):
    """Single-optimizer facade: Muon on 2D matrix weights, AdamW on everything else (1D biases/norms,
    3D tokens, 4D conv kernels). The children's param_groups are exposed directly (shared dict objects),
    so a scheduler scaling group['lr'] propagates into each child; step()/zero_grad()/state_dict() fan
    out to both. adjust_lr_fn='match_rms_adamw' matches Muon's update RMS to AdamW's so one lr is shared
    (a matrix then steps at 0.2 * muon_lr per element).

    `nonmatrix_lr` (None = off) puts every parameter that is neither a 2-D matrix nor a 1-D vector (the
    cls/position tokens, the patch-embedding conv) into its own AdamW group at that learning rate.
    Weight decay follows the kind: conv kernels and matrices keep it, tokens, norms and biases do not.
    `no_muon` names 2-D parameters (embedding tables) that stay on AdamW."""

    def __init__(self, params, lr=1e-3, weight_decay=0.0, momentum=0.95,
                 betas=(0.9, 0.999), eps=1e-8, muon_lr=None, adamw_lr=None,
                 adjust_lr_fn="match_rms_adamw", nonmatrix_lr=None, muon_graph=True, no_muon=()):
        no_muon = set(no_muon)
        params = list(params)
        tensors = ([p for g in params for p in g["params"]]
                   if params and isinstance(params[0], dict) else params)
        tensors = [p for p in tensors if p.requires_grad]
        matrix = [p for p in tensors if p.ndim == 2 and id(p) not in no_muon]
        other = [p for p in tensors if p.ndim != 2 or id(p) in no_muon]
        _split = nonmatrix_lr is not None
        no_decay = [p for p in other if p.ndim == 1 or (p.ndim == 3 and not _split) or id(p) in no_muon]
        decay_other = [p for p in other if p.ndim not in (1, 3) and not _split and id(p) not in no_muon]
        fast_tok = [p for p in other if p.ndim == 3] if _split else []
        fast_dec = [p for p in other if p.ndim not in (1, 3) and id(p) not in no_muon] if _split else []
        super().__init__(tensors, dict(lr=lr, weight_decay=weight_decay))  # install hook machinery
        self._children = []
        self.muon = self.adamw = None
        if matrix:
            _cls = GraphedMuon if muon_graph else torch.optim.Muon
            self.muon = _cls(matrix, lr=(muon_lr or lr), weight_decay=weight_decay,
                             momentum=momentum, adjust_lr_fn=adjust_lr_fn)
            self._children.append(self.muon)
        if other:
            alr = (adamw_lr or lr)
            groups = []
            if decay_other:
                groups.append({"params": decay_other, "weight_decay": weight_decay, "lr": alr})
            if no_decay:
                groups.append({"params": no_decay, "weight_decay": 0.0, "lr": alr})
            if fast_tok:
                groups.append({"params": fast_tok, "weight_decay": 0.0,
                               "lr": float(nonmatrix_lr)})
            if fast_dec:
                groups.append({"params": fast_dec, "weight_decay": weight_decay,
                               "lr": float(nonmatrix_lr)})
            self.adamw = torch.optim.AdamW(groups, lr=alr, betas=betas, eps=eps, weight_decay=weight_decay)
            self._children.append(self.adamw)
        if not self._children:
            raise ValueError("MuonAdamW: no trainable parameters")
        # repoint the facade at the children's groups/state (shared dict objects)
        self.param_groups = [g for c in self._children for g in c.param_groups]
        self.state = collections.ChainMap(*[c.state for c in self._children])
        _mstep = 0.2 * (muon_lr or lr) if adjust_lr_fn == "match_rms_adamw" else (muon_lr or lr)
        print(f"[MuonAdamW] {'Graphed' if muon_graph else 'eager'} Muon over {len(matrix)} matrix (2D, wd={weight_decay}, "
              f"~{_mstep:.2e}/element); AdamW over {len(decay_other)} conv/other "
              f"(wd={weight_decay}) + {len(no_decay)} norm/bias{'' if _split else '/token'} "
              f"(wd=0, {(adamw_lr or lr):.2e})"
              + (f" + {len(fast_tok)} token 3D (wd=0) + {len(fast_dec)} conv/other "
                 f"(wd={weight_decay}) at NONMATRIX {float(nonmatrix_lr):.2e}" if _split else ""))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for c in self._children:
            c.step()
        return loss

    def zero_grad(self, set_to_none=True):
        for c in self._children:
            c.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return {"children": [c.state_dict() for c in self._children]}

    def load_state_dict(self, sd):
        for c, s in zip(self._children, sd["children"]):
            c.load_state_dict(s)


