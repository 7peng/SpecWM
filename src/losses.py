"""Training objectives: SIGReg and the live IQE quasimetric head that trains the quasimetric teachers."""
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn


class SIGReg(torch.nn.Module):
    """Epps-Pulley isotropic-Gaussian regularizer on random unit projections (single-GPU statistic)."""

    def __init__(self, knots=17, num_proj=1024):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        """proj: (T, B, D); the statistic is estimated independently per timestep, in fp32 with
        autocast off so the logged value is not quantised to the bf16 grid."""
        with torch.autocast(device_type=proj.device.type, enabled=False):
            projf = proj.float()
            A = torch.randn(projf.size(-1), self.num_proj, device=projf.device)
            A = A.div_(A.norm(p=2, dim=0))
            x_t = (projf @ A).unsqueeze(-1) * self.t
            err = (x_t.cos().mean(-3) - self.phi).square() + \
                x_t.sin().mean(-3).square()
            statistic = (err @ self.weights) * projf.size(-2)
            return statistic.mean()


def _prefix_sum_matrix(n, device, dtype):
    """Lower-triangular ones (n,n): `x @ M.T` is a prefix sum of `x` along its last dim."""
    return torch.tril(torch.ones(n, n, device=device, dtype=dtype))


class QRLHead(nn.Module):
    """Live quasimetric head: z = MLP(emb) (optionally group-RMS-normalised), d = asymmetric IQE
    (arXiv:2211.15120) with a learnable max/mean mix alpha and a positive scale. Objective
    push_weight * L_push + lambda * L_step: L_push = -mean d over the within-clip grid (rows at
    least linear_min_gap apart, forward pairs only under push_forward_only), L_step a soft ceiling
    on adjacent-frame distance, lambda a leaky integral dual."""

    def __init__(self, embed_dim, hidden=384, z_dim=384, group_size=16,
                 push_offset=15.0, push_beta=0.1, step_cost=1.0, step_eps=0.25,
                 dual_lr=0.05, lambda_init=1.0, lambda_max=None, dual_leak=0.0,
                 push_samples=128, push_within_clip=False, push_weight=1.0,
                 push_linear=False, linear_min_gap=1, step_clamp=0.0, push_forward_only=False,
                 alpha_init=0.5, scale_init=1.0, alpha_learn=True, alpha_lr_scale=1.0,
                 kernel_normalize=False, mlp_hidden_layers=1,
                 z_norm=False, z_group_norm=False, window_size=1):
        super().__init__()
        assert push_linear, "only the linear within-clip push is supported (push_linear=true)"
        assert not kernel_normalize, "kernel_normalize is not supported"
        # one row of z is one encoder window of window_size frames; the per-step costs scale with
        # it, distances (push_offset) and row indices (linear_min_gap) do not
        self.window_size = int(window_size)
        assert self.window_size >= 1, "window_size is a frame count per encoder window"
        h = int(hidden)
        # mlp_hidden_layers: number of hidden (Linear+SiLU) blocks; 0 = a pure linear map z = W e
        if int(mlp_hidden_layers) == 0:
            self.mlp = nn.Sequential(nn.Linear(int(embed_dim), int(z_dim)))
        else:
            layers = [nn.Linear(int(embed_dim), h), nn.SiLU()]
            for _ in range(int(mlp_hidden_layers) - 1):
                layers += [nn.Linear(h, h), nn.SiLU()]
            layers += [nn.Linear(h, int(z_dim))]
            self.mlp = nn.Sequential(*layers)
        assert int(z_dim) % int(group_size) == 0, \
            f"z_dim {z_dim} not divisible by IQE group_size {group_size}"
        self.z_dim = int(z_dim)
        self.group_size = int(group_size)
        # alpha = sigmoid(alpha_lr_scale * raw_alpha): the scale moves alpha faster per AdamW step
        # without changing its value at init. Shape (1,) so the optimizer routes both scalars to
        # the no-decay group. alpha_learn=False freezes the mix.
        k = float(alpha_lr_scale)
        assert k > 0, f"alpha_lr_scale must be > 0, got {k}"
        self.alpha_lr_scale = k
        a = min(max(float(alpha_init), 1e-4), 1 - 1e-4)
        self.raw_alpha = nn.Parameter((torch.logit(torch.tensor(a)) / k).reshape(1),
                                      requires_grad=bool(alpha_learn))
        self.raw_scale = nn.Parameter(torch.log(torch.expm1(torch.tensor(float(scale_init)))).reshape(1))
        self.push_offset = float(push_offset)
        self.push_beta = float(push_beta)
        self.push_samples = int(push_samples)
        self.push_within_clip = bool(push_within_clip)
        self.step_cost_per_frame = float(step_cost)
        self.step_cost = float(step_cost) * self.window_size
        self.step_eps = float(step_eps) * self.window_size
        # winsorise each pair's L_step excess before the square (0 = off)
        self.step_clamp = float(step_clamp) * self.window_size
        assert self.step_clamp >= 0.0, "step_clamp must be >= 0 (0 disables)"
        # z_group_norm: per-group RMSNorm, each IQE group at radius sqrt(group_size);
        # z_norm: global RMSNorm to radius sqrt(z_dim)
        self.z_norm = bool(z_norm)
        self.z_group_norm = bool(z_group_norm)
        assert not (self.z_norm and self.z_group_norm), \
            "z_group_norm subsumes z_norm (it fixes the global radius as a corollary); enable one"
        self.dual_lr = float(dual_lr)
        self.push_weight = float(push_weight)
        self.push_linear = True
        # ROW units: 1 drops the self-pairs, 2 additionally drops the adjacent bond L_step owns
        self.linear_min_gap = int(linear_min_gap)
        assert self.linear_min_gap in (1, 2), "linear_min_gap must be 1 (drop self-pairs) or 2"
        self.push_forward_only = bool(push_forward_only)
        self.register_buffer("_lambda", torch.tensor(float(lambda_init)))
        # anti-windup ceiling on the integral dual (None = uncapped)
        self.lambda_max = None if lambda_max is None else float(lambda_max)
        assert self.lambda_max is None or self.lambda_max > 0, "lambda_max must be positive (or None)"
        # leaky dual: lambda <- max(0, (1-rho)*lambda + dual_lr*(L_step - step_eps^2))
        self.dual_leak = float(dual_leak)
        assert 0.0 <= self.dual_leak < 1.0, "dual_leak must be in [0, 1)"
        self.kernel_normalize = False
        self._parts = None                       # (l_push, l_step, lambda) detached, for logging
        self._d_stats = None                     # (mean, std) of the pushed pair distances
        self._near_frac = None                   # share of pairs within one step_cost
        self._z_group_share = None               # top IQE-group energy share (collapse monitor)
        self._z_group_unif = None
        self._z_rms = None
        self._n_pairs = None
        self._dir_stats = None
        self._bal_stats = None

    def _znorm(self, z):
        if self.z_group_norm:
            g = z.unflatten(-1, (-1, self.group_size))
            g = g * (self.group_size ** 0.5) / g.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            return g.flatten(-2)
        if not self.z_norm:
            return z
        return z * (z.shape[-1] ** 0.5) / z.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    def encode_z(self, emb):
        """(B,N,D) frame embeddings -> (B,N,z) quasimetric latents (fp32)."""
        return self._znorm(self.mlp(emb.float()))

    @property
    def alpha(self):
        """The max/mean mix actually used (sigmoid of the scaled raw parameter)."""
        return torch.sigmoid(self.alpha_lr_scale * self.raw_alpha)

    @staticmethod
    def _iqe_components(u, v, group_size):
        """Per-component IQE union lengths: the Lebesgue measure of the union of per-dim
        intervals [u_i, max(u_i, v_i)] over each group. Asymmetric. Returns (..., C)."""
        G = int(group_size)
        x = u.float().unflatten(-1, (-1, G))                     # (...,C,G)
        y = v.float().unflatten(-1, (-1, G))
        D = x.shape[-1]
        valid = x < y
        xy = torch.cat(torch.broadcast_tensors(x, y), dim=-1)    # (...,C,2G)
        sxy, ixy = xy.sort(dim=-1)
        neg_inc = torch.gather(valid, dim=-1, index=ixy % D) * torch.where(ixy < D, -1, 1)
        # prefix sum as a GEMM: exact (summands in {-1,0,1}, partial sums bounded by 2G)
        n = neg_inc.shape[-1]
        M = _prefix_sum_matrix(n, neg_inc.device, torch.float32)
        covered = (neg_inc.float().reshape(-1, n) @ M.T).view(neg_inc.shape) < 0
        neg_f = covered * (-1.0)
        neg_incf = torch.cat([neg_f.narrow(-1, 0, 1), torch.diff(neg_f, dim=-1)], dim=-1)
        return (sxy * neg_incf).sum(-1)                          # (...,C) union length

    def _dist_parts(self, u, v):
        """_dist plus the per-group components it reduces over (one sweep, identical math)."""
        comp = self._iqe_components(u, v, self.group_size)
        alpha = self.alpha
        scale = F.softplus(self.raw_scale)
        return scale * (alpha * comp.amax(-1) + (1.0 - alpha) * comp.mean(-1)), comp

    def _dist(self, u, v):
        """Asymmetric IQE quasimetric d(u,v); u, v broadcast over all but the z dim. d(u,u)=0."""
        return self._dist_parts(u, v)[0]

    def pairwise_dist(self, z, chunk=8):
        """(B,N,z) -> (B,N,N) directed d(i->j), for the detached regression kernel; `chunk` source
        frames at a time so the full (B,N,N,z) grid never materialises."""
        B, N = z.shape[:2]
        out = []
        for i in range(0, N, int(chunk)):
            r = min(int(chunk), N - i)
            out.append(self._dist(z[:, i:i + r, :].unsqueeze(2), z[:, None, :, :]))
        return torch.cat(out, dim=1)

    def pairwise_dist_grad(self, z):
        """(B,N,z) -> (B,N,N) directed d(i->j), gradient-carrying (stacked, not slice-assigned)."""
        B, N = z.shape[:2]
        return torch.stack([self._dist(z[:, i:i + 1, :].expand(B, N, -1), z) for i in range(N)], dim=1)

    def _push_linear(self, z):
        """L_push = -mean(d) over the within-clip grid: |i-j| >= linear_min_gap in both directions,
        or only the forward pairs j - i >= linear_min_gap under push_forward_only. fp32 with
        autocast disabled."""
        B, N, _ = z.shape
        Dm = self.pairwise_dist_grad(z)
        idx = torch.arange(N, device=z.device)
        gap = idx[None, :] - idx[:, None]                 # gap[i, j] = j - i
        keep = (gap >= self.linear_min_gap) if self.push_forward_only \
            else (gap.abs() >= self.linear_min_gap)
        with torch.autocast(device_type=z.device.type, enabled=False):
            l_push = -Dm.float()[:, keep].mean()
        return l_push, Dm.detach()[:, keep]

    def _dual_update(self, l_step):
        """Leaky integral dual; L_step all-reduced so every rank holds the same lambda."""
        if dist.is_available() and dist.is_initialized():
            ls = l_step.detach().clone()
            dist.all_reduce(ls)
            ls = ls / dist.get_world_size()
        else:
            ls = l_step.detach()
        if self.training:
            with torch.no_grad():
                nxt = ((1.0 - self.dual_leak) * self._lambda
                       + self.dual_lr * (ls - self.step_eps ** 2)).clamp_min(0.0)
                if self.lambda_max is not None:
                    nxt = nxt.clamp_max(self.lambda_max)
                self._lambda.copy_(nxt)
        return self._lambda.detach()

    def qrl_loss(self, z):
        """QRL objective on z (B,N,z): push + lambda*step. Flows z -> encoder (not detached);
        updates the lambda buffer internally. Diagnostics kept as 0-dim tensors (no host sync)."""
        B, N = z.shape[:2]
        with torch.no_grad():
            # group-concentration monitor
            _g = z.detach().float().unflatten(-1, (-1, self.group_size)).pow(2).sum(-1)
            _share = _g / _g.sum(-1, keepdim=True).clamp_min(1e-12)
            self._z_group_share = _share.amax(-1).mean()
            self._z_group_unif = 1.0 / _g.shape[-1]
            self._z_rms = z.detach().float().norm(dim=-1).mean() / (z.shape[-1] ** 0.5)
        # L_step: adjacent same-clip transitions, soft d(z_t, z_{t+1}) <= step_cost
        if N >= 2:
            d_step = self._dist(z[:, :-1, :], z[:, 1:, :])
            _ex = (d_step - self.step_cost).clamp_min(0.0)
            if self.step_clamp > 0.0:
                _ex = _ex.clamp_max(self.step_clamp)
            l_step = _ex.square().mean()
        else:
            l_step = z.new_zeros(())
        l_push, d_keep = self._push_linear(z)
        dd = d_keep.reshape(-1)
        self._d_stats = (dd.mean(), dd.std(unbiased=True) if dd.numel() > 1 else dd.new_zeros(()))
        self._near_frac = (dd < self.step_cost).to(dd.dtype).mean()
        self._n_pairs = dd.new_tensor(float(dd.numel()))
        self._dir_stats = None
        self._bal_stats = None
        lam = self._dual_update(l_step)
        self._parts = (l_push.detach(), l_step.detach(), lam)
        return self.push_weight * l_push + lam * l_step
