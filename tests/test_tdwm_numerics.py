"""What is computed in the autocast dtype: the cosine Gram of the kernel term must be an fp32
matmul (a bf16 one rounds the centred Gram away), and `_factor_kernel` must return fp32 under bf16
autocast for every form, matching an fp64 reference; plus the cos_sin wrap invariance and the
indicator form."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import train_spec as T  # noqa: E402

torch.manual_seed(0)


def centred(S):
    return S - S.mean(1, keepdim=True) - S.mean(0, keepdim=True) + S.mean()


def test_gram_matmul_must_be_fp32():
    for target_cos in (0.5, 0.81, 0.95, 0.99):
        n, d = 128, 192
        mu = torch.randn(d)
        mu = mu / mu.norm()
        a = float(np.sqrt(max(target_cos, 1e-6)))
        h = torch.nn.functional.normalize(a * mu + (1 - a) * torch.randn(n, d) / d ** 0.5, dim=-1)
        C64 = centred(h.double() @ h.double().t())
        with torch.autocast("cpu", dtype=torch.bfloat16):
            S_bf = h.float() @ h.float().t()
            with torch.autocast("cpu", enabled=False):
                S_32 = h.float() @ h.float().t()
        assert S_bf.dtype == torch.bfloat16 and S_32.dtype == torch.float32
        rel_bf = float((centred(S_bf.float().double()) - C64).norm() / C64.norm())
        rel_32 = float((centred(S_32.double()) - C64).norm() / C64.norm())
        assert rel_32 < 0.1 * max(rel_bf, 1e-12) or rel_32 < 1e-4, (rel_bf, rel_32)


def test_factor_kernel_forms_are_fp32_under_autocast():
    X = torch.randn(64, 5)
    X64 = X.double()
    Xn = torch.nn.functional.normalize(X64, dim=-1)
    refs = {"cos": torch.exp(-(1.0 - Xn @ Xn.t()) / 0.5), "laplace": torch.exp(-torch.cdist(X64, X64) / 0.5)}
    for form, ref in refs.items():
        K = T._factor_kernel(form, X, 0.5, {})
        assert K.dtype == torch.float32 and float((K.double() - ref).abs().max()) < 1e-4, form
    # laplace guards autocast on the tensor's device; cos guards CUDA autocast (the training device)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        Kb = T._factor_kernel("laplace", X, 0.5, {})
    assert Kb.dtype == torch.float32 and float((Kb.double() - refs["laplace"]).abs().max()) < 1e-4
    if torch.cuda.is_available():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            Kc = T._factor_kernel("cos", X.cuda(), 0.5, {}).cpu()
        assert Kc.dtype == torch.float32 and float((Kc.double() - refs["cos"]).abs().max()) < 1e-4


def test_factor_kernel_invariances():
    th = torch.randn(32, 3)
    k1 = T._factor_kernel("laplace", th, 0.7, {"embed": "cos_sin"})
    k2 = T._factor_kernel("laplace", th + 2 * np.pi * torch.randint(-2, 3, (32, 3)).float(), 0.7,
                          {"embed": "cos_sin"})
    assert float((k1 - k2).abs().max()) < 1e-4
    ii = torch.randint(0, 3, (16, 2)).float()
    Ki = T._factor_kernel("indicator", ii, 1.0, {})
    assert bool((Ki == ((ii[:, None] == ii[None, :]).all(-1)).float()).all())
    try:
        T._factor_kernel("gauss", ii, 1.0, {})
    except AssertionError:
        return
    raise AssertionError("an unknown factor form was accepted")


if __name__ == "__main__":
    test_gram_matmul_must_be_fp32()
    test_factor_kernel_forms_are_fp32_under_autocast()
    test_factor_kernel_invariances()
    print("test_tdwm_numerics: OK")
