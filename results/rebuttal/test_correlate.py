"""Known-answer checks for correlate.py: the Spearman helper against scipy, the embedding distances against
brute force, and the bootstrap interval's coverage of a planted rank correlation.
    python results/rebuttal/test_correlate.py"""
import sys
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, str(Path(__file__).resolve().parent))
from correlate import model_dist, spearman  # noqa: E402

rng = np.random.default_rng(0)

x, y = rng.normal(size=5000), rng.normal(size=5000)
y2 = np.round(x * 3 + rng.normal(size=5000))                      # heavy ties
for a, b in [(x, y), (x, y2), (x, x ** 3)]:
    assert abs(spearman(a, b) - spearmanr(a, b)[0]) < 1e-9
print("spearman matches scipy, ties included")

Z = rng.normal(size=(300, 16)).astype(np.float32); A = np.array([0, 5, 17])
Zn = Z / np.linalg.norm(Z, axis=1, keepdims=True)
cos_bf = np.array([[1 - Zn[i] @ Zn[j] for j in range(len(Z))] for i in A])
l2_bf = np.array([[np.linalg.norm(Z[i] - Z[j]) for j in range(len(Z))] for i in A])
assert np.allclose(model_dist(Z, A, "cos"), cos_bf, atol=1e-5) and np.allclose(model_dist(Z, A, "l2"), l2_bf, atol=1e-3)
print("model_dist matches brute-force cosine and L2")

n = 200000; truth = rng.exponential(100, n)
for noise, label in [(0.0, "exact"), (None, "pure noise")]:
    emb = truth if noise == 0.0 else rng.normal(size=n)
    r = spearman(np.abs(emb), truth)
    assert (label == "exact" and r > 0.999) or (label == "pure noise" and abs(r) < 0.01), (label, r)
    print(f"{label}: rho {r:+.4f}")
# planted Gaussian-copula correlation r = 0.5 -> Spearman 6/pi * asin(r/2)
z1 = rng.normal(size=n); z2 = 0.5 * z1 + np.sqrt(1 - 0.25) * rng.normal(size=n)
r = spearman(z1, z2); expect = 6 / np.pi * np.arcsin(0.25)
assert abs(r - expect) < 0.01, (r, expect)
print(f"planted correlation: rho {r:.4f} vs analytic {expect:.4f}")

# bootstrap over groups: the 95% interval should cover the population value about 95% of the time
cover, trials = 0, 200
for _ in range(trials):
    g = rng.integers(0, 100, 4000); e = rng.normal(size=100)[g]
    a = rng.normal(size=4000) + e; b = 0.5 * a + rng.normal(size=4000) + e
    eps_u, inv = np.unique(g, return_inverse=True)
    by = np.split(np.argsort(inv, kind="stable"), np.cumsum(np.bincount(inv))[:-1])
    bs = [spearman(a[w], b[w]) for w in (np.concatenate([by[k] for k in rng.integers(0, len(eps_u), len(eps_u))]) for _ in range(200))]
    big_g = rng.integers(0, 100000, 400000); big_e = rng.normal(size=100000)[big_g]
    pa = rng.normal(size=400000) + big_e; pb = 0.5 * pa + rng.normal(size=400000) + big_e
    pop = spearman(pa, pb)
    cover += np.quantile(bs, .025) <= pop <= np.quantile(bs, .975)
print(f"grouped bootstrap 95% interval coverage: {cover / trials:.0%} (expect about 95%)")
