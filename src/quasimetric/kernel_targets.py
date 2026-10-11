"""Transforms of frozen within-clip quasimetric targets."""
import torch


def distance_affinity(distance, sigma, symmetrize="none"):
    """exp(-distance / sigma), optionally averaged over the two directions (symmetrize="mean",
    which takes unsymmetrised distances)."""
    assert float(sigma) > 0
    assert symmetrize in ("none", "mean")
    affinity = torch.exp(-distance / float(sigma))
    if symmetrize == "mean":
        affinity = 0.5 * (affinity + affinity.transpose(-1, -2))
    return affinity
