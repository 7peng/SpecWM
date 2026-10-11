"""The frozen quasimetric teacher and the affinity transform of its distances."""
from .frozen_teacher import FrozenQRLTeacher
from .kernel_targets import distance_affinity

__all__ = ["FrozenQRLTeacher", "distance_affinity"]
