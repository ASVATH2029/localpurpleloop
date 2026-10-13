"""Synthetic event generation: attack families and benign background traffic.

Nothing here touches a network or a cloud credential -- every event is built
in-process from a seeded numpy Generator.
"""
from .attacks import generate_variant, sample_knobs, seed_variant_ids
from .benign import generate_benign, BENIGN_KINDS

__all__ = [
    "generate_variant",
    "sample_knobs",
    "seed_variant_ids",
    "generate_benign",
    "BENIGN_KINDS",
]
