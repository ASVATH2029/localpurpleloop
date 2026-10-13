"""Generalisation test harness: score brand-new variants before training on them."""
from .heldout import assert_unseen, heldout_eval

__all__ = ["assert_unseen", "heldout_eval"]
