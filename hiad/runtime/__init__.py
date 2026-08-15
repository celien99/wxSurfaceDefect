"""Shared detector execution primitives."""

from .evidence import (
    denormalize_imagenet_batch,
    fuse_evidence_tensors,
    high_frequency_map,
)

__all__ = [
    "fuse_evidence_tensors",
    "denormalize_imagenet_batch",
    "high_frequency_map",
]
