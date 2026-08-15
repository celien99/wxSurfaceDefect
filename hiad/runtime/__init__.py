"""Shared detector execution primitives."""

from .evidence import (
    fuse_evidence_maps,
    fuse_evidence_tensors,
    high_frequency_map,
)

__all__ = [
    "fuse_evidence_maps",
    "fuse_evidence_tensors",
    "high_frequency_map",
]
