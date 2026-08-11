from __future__ import annotations

import torch
from torch.utils.data import Sampler


class SourceGroupedRandomSampler(Sampler[int]):
    """Shuffle source images and their patches while keeping each image contiguous."""

    def __init__(self, dataset, *, generator: torch.Generator | None = None) -> None:
        if not hasattr(dataset, "source_sample_index"):
            raise TypeError("dataset must expose source_sample_index(index)")
        self.dataset = dataset
        self.generator = generator

    def __iter__(self):
        groups: dict[int, list[int]] = {}
        for index in range(len(self.dataset)):
            groups.setdefault(self.dataset.source_sample_index(index), []).append(index)

        group_ids = list(groups)
        group_order = torch.randperm(len(group_ids), generator=self.generator).tolist()
        for group_position in group_order:
            indexes = groups[group_ids[group_position]]
            patch_order = torch.randperm(len(indexes), generator=self.generator).tolist()
            for patch_position in patch_order:
                yield indexes[patch_position]

    def __len__(self) -> int:
        return len(self.dataset)
