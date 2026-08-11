from __future__ import annotations

import torch
from torch.utils.data import Sampler


class SourceGroupedRandomSampler(Sampler[int]):
    """Sample a bounded random patch set for every source image each epoch."""

    def __init__(
        self,
        dataset,
        *,
        patches_per_source: int,
        generator: torch.Generator | None = None,
    ) -> None:
        if not hasattr(dataset, "source_sample_index"):
            raise TypeError("dataset must expose source_sample_index(index)")
        if (
            isinstance(patches_per_source, bool)
            or not isinstance(patches_per_source, int)
            or patches_per_source <= 0
        ):
            raise ValueError("patches_per_source must be a positive integer")
        self.dataset = dataset
        self.patches_per_source = patches_per_source
        self.generator = generator
        self._groups: dict[int, list[int]] = {}
        for index in range(len(self.dataset)):
            self._groups.setdefault(self.dataset.source_sample_index(index), []).append(index)

    def __iter__(self):
        group_ids = list(self._groups)
        group_order = torch.randperm(len(group_ids), generator=self.generator).tolist()
        for group_position in group_order:
            indexes = self._groups[group_ids[group_position]]
            patch_order = torch.randperm(len(indexes), generator=self.generator).tolist()
            for patch_position in patch_order[: self.patches_per_source]:
                yield indexes[patch_position]

    def __len__(self) -> int:
        return sum(
            min(len(indexes), self.patches_per_source)
            for indexes in self._groups.values()
        )
