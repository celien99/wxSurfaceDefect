from __future__ import annotations

import torch
from torch.utils.data import Sampler


class SourceGroupedRandomSampler(Sampler[int]):
    """Sample each source fairly while covering unseen patches before repeats."""

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
        self._remaining: dict[int, list[int]] = {
            group_id: [] for group_id in self._groups
        }

    def _sample_group(self, group_id: int) -> list[int]:
        indexes = self._groups[group_id]
        target_count = min(len(indexes), self.patches_per_source)
        selected = []
        while len(selected) < target_count:
            remaining = self._remaining[group_id]
            if not remaining:
                available = [index for index in indexes if index not in selected]
                order = torch.randperm(
                    len(available),
                    generator=self.generator,
                ).tolist()
                remaining.extend(available[position] for position in order)
            take_count = min(target_count - len(selected), len(remaining))
            selected.extend(remaining[:take_count])
            del remaining[:take_count]
        return selected

    def __iter__(self):
        group_ids = list(self._groups)
        group_order = torch.randperm(len(group_ids), generator=self.generator).tolist()
        for group_position in group_order:
            yield from self._sample_group(group_ids[group_position])

    def __len__(self) -> int:
        return sum(
            min(len(indexes), self.patches_per_source)
            for indexes in self._groups.values()
        )
