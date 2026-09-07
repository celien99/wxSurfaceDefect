from __future__ import annotations

from collections.abc import Iterator, Sized
from typing import Protocol

import torch
from torch.utils.data import Sampler


class SourceIndexedDataset(Sized, Protocol):
    """支持把数据集条目追溯到原始样本编号的最小协议。"""

    def source_sample_index(self, index: int) -> int:
        """返回指定数据集条目所属的源样本编号。

        Args:
            index (int): 数据集条目索引。

        Returns:
            int: 可用于分组采样的稳定源样本编号。
        """
        ...


class SourceGroupedRandomSampler(Sampler[int]):
    """按源图公平采样，并在重复前优先覆盖尚未使用的补丁。

    Attributes:
        dataset (SourceIndexedDataset): 可将补丁索引映射到源图编号的数据集。
        patches_per_source (int): 每轮从每张源图最多采样的补丁数。
        generator (torch.Generator | None): 控制源图和补丁乱序的随机生成器。
    """

    def __init__(
        self,
        dataset: SourceIndexedDataset,
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
        self.dataset: SourceIndexedDataset = dataset
        self.patches_per_source: int = patches_per_source
        self.generator: torch.Generator | None = generator
        self._groups: dict[int, list[int]] = {}
        for index in range(len(self.dataset)):
            self._groups.setdefault(self.dataset.source_sample_index(index), []).append(index)
        self._remaining: dict[int, list[int]] = {
            group_id: [] for group_id in self._groups
        }

    def _sample_group(self, group_id: int) -> list[int]:
        """从一个源图组中抽样，优先消费上一轮未使用的补丁。

        Args:
            group_id (int): :meth:`source_sample_index` 返回的源图编号。

        Returns:
            list[int]: 不重复的数据集条目索引，数量不超过
            ``patches_per_source``。
        """
        indexes = self._groups[group_id]
        target_count = min(len(indexes), self.patches_per_source)
        selected: list[int] = []
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

    def __iter__(self) -> Iterator[int]:
        group_ids = list(self._groups)
        group_order = torch.randperm(len(group_ids), generator=self.generator).tolist()
        for group_position in group_order:
            yield from self._sample_group(group_ids[group_position])

    def __len__(self) -> int:
        return sum(
            min(len(indexes), self.patches_per_source)
            for indexes in self._groups.values()
        )
