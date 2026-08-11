import torch

from hiad.datasets.samplers import SourceGroupedRandomSampler


class _SourceDataset:
    def __init__(self, source_indexes):
        self.source_indexes = source_indexes

    def __len__(self):
        return len(self.source_indexes)

    def source_sample_index(self, index):
        return self.source_indexes[index]


def test_sampler_keeps_every_source_image_in_each_epoch():
    dataset = _SourceDataset([0, 0, 0, 1, 1, 2])
    sampler = SourceGroupedRandomSampler(
        dataset,
        patches_per_source=2,
        generator=torch.Generator().manual_seed(42),
    )

    indexes = list(sampler)

    assert len(indexes) == len(sampler) == 5
    assert {dataset.source_sample_index(index) for index in indexes} == {0, 1, 2}
    assert sum(dataset.source_sample_index(index) == 0 for index in indexes) == 2
    assert sum(dataset.source_sample_index(index) == 1 for index in indexes) == 2
    assert sum(dataset.source_sample_index(index) == 2 for index in indexes) == 1


def test_sampler_covers_source_patches_before_repeating():
    dataset = _SourceDataset([0] * 16)
    sampler = SourceGroupedRandomSampler(
        dataset,
        patches_per_source=4,
        generator=torch.Generator().manual_seed(42),
    )

    epochs = [list(sampler) for _ in range(4)]

    assert all(len(set(epoch)) == 4 for epoch in epochs)
    assert set().union(*map(set, epochs)) == set(range(16))


def test_sampler_uses_every_patch_when_budget_covers_the_group():
    dataset = _SourceDataset([0, 0, 0, 1, 1])
    sampler = SourceGroupedRandomSampler(
        dataset,
        patches_per_source=16,
        generator=torch.Generator().manual_seed(42),
    )

    indexes = list(sampler)

    assert len(indexes) == len(dataset)
    assert set(indexes) == set(range(len(dataset)))
