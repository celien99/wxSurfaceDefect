import pytest
import torch

from hiad.models.memory import NormalFeatureMemory


def test_normal_feature_memory_scores_training_prototype_lower_than_shifted_feature():
    memory = NormalFeatureMemory(embed_dim=4, layers=2)
    normal = [torch.zeros(2, 4, 3, 3), torch.zeros(2, 4, 2, 2)]
    memory.update(normal)
    memory.update([value + 0.01 for value in normal])

    normal_score = memory.score([value + 0.005 for value in normal])
    shifted_score = memory.score([value + 2.0 for value in normal])

    assert all(
        float(reference.mean()) < float(anomaly.mean())
        for reference, anomaly in zip(normal_score, shifted_score)
    )


def test_normal_feature_memory_requires_fitted_statistics():
    memory = NormalFeatureMemory(embed_dim=4, layers=1)

    with pytest.raises(RuntimeError, match="not fitted"):
        memory.score([torch.zeros(1, 4, 2, 2)])
