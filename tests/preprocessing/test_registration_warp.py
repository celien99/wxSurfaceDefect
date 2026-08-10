from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from hiad.preprocessing.masks import MaskRejected
from hiad.preprocessing.registration import register_and_warp_mask


def test_register_and_warp_mask_rejects_empty_matches(monkeypatch):
    rgb = np.zeros((32, 32, 3), dtype=np.uint8)
    device = torch.device("cpu")

    def fake_extract(encoder, rgb, config, device, working_longest_edge):
        feats = F.normalize(torch.ones(4, 8), dim=1)
        centers = np.array(
            [[8.0, 8.0], [24.0, 8.0], [8.0, 24.0], [24.0, 24.0]],
            dtype=np.float32,
        )
        return feats, centers, (2, 2), (32, 32)

    monkeypatch.setattr(
        "hiad.preprocessing.registration.extract_dino_grid",
        fake_extract,
    )

    template = {
        "working_longest_edge": 32,
        "foreground_cells": torch.tensor([True, True, False, False]),
        "features": F.normalize(torch.ones(4, 8), dim=1),
        "centers_xy": torch.tensor(
            [[8.0, 8.0], [24.0, 8.0], [8.0, 24.0], [24.0, 24.0]],
        ),
    }
    # Background prototype aligns with features; foreground is opposite → no FG matches.
    prototypes = {
        "foreground": F.normalize(-torch.ones(8), dim=0),
        "background": F.normalize(torch.ones(8), dim=0),
    }
    config = {"min_dino_matches": 4}
    reference_mask = np.ones((32, 32), dtype=bool)

    with pytest.raises(MaskRejected) as exc_info:
        register_and_warp_mask(
            rgb,
            encoder=MagicMock(),
            prototypes=prototypes,
            template=template,
            reference_mask=reference_mask,
            config=config,
            device=device,
        )

    message = str(exc_info.value).lower()
    assert "insufficient" in message or "reason" in message or "match" in message
