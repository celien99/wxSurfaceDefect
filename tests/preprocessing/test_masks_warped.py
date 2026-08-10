import numpy as np
import pytest

from hiad.preprocessing.masks import MaskRejected, validate_warped_mask


def test_validate_warped_mask_rejects_empty():
    empty = np.zeros((32, 32), dtype=bool)
    ref = np.ones((32, 32), dtype=bool)
    config = {
        "boundary_expand_ratio": 0.0,
        "min_reference_coverage": 1.0,
        "max_area_ratio_deviation": 0.35,
    }
    with pytest.raises(MaskRejected):
        validate_warped_mask(empty, ref, config)
