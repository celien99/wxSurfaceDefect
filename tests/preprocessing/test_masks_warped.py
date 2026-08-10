import cv2
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
        validate_warped_mask(empty, ref, config, affine_abs_det=1.0)


def test_validate_warped_mask_accepts_uniform_scale():
    """Area gate must use |det(A)| so a pure 2x scale warp is not rejected."""
    reference_mask = np.zeros((40, 40), dtype=bool)
    reference_mask[10:30, 10:30] = True  # 20x20 = 400 px
    scale = 2.0
    affine = np.array([[scale, 0.0, 0.0], [0.0, scale, 0.0]], dtype=np.float64)
    warped = cv2.warpAffine(
        reference_mask.view(np.uint8),
        affine,
        (80, 80),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(bool)
    abs_det = float(abs(np.linalg.det(affine[:, :2])))
    assert abs_det == pytest.approx(4.0)
    # Without scale normalization, warped/ref ~ 4 would fail this gate.
    config = {
        "boundary_expand_ratio": 0.0,
        "min_reference_coverage": 0.9,
        "max_area_ratio_deviation": 0.15,
    }
    cleaned, metrics = validate_warped_mask(
        warped,
        reference_mask,
        config,
        affine_abs_det=abs_det,
    )
    assert cleaned.any()
    assert metrics["area_ratio_deviation"] <= config["max_area_ratio_deviation"]
    assert metrics["reference_coverage"] >= config["min_reference_coverage"]
