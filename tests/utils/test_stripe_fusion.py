import numpy as np
import pytest

from hiad.utils.fusion import CarbonFiberStripeFusion, FusionConfig, FusionResult
from hiad.utils.fusion.phase import (
    PhaseComponents,
    decompose_four_step,
    images_to_unit_range,
    unwrap_phase,
    validate_image_group,
)
from hiad.utils.fusion.responses import (
    detect_dark_spots,
    detect_shape,
    normalized_weights,
    robust_normalize,
)


def test_default_config_uses_orthogonal_phase_axes():
    cfg = FusionConfig()
    assert cfg.x_phase_axis == 1
    assert cfg.y_phase_axis == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"gaussian_kernel": 4},
        {"phase_gradient_kernel": 2},
        {"input_max": 0.0},
        {"percentile_low": 80.0, "percentile_high": 20.0},
        {"x_phase_axis": 1, "y_phase_axis": 1},
        {"weight_phase": -0.1},
        {
            "weight_phase": 0.0,
            "weight_scratch": 0.0,
            "weight_dark": 0.0,
            "weight_texture": 0.0,
        },
        {"soft_threshold": 1.5},
        {"saturation_onset": -0.1},
    ],
)
def test_invalid_config_raises(kwargs):
    with pytest.raises(ValueError):
        FusionConfig(**kwargs)


def _phase_shift_stack(
    height: int,
    width: int,
    axis: int,
    background: float = 0.4,
    modulation: float = 0.2,
    dtype=np.uint8,
):
    coords = np.linspace(0.0, 4 * np.pi, width if axis == 1 else height, dtype=np.float32)
    if axis == 1:
        phi = np.broadcast_to(coords[None, :], (height, width))
    else:
        phi = np.broadcast_to(coords[:, None], (height, width))
    images = []
    for step in range(4):
        values = background + modulation * np.cos(phi + step * np.pi / 2)
        if dtype == np.uint8:
            images.append(np.clip(values * 255.0, 0, 255).astype(np.uint8))
        else:
            images.append(np.clip(values * 65535.0, 0, 65535).astype(np.uint16))
    return images, phi.astype(np.float32)


def test_validate_rejects_wrong_count_and_nan():
    good = [np.zeros((8, 8), dtype=np.uint8) for _ in range(4)]
    with pytest.raises(ValueError, match="exactly 4"):
        validate_image_group(good[:3], "x_images")
    rgb = [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(4)]
    with pytest.raises(ValueError, match="single-channel"):
        validate_image_group(rgb, "x_images")
    mismatched = list(good)
    mismatched[1] = np.zeros((7, 8), dtype=np.uint8)
    with pytest.raises(ValueError, match="shape"):
        validate_image_group(mismatched, "x_images")
    bad = list(good)
    bad[1] = np.full((8, 8), np.nan, dtype=np.float32)
    with pytest.raises(ValueError, match="NaN|Inf|finite"):
        validate_image_group(bad, "x_images")


def test_uint8_and_uint16_scale_to_unit_range():
    uint8_images = [np.full((4, 4), 255, dtype=np.uint8) for _ in range(4)]
    uint16_images = [np.full((4, 4), 4095, dtype=np.uint16) for _ in range(4)]
    unit8 = images_to_unit_range(uint8_images, None)
    unit12 = images_to_unit_range(uint16_images, 4095.0)
    assert np.allclose(unit8[0], 1.0)
    assert np.allclose(unit12[0], 1.0)


def test_four_step_recovers_background_modulation_and_phase():
    images, phi = _phase_shift_stack(16, 32, axis=1)
    components = decompose_four_step(images, FusionConfig(gaussian_sigma=0.0))
    assert np.allclose(components.background, 0.4, atol=0.02)
    assert np.allclose(components.modulation, 0.2, atol=0.02)
    recovered = np.unwrap(components.phase, axis=1)
    offset = np.median(recovered - phi)
    assert np.allclose(recovered - offset, phi, atol=0.15)


def test_unsaturated_bright_region_keeps_validity():
    images, _ = _phase_shift_stack(16, 16, axis=1, background=0.7, modulation=0.2)
    components = decompose_four_step(images, FusionConfig(gaussian_sigma=0.0))
    assert float(components.saturation_validity.min()) > 0.95


def test_near_full_scale_pixels_reduce_validity():
    images, _ = _phase_shift_stack(16, 16, axis=1, background=0.4, modulation=0.2)
    images[0] = images[0].copy()
    images[0][4:8, 4:8] = 255
    components = decompose_four_step(images, FusionConfig(gaussian_sigma=0.0))
    assert float(components.saturation_validity[6, 6]) < 0.2
    assert float(components.saturation_validity[0, 0]) > 0.95


def test_unwrap_uses_configured_axis():
    phase = np.linspace(-np.pi, np.pi, 20, dtype=np.float32)
    wrapped = np.broadcast_to(np.angle(np.exp(1j * phase))[None, :], (8, 20)).copy()
    confidence = np.ones_like(wrapped)
    unwrapped = unwrap_phase(
        wrapped,
        axis=1,
        confidence=confidence,
        config=FusionConfig(),
    )
    diffs = np.diff(unwrapped[0])
    assert np.all(diffs > -0.5)


def test_low_dynamic_range_normalizes_to_zero():
    noise = np.full((32, 32), 0.1, dtype=np.float32)
    noise += np.linspace(0.0, 2e-5, 32, dtype=np.float32)[None, :]
    result = robust_normalize(noise, FusionConfig(min_dynamic_range=1e-3))
    assert np.allclose(result, 0.0)


def test_normalized_weights_sum_to_one():
    weights = normalized_weights(FusionConfig())
    assert weights.shape == (4,)
    assert np.isclose(weights.sum(), 1.0)
    assert np.all(weights > 0)


def test_shape_gradient_follows_configured_axis():
    images, _ = _phase_shift_stack(24, 32, axis=1)
    x_dec = decompose_four_step(images, FusionConfig(gaussian_sigma=0.0))
    y_images, _ = _phase_shift_stack(24, 32, axis=0)
    y_dec = decompose_four_step(y_images, FusionConfig(gaussian_sigma=0.0))
    shape = detect_shape(x_dec, y_dec, FusionConfig(gaussian_sigma=0.0))
    assert shape.shape == (24, 32)
    assert shape.dtype == np.float32
    assert float(shape.max()) > 0.0


def test_dark_spots_use_float_median_without_uint8_quantize():
    background = np.full((31, 31), 0.6, dtype=np.float32)
    background[12:19, 12:19] = 0.2

    def _pack(bg: np.ndarray) -> PhaseComponents:
        return PhaseComponents(
            phase=np.zeros_like(bg),
            modulation=np.ones_like(bg) * 0.2,
            background=bg,
            phase_confidence=np.ones_like(bg),
            saturation_validity=np.ones_like(bg),
        )

    cfg = FusionConfig(dark_background_kernel=15, min_dynamic_range=1e-6)
    dark = detect_dark_spots(_pack(background), _pack(background), cfg)
    assert dark.dtype == np.float32
    assert float(dark[15, 15]) > float(dark[0, 0])


def _eight_images():
    x_images, _ = _phase_shift_stack(24, 32, axis=1)
    y_images, _ = _phase_shift_stack(24, 32, axis=0)
    y_images[0] = y_images[0].copy()
    y_images[0][10:13, 8:20] = np.clip(
        y_images[0][10:13, 8:20].astype(np.int16) - 40,
        0,
        255,
    ).astype(np.uint8)
    return x_images, y_images


def test_fuse_returns_uint8_map_with_stable_shape():
    fusion = CarbonFiberStripeFusion(FusionConfig(gaussian_sigma=0.0))
    x_images, y_images = _eight_images()
    fused = fusion.fuse(x_images, y_images)
    again = fusion.fuse(x_images, y_images)
    assert fused.shape == (24, 32)
    assert fused.dtype == np.uint8
    assert int(fused.min()) >= 0 and int(fused.max()) <= 255
    assert np.array_equal(fused, again)


def test_analyze_matches_fuse_and_exposes_channels():
    fusion = CarbonFiberStripeFusion(FusionConfig(gaussian_sigma=0.0))
    x_images, y_images = _eight_images()
    result = fusion.analyze(x_images, y_images)
    fused = fusion.fuse(x_images, y_images)
    assert isinstance(result, FusionResult)
    assert np.array_equal(result.fused, fused)
    for name in (
        "shape",
        "scratch",
        "dark",
        "texture",
        "phase_confidence",
        "saturation_validity",
    ):
        array = getattr(result, name)
        assert array.shape == (24, 32)
        assert array.dtype == np.float32
