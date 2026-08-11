import cv2
import numpy as np

from hiad.data import HRSample


def test_hrsample_uses_optional_foreground_to_clean_image(tmp_path):
    source_path = tmp_path / "source.png"
    foreground_path = tmp_path / "foreground.png"
    defect_mask_path = tmp_path / "defect.png"
    source = np.array(
        [
            [[10, 20, 30], [40, 50, 60], [70, 80, 90]],
            [[15, 25, 35], [45, 55, 65], [75, 85, 95]],
        ],
        dtype=np.uint8,
    )
    foreground = np.array([[255, 0]], dtype=np.uint8)
    defect_mask = np.array([[0, 255, 0], [0, 0, 0]], dtype=np.uint8)
    assert cv2.imwrite(str(source_path), source)
    assert cv2.imwrite(str(foreground_path), foreground)
    assert cv2.imwrite(str(defect_mask_path), defect_mask)

    sample = HRSample(
        str(source_path),
        foreground=str(foreground_path),
        mask=str(defect_mask_path),
    )

    clean_image = cv2.imread(sample.image.image_path, cv2.IMREAD_COLOR)
    expected_foreground = cv2.resize(
        foreground,
        (source.shape[1], source.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    expected = cv2.bitwise_and(source, source, mask=expected_foreground)
    assert sample.image.image_path != str(source_path)
    assert clean_image.shape == source.shape
    assert np.array_equal(clean_image, expected)
    assert sample.mask.image_path == str(defect_mask_path)
    assert HRSample(
        str(source_path),
        foreground=str(foreground_path),
    ).image.image_path == sample.image.image_path


def test_hrsample_keeps_original_image_when_foreground_cannot_be_read(tmp_path):
    source_path = tmp_path / "source.png"
    source = np.full((2, 2, 3), 127, dtype=np.uint8)
    assert cv2.imwrite(str(source_path), source)

    sample = HRSample(
        str(source_path),
        foreground=str(tmp_path / "missing_foreground.png"),
    )

    assert sample.image.image_path == str(source_path)
