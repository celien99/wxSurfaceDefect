import json
from dataclasses import dataclass
from typing import List, Union


@dataclass
class HRImageIndex:
    x: int
    y: int
    width: int
    height: int

    def __str__(self):
        return json.dumps(self.to_dict())

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        return isinstance(other, HRImageIndex) and self.to_dict() == other.to_dict()

    def to_dict(self):
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
        }

    @staticmethod
    def from_str(value):
        data = json.loads(value)
        return HRImageIndex(
            x=data["x"],
            y=data["y"],
            width=data["width"],
            height=data["height"],
        )


class MultiResolutionIndex:
    def __init__(
        self,
        main_index: HRImageIndex,
        low_resolution_indexes: List[HRImageIndex] = None,
    ):
        self.main_index = main_index
        self.low_resolution_indexes = low_resolution_indexes

    def add_low_resolution_index(self, candidate_indexes):
        main_x_end = self.main_index.x + self.main_index.width
        main_y_end = self.main_index.y + self.main_index.height
        for index in candidate_indexes:
            x_end = index.x + index.width
            y_end = index.y + index.height
            if (
                self.main_index.x >= index.x
                and self.main_index.y >= index.y
                and main_x_end <= x_end
                and main_y_end <= y_end
            ):
                if self.low_resolution_indexes is None:
                    self.low_resolution_indexes = []
                self.low_resolution_indexes.append(index)
                return True
        return False

    def __str__(self):
        data = {
            "main_index": self.main_index.to_dict(),
            "low_resolution_indexes": (
                [index.to_dict() for index in self.low_resolution_indexes]
                if self.low_resolution_indexes is not None
                else None
            ),
        }
        return json.dumps(data)

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        return isinstance(other, MultiResolutionIndex) and str(self) == str(other)


def split_multiresolution_regions(
    image_size: Union[int, List],
    patch_size: Union[int, List],
    ds_factors: List[int] = None,
    stride: Union[int, List] = None,
) -> List[MultiResolutionIndex]:
    """Build nested source regions in native ``(width, height)`` coordinates."""
    if ds_factors is None:
        ds_factors = [0]

    scale_factors = [2 ** factor for factor in sorted(ds_factors)]
    main_factor = scale_factors[0]
    if isinstance(patch_size, int):
        patch_size = [patch_size, patch_size]
    if stride is not None and isinstance(stride, int):
        stride = [stride, stride]

    main_patch_size = [value * main_factor for value in patch_size]
    main_stride = None if stride is None else [value * main_factor for value in stride]
    main_indexes = split_image_regions(image_size, main_patch_size, main_stride)
    indexes = [MultiResolutionIndex(main_index=index) for index in main_indexes]

    for factor in scale_factors[1:]:
        scaled_patch_size = [value * factor for value in patch_size]
        scaled_stride = None if stride is None else [value * factor for value in stride]
        low_resolution_indexes = split_image_regions(
            image_size,
            scaled_patch_size,
            scaled_stride,
        )
        for index in indexes:
            if not index.add_low_resolution_index(low_resolution_indexes):
                raise RuntimeError(
                    f"No enclosing region found for source index {index.main_index}"
                )
    return indexes


def build_multiresolution_region(
    image_size: Union[int, List],
    main_index: HRImageIndex,
    ds_factors: List[int],
) -> MultiResolutionIndex:
    """Build centered, boundary-aligned context regions around one native tile."""
    if not isinstance(main_index, HRImageIndex):
        raise TypeError("main_index must be an HRImageIndex")
    if not ds_factors or ds_factors[0] != 0 or ds_factors != sorted(set(ds_factors)):
        raise ValueError("ds_factors must be unique, sorted, and start with 0")
    if isinstance(image_size, int):
        image_width = image_height = image_size
    else:
        image_width, image_height = image_size
    contexts = []
    center_x = main_index.x + main_index.width / 2
    center_y = main_index.y + main_index.height / 2
    for factor in ds_factors[1:]:
        scale = 2 ** factor
        width = main_index.width * scale
        height = main_index.height * scale
        x = min(max(int(center_x - width / 2), 0), max(image_width - width, 0))
        y = min(max(int(center_y - height / 2), 0), max(image_height - height, 0))
        contexts.append(HRImageIndex(x=x, y=y, width=width, height=height))
    return MultiResolutionIndex(
        main_index=main_index,
        low_resolution_indexes=contexts or None,
    )


def split_image_regions(
    image_size: Union[int, List],
    patch_size: Union[int, List],
    stride: Union[int, List] = None,
) -> List[HRImageIndex]:
    """Split native ``(width, height)`` geometry into edge-aligned regions."""

    def extract_starts(axis_size, region_size, axis_stride):
        if axis_size <= region_size:
            return [0]
        starts = list(range(0, axis_size, axis_stride))
        for index, start in enumerate(starts):
            if start + region_size > axis_size:
                starts[index] = axis_size - region_size
        return list(dict.fromkeys(starts))

    if isinstance(image_size, int):
        image_width, image_height = image_size, image_size
    else:
        image_width, image_height = image_size
    if isinstance(patch_size, int):
        patch_width, patch_height = patch_size, patch_size
    else:
        patch_width, patch_height = patch_size
    if stride is None:
        stride_width, stride_height = patch_width, patch_height
    elif isinstance(stride, int):
        stride_width, stride_height = stride, stride
    else:
        stride_width, stride_height = stride

    y_starts = extract_starts(image_height, patch_height, stride_height)
    x_starts = extract_starts(image_width, patch_width, stride_width)
    return [
        HRImageIndex(x=x, y=y, width=patch_width, height=patch_height)
        for y in y_starts
        for x in x_starts
    ]
