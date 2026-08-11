from dataclasses import dataclass
from typing import List, Union

import cv2
import numpy as np
from PIL import Image

from .geometry import HRImageIndex, MultiResolutionIndex


class HRImage:
    def __init__(self, image_path: str, is_mask: bool = False):
        self.image_path = image_path
        self.is_mask = is_mask
        self.image = None

    def open(self):
        if self.is_mask:
            with Image.open(self.image_path) as image_file:
                self.image = np.array(image_file.convert("L"), copy=True)
        else:
            with Image.open(self.image_path) as image_file:
                self.image = np.array(image_file.convert("RGB"), copy=True)

    def close(self):
        self.image = None

    def size(self):
        if self.image is None:
            raise RuntimeError("Image must be open before reading its size")
        return self.image.shape[:2][::-1]

    def resize(self, image_size: Union[int, List[int]]):
        if self.image is None:
            raise RuntimeError("Image must be open before resizing")
        if isinstance(image_size, int):
            output_size = (image_size, image_size)
        elif isinstance(image_size, (tuple, list)) and len(image_size) == 2:
            output_size = tuple(image_size)
        else:
            raise TypeError("image_size must be an integer or width-height pair")
        interpolation = cv2.INTER_NEAREST if self.is_mask else cv2.INTER_LINEAR
        return cv2.resize(self.image, output_size, interpolation=interpolation)

    def __getitem__(self, item: HRImageIndex):
        if self.image is None:
            raise RuntimeError("Image must be open before extracting a region")
        if item.x < 0 or item.y < 0 or item.width <= 0 or item.height <= 0:
            raise ValueError(f"Invalid patch geometry: {item}")

        image_height, image_width = self.image.shape[:2]
        if item.x >= image_width or item.y >= image_height:
            raise ValueError(f"Patch origin is outside image bounds: {item}")

        patch = self.image[item.y:item.y + item.height, item.x:item.x + item.width]
        pad_height = item.height - patch.shape[0]
        pad_width = item.width - patch.shape[1]
        if pad_height < 0 or pad_width < 0:
            raise ValueError(f"Requested patch is larger than its source crop: {item}")
        if pad_height or pad_width:
            padding = ((0, pad_height), (0, pad_width))
            if patch.ndim == 3:
                padding += ((0, 0),)
            patch = np.pad(
                patch,
                padding,
                mode="constant" if self.is_mask else "edge",
            )
        return np.array(patch, copy=True)


@dataclass
class LRPatch:
    image: np.ndarray
    mask: np.ndarray = None
    label: int = None
    label_name: str = None
    clsname: str = None
    main_index: HRImageIndex = None
    valid_source_hw: tuple[int, int] = None
    low_resolution_images: List[np.ndarray] = None
    low_resolution_indexes: List[HRImageIndex] = None

    def add_low_resolution_images(
        self,
        low_resolution_index: HRImageIndex,
        image: HRImage,
    ):
        main_height, main_width = self.image.shape[:2]
        low_resolution_image = image[low_resolution_index]
        low_height, low_width = low_resolution_image.shape[:2]
        low_resolution_image = cv2.resize(
            low_resolution_image,
            (main_width, main_height),
        )
        mapped_index = HRImageIndex(
            x=int((self.main_index.x - low_resolution_index.x) / low_width * main_width),
            y=int((self.main_index.y - low_resolution_index.y) / low_height * main_height),
            width=int(self.main_index.width / low_resolution_index.width * main_width),
            height=int(self.main_index.height / low_resolution_index.height * main_height),
        )
        if self.low_resolution_indexes is None:
            self.low_resolution_indexes = []
        if self.low_resolution_images is None:
            self.low_resolution_images = []
        self.low_resolution_images.append(low_resolution_image)
        self.low_resolution_indexes.append(mapped_index)


class HRSample:
    def __init__(
        self,
        image: Union[str, HRImage],
        mask: Union[str, HRImage] = None,
        label: int = None,
        label_name: str = None,
        clsname: str = None,
    ):
        self.image = HRImage(image, is_mask=False) if isinstance(image, str) else image
        self.mask = HRImage(mask, is_mask=True) if isinstance(mask, str) else mask
        self.label = label
        self.label_name = label_name
        self.clsname = clsname

    def __getitem__(self, item: HRImageIndex) -> LRPatch:
        if self.image.image is None:
            raise RuntimeError("Sample image must be open before extracting a patch")
        values = {"image": self.image[item], "main_index": item}
        if self.mask is not None:
            values["mask"] = self.mask[item]
        if self.clsname is not None:
            values["clsname"] = self.clsname
        if self.label is not None:
            values["label"] = self.label
        if self.label_name is not None:
            values["label_name"] = self.label_name
        return LRPatch(**values)

    def open(self):
        self.image.open()
        if self.mask is not None:
            self.mask.open()

    def close(self):
        self.image.close()
        if self.mask is not None:
            self.mask.close()

    def down_sampling_to_LR(self, image_size: Union[int, List[int]]) -> LRPatch:
        if self.image.image is None:
            self.open()
        return LRPatch(
            image=self.image.resize(image_size),
            mask=self.mask.resize(image_size) if self.mask is not None else None,
            label=self.label,
            clsname=self.clsname,
            label_name=self.label_name,
        )


def create_dynamic_patch(sample: HRSample, index: MultiResolutionIndex) -> LRPatch:
    patch = sample[index.main_index]
    patch.valid_source_hw = (
        min(index.main_index.height, sample.image.image.shape[0] - index.main_index.y),
        min(index.main_index.width, sample.image.image.shape[1] - index.main_index.x),
    )
    if index.low_resolution_indexes is not None:
        for low_resolution_index in index.low_resolution_indexes:
            patch.add_low_resolution_images(low_resolution_index, sample.image)
    return patch
