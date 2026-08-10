from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, List, Union

import cv2
import numpy as np
import torch
from PIL import Image

from .geometry import HRImageIndex, MultiResolutionIndex

if TYPE_CHECKING:
    from hiad.preprocessing import ForegroundPreprocessor


class HRImage:
    _preprocessor: ClassVar["ForegroundPreprocessor | None"] = None

    def __init__(self, image_path: str, is_mask: bool = False):
        self.image_path = image_path
        self.is_mask = is_mask
        self.image = None
        self._is_processed = False
        self._shared_tensor = None

    @classmethod
    def bind_preprocessor(cls, preprocessor: "ForegroundPreprocessor"):
        if preprocessor is None:
            raise ValueError("preprocessor must not be None")
        if cls._preprocessor is not None and cls._preprocessor is not preprocessor:
            raise RuntimeError("A different ForegroundPreprocessor is already bound")
        cls._preprocessor = preprocessor

    @classmethod
    def clear_preprocessor(cls):
        cls._preprocessor = None

    @property
    def is_processed(self):
        return self._is_processed

    def open(self, category=None):
        if self.image is not None:
            if self.is_mask or self._is_processed:
                return
            raise RuntimeError("RGB image is assigned but has not been preprocessed")
        if self.is_mask:
            with Image.open(self.image_path) as image_file:
                self.image = np.array(image_file.convert("L"), copy=True)
            self._is_processed = False
        else:
            if self._preprocessor is None:
                raise RuntimeError("No ForegroundPreprocessor is bound for RGB images")
            self.image = self._preprocessor.process_file(self.image_path, category)
            self._is_processed = True
        self._shared_tensor = None

    def set_image(self, image, category=None):
        if self.is_mask:
            raise RuntimeError("Mask images cannot be assigned through RGB preprocessing")
        if self._preprocessor is None:
            raise RuntimeError("No ForegroundPreprocessor is bound for RGB images")
        self.image = self._preprocessor.process_array(image, category)
        self._is_processed = True
        self._shared_tensor = None

    def close(self):
        self.image = None
        self._shared_tensor = None
        self._is_processed = False

    def size(self):
        if self.image is None:
            raise RuntimeError("Image must be open before reading its size")
        return self.image.shape[:2][::-1]

    def resize(self, image_size: Union[int, List[int]]):
        if self.image is None:
            raise RuntimeError("Image must be open before resizing")
        if isinstance(image_size, int):
            output_size = (image_size, image_size)
        else:
            if not isinstance(image_size, (tuple, list)) or len(image_size) != 2:
                raise TypeError("image_size must be an integer or width-height pair")
            output_size = tuple(image_size)
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in output_size
        ):
            raise ValueError("image_size dimensions must be positive integers")

        if self.is_mask:
            if self.image.dtype.kind not in "biu":
                raise TypeError("Mask images must use an integer or boolean dtype")
            interpolation = cv2.INTER_NEAREST
        else:
            if not self._is_processed or self.image.dtype != np.float32:
                raise TypeError(
                    "RGB images must be preprocessed float32 arrays before resizing"
                )
            interpolation = cv2.INTER_LINEAR
        return cv2.resize(self.image, output_size, interpolation=interpolation)

    def __getitem__(self, item: HRImageIndex):
        if self.image is None:
            raise RuntimeError("Image must be open before extracting a region")
        if item.x < 0 or item.y < 0 or item.width <= 0 or item.height <= 0:
            raise ValueError(f"Invalid patch geometry: {item}")
        image_height, image_width = self.image.shape[:2]
        if item.x >= image_width or item.y >= image_height:
            raise ValueError(f"Patch origin is outside image bounds: {item}")

        image_patch = self.image[
            item.y:item.y + item.height,
            item.x:item.x + item.width,
        ]
        patch_height, patch_width = image_patch.shape[:2]
        pad_height = item.height - patch_height
        pad_width = item.width - patch_width
        if pad_height < 0 or pad_width < 0:
            raise ValueError(
                f"Invalid patch geometry: requested {item}, got {image_patch.shape}"
            )
        if pad_height or pad_width:
            padding = ((0, pad_height), (0, pad_width))
            if image_patch.ndim == 3:
                padding += ((0, 0),)
            if self.is_mask:
                image_patch = np.pad(
                    image_patch,
                    padding,
                    mode="constant",
                    constant_values=0,
                )
            else:
                image_patch = np.pad(image_patch, padding, mode="edge")
        return image_patch

    def share_memory_(self):
        if self.is_mask:
            raise RuntimeError("Mask images are not shared through the RGB image path")
        if (
            self.image is None
            or not self._is_processed
            or self.image.dtype != np.float32
            or self.image.ndim != 3
            or self.image.shape[2] != 3
            or not self.image.flags.c_contiguous
            or not np.isfinite(self.image).all()
        ):
            raise ValueError(
                "Only processed contiguous finite float32 RGB images can be shared"
            )
        if self._shared_tensor is None:
            shared_tensor = torch.from_numpy(self.image)
            shared_tensor.share_memory_()
            self._shared_tensor = shared_tensor
            self.image = shared_tensor.numpy()
        return self

    def __getstate__(self):
        state = self.__dict__.copy()
        if self._shared_tensor is not None:
            state["image"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._is_processed = state.get("_is_processed", False)
        self._shared_tensor = state.get("_shared_tensor")
        if self._shared_tensor is not None:
            self.image = self._shared_tensor.numpy()


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
            x=int(
                (self.main_index.x - low_resolution_index.x)
                / low_width
                * main_width
            ),
            y=int(
                (self.main_index.y - low_resolution_index.y)
                / low_height
                * main_height
            ),
            width=int(
                self.main_index.width
                / low_resolution_index.width
                * main_width
            ),
            height=int(
                self.main_index.height
                / low_resolution_index.height
                * main_height
            ),
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
        self.image.open(self.clsname)
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


def create_dynamic_patch(
    sample: HRSample,
    index: MultiResolutionIndex,
) -> LRPatch:
    patch = sample[index.main_index]
    if index.low_resolution_indexes is not None:
        for low_resolution_index in index.low_resolution_indexes:
            patch.add_low_resolution_images(low_resolution_index, sample.image)
    return patch
