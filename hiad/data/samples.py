from __future__ import annotations

import hashlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image

from .geometry import HRImageIndex, MultiResolutionIndex


UInt8Array: TypeAlias = NDArray[np.uint8]
ImageSizeInput: TypeAlias = int | tuple[int, int] | list[int]


class HRImage:
    """管理按需加载的 RGB 原图或单通道掩码。

    Attributes:
        image_path (str): 本地图像文件路径。
        is_mask (bool): 是否按单通道掩码解码；否则强制转换为 RGB。
        image (UInt8Array | None): 已解码数组。业务图像为 ``(H, W, 3)`` RGB
            ``uint8``，掩码为 ``(H, W)`` ``uint8``；关闭时为 ``None``。
    """

    def __init__(self, image_path: str, is_mask: bool = False) -> None:
        self.image_path: str = image_path
        self.is_mask: bool = is_mask
        self.image: UInt8Array | None = None

    def open(self) -> None:
        """使用 PIL 解码，业务图像始终转为 RGB，掩码转为单通道。"""
        if self.is_mask:
            with Image.open(self.image_path) as image_file:
                self.image = np.array(image_file.convert("L"), copy=True)
        else:
            with Image.open(self.image_path) as image_file:
                self.image = np.array(image_file.convert("RGB"), copy=True)

    def close(self) -> None:
        """释放当前解码数组，保留文件路径以便后续重新打开。"""
        self.image = None

    def size(self) -> tuple[int, int]:
        """返回 OpenCV/PIL 通用的 ``(width, height)`` 尺寸。"""
        if self.image is None:
            raise RuntimeError("Image must be open before reading its size")
        height, width = self.image.shape[:2]
        return int(width), int(height)

    def resize(self, image_size: ImageSizeInput) -> UInt8Array:
        """将已打开图像缩放到模型或显示尺寸。

        Args:
            image_size (ImageSizeInput): 目标尺寸；整数表示正方形，二元序列按
                OpenCV 的 ``(width, height)`` 顺序解释。

        Returns:
            UInt8Array: 缩放后的数组。RGB 图像保持 ``(H, W, 3)``，掩码保持
            ``(H, W)``；掩码使用最近邻插值，业务图像使用线性插值。

        Raises:
            RuntimeError: 图像尚未通过 :meth:`open` 解码。
            TypeError: ``image_size`` 不是整数或二元宽高序列。
        """
        if self.image is None:
            raise RuntimeError("Image must be open before resizing")
        if isinstance(image_size, int):
            output_size = (image_size, image_size)
        elif isinstance(image_size, (tuple, list)) and len(image_size) == 2:
            output_size = (int(image_size[0]), int(image_size[1]))
        else:
            raise TypeError("image_size must be an integer or width-height pair")
        interpolation = cv2.INTER_NEAREST if self.is_mask else cv2.INTER_LINEAR
        return cv2.resize(self.image, output_size, interpolation=interpolation)

    def __getitem__(self, item: HRImageIndex) -> UInt8Array:
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
            padding = (
                ((0, pad_height), (0, pad_width), (0, 0))
                if patch.ndim == 3
                else ((0, pad_height), (0, pad_width))
            )
            patch = np.pad(
                patch,
                padding,
                mode="constant" if self.is_mask else "edge",
            )
        return np.array(patch, copy=True)


@dataclass
class LRPatch:
    """保存送入模型的补丁、标签及其原图空间上下文。

    Attributes:
        image (UInt8Array): ``(H, W, 3)`` RGB 补丁，通常为 ``uint8``。
        mask (UInt8Array | None): 与主补丁同高宽的单通道缺陷掩码。
        label (int | None): 图像级标签，``0`` 表示正常，``1`` 表示异常。
        label_name (str | None): 便于报告展示的标签名称。
        clsname (str | None): 业务类别名称。
        main_index (HRImageIndex | None): 主补丁在原图中的 ``xywh`` 坐标。
        valid_source_hw (tuple[int, int] | None): 边界填充前的有效源区域
            ``(height, width)``。
        low_resolution_images (list[UInt8Array] | None): 已缩放到主补丁尺寸的
            多尺度 RGB 上下文图像。
        low_resolution_indexes (list[HRImageIndex] | None): 主补丁在各上下文
            图像坐标系中的映射区域，与上下文图像逐项对应。
    """

    image: UInt8Array
    mask: UInt8Array | None = None
    label: int | None = None
    label_name: str | None = None
    clsname: str | None = None
    main_index: HRImageIndex | None = None
    valid_source_hw: tuple[int, int] | None = None
    low_resolution_images: list[UInt8Array] | None = None
    low_resolution_indexes: list[HRImageIndex] | None = None

    def add_low_resolution_images(
        self,
        low_resolution_index: HRImageIndex,
        image: HRImage,
    ) -> None:
        """提取上下文图像并记录主补丁在缩放后上下文中的位置。

        Args:
            low_resolution_index (HRImageIndex): 上下文在原图中的 ``xywh`` 区域。
            image (HRImage): 已打开的 RGB 原图。

        Raises:
            RuntimeError: 当前补丁没有 ``main_index``，无法建立坐标映射。
        """
        if self.main_index is None:
            raise RuntimeError("Main patch index is required before adding context images")
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
    """聚合一张原图及其可选前景、缺陷掩码和业务标签。

    Attributes:
        image (HRImage): RGB 原图；提供前景掩码时可能指向缓存后的合成图。
        foreground (HRImage | None): 单通道有效前景掩码。
        mask (HRImage | None): 单通道缺陷真值掩码。
        label (int | None): 图像级正常/异常标签。
        label_name (str | None): 标签的展示名称。
        clsname (str | None): 样本所属业务类别。
    """

    def __init__(
        self,
        image: str | HRImage,
        mask: str | HRImage | None = None,
        label: int | None = None,
        label_name: str | None = None,
        clsname: str | None = None,
        foreground: str | HRImage | None = None,
    ) -> None:
        self.image: HRImage = (
            HRImage(image, is_mask=False) if isinstance(image, str) else image
        )
        self.foreground: HRImage | None = (
            HRImage(foreground, is_mask=True)
            if isinstance(foreground, str)
            else foreground
        )
        self.image = self._apply_foreground(self.image, self.foreground)
        self.mask: HRImage | None = HRImage(mask, is_mask=True) if isinstance(mask, str) else mask
        self.label: int | None = label
        self.label_name: str | None = label_name
        self.clsname: str | None = clsname

    @staticmethod
    def _apply_foreground(
        image: HRImage,
        foreground: str | HRImage | None,
    ) -> HRImage:
        """生成可复用的前景合成图；缓存键同时绑定原图和掩码版本。

        OpenCV 在内部按 BGR 读取和写入缓存，缓存文件后续仍由
        :class:`HRImage` 通过 PIL 转为 RGB，因此不会改变模型的颜色约定。
        文件不可用、解码失败或 OpenCV 处理失败时返回原始图像对象。

        Args:
            image (HRImage): 待应用前景的业务原图。
            foreground (str | HRImage | None): 单通道前景掩码路径或图像对象。

        Returns:
            HRImage: 指向合成缓存的 RGB 图像；无法合成时返回 ``image``。
        """
        if not isinstance(image, HRImage) or foreground is None:
            return image
        foreground_path = (
            foreground.image_path if isinstance(foreground, HRImage) else foreground
        )
        if not isinstance(image.image_path, str) or not isinstance(foreground_path, str):
            return image

        try:
            source_path = Path(image.image_path).expanduser().resolve()
            mask_path = Path(foreground_path).expanduser().resolve()
            source_stat = source_path.stat()
            mask_stat = mask_path.stat()
            cache_key = "|".join(
                (
                    os.fspath(source_path),
                    str(source_stat.st_mtime_ns),
                    str(source_stat.st_size),
                    os.fspath(mask_path),
                    str(mask_stat.st_mtime_ns),
                    str(mask_stat.st_size),
                )
            )
            cache_path = (
                Path(tempfile.gettempdir())
                / "hiad_foreground_cache"
                / f"{hashlib.sha256(cache_key.encode()).hexdigest()}.png"
            )
            if cache_path.is_file():
                return HRImage(os.fspath(cache_path), is_mask=False)

            source_image = cv2.imread(os.fspath(source_path), cv2.IMREAD_COLOR)
            foreground_image = cv2.imread(os.fspath(mask_path), cv2.IMREAD_GRAYSCALE)
            if source_image is None or foreground_image is None:
                return image

            source_height, source_width = source_image.shape[:2]
            if foreground_image.shape != (source_height, source_width):
                foreground_image = cv2.resize(
                    foreground_image,
                    (source_width, source_height),
                    interpolation=cv2.INTER_NEAREST,
                )

            cache_path.parent.mkdir(parents=True, exist_ok=True)
            # OpenCV 在此处按 BGR 读写；缓存再次由 PIL 打开时统一转换为 RGB。
            clean_image = cv2.bitwise_and(
                source_image,
                source_image,
                mask=foreground_image,
            )
            temporary_path = cache_path.with_name(
                f".{cache_path.stem}.{os.getpid()}.tmp.png"
            )
            try:
                if not cv2.imwrite(os.fspath(temporary_path), clean_image):
                    return image
                os.replace(temporary_path, cache_path)
            finally:
                if temporary_path.exists():
                    temporary_path.unlink()
            return HRImage(os.fspath(cache_path), is_mask=False)
        except (OSError, cv2.error):
            return image

    def __getitem__(self, item: HRImageIndex) -> LRPatch:
        if self.image.image is None:
            raise RuntimeError("Sample image must be open before extracting a patch")
        return LRPatch(
            image=self.image[item],
            mask=self.mask[item] if self.mask is not None else None,
            clsname=self.clsname,
            label=self.label,
            label_name=self.label_name,
            main_index=item,
        )

    def open(self) -> None:
        """解码原图以及当前样本附带的前景和缺陷掩码。"""
        self.image.open()
        if self.foreground is not None:
            self.foreground.open()
        if self.mask is not None:
            self.mask.open()

    def close(self) -> None:
        """释放原图、前景掩码和缺陷掩码的内存数组。"""
        self.image.close()
        if self.foreground is not None:
            self.foreground.close()
        if self.mask is not None:
            self.mask.close()

    def down_sampling_to_LR(self, image_size: ImageSizeInput) -> LRPatch:
        """按模型尺寸生成整图缩略输入，同时保持标签契约不变。

        Args:
            image_size (ImageSizeInput): 模型输入尺寸；整数表示正方形，二元序列
                按 ``(width, height)`` 解释。

        Returns:
            LRPatch: 缩放后的 RGB 整图及同尺寸掩码、类别和图像级标签。
        """
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
    """从已打开的原图提取主补丁，并记录边界填充前的有效区域。

    Args:
        sample (HRSample): 原图已经解码到内存的样本。
        index (MultiResolutionIndex): 主补丁及可选上下文的原图 ``xywh`` 索引。

    Returns:
        LRPatch: RGB 主补丁、可选掩码、上下文及原图坐标元数据。

    Raises:
        RuntimeError: ``sample`` 的原图尚未打开。
    """
    source_image = sample.image.image
    if source_image is None:
        raise RuntimeError("Sample image must be open before creating a dynamic patch")
    patch = sample[index.main_index]
    patch.valid_source_hw = (
        min(index.main_index.height, source_image.shape[0] - index.main_index.y),
        min(index.main_index.width, source_image.shape[1] - index.main_index.x),
    )
    if index.low_resolution_indexes is not None:
        for low_resolution_index in index.low_resolution_indexes:
            patch.add_low_resolution_images(low_resolution_index, sample.image)
    return patch
