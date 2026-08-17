from __future__ import annotations

from typing import cast

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from torch.utils.data import Dataset

from hiad.constants import (
    SUPPORTED_TASK_TYPES,
    TASK_TYPE_DYNAMIC_PATCH,
    TASK_TYPE_REFINEMENT_PATCH,
)
from hiad.data import (
    HRImageIndex,
    HRSample,
    MultiResolutionIndex,
    build_multiresolution_region,
    create_dynamic_patch,
    split_multiresolution_regions,
)
from hiad.datasets.patch_dataset import PatchDataset, PatchItem
from hiad.runtime.contracts import TaskInputRecord
from hiad.task.contracts import (
    DynamicPatchTask,
    RefinementPatchTask,
    TaskDefinition,
    ThumbnailTask,
)


class StreamingTaskDataset(Dataset[PatchItem]):
    """按需构建任务补丁，并限制进程内最多缓存一张解码原图。

    Attributes:
        samples (list[HRSample]): 路径唯一的源图样本。
        task (TaskDefinition): 当前粗扫、复核或缩略图任务定义。
        training (bool): 是否为训练数据集。
        regions_by_path (dict[str, list[HRImageIndex]] | None): 复核任务按图像路径
            指定的原图 ``xywh`` 区域；``None`` 表示规则滑窗。
        records (list[TaskInputRecord]): 与每个数据集条目一一对应的坐标、尺寸和
            任务元数据。
    """

    def __init__(
        self,
        samples: list[HRSample],
        task: TaskDefinition,
        *,
        training: bool,
        regions_by_path: dict[str, list[HRImageIndex]] | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(samples, list) or not samples:
            raise ValueError("StreamingTaskDataset requires a non-empty sample list")
        if any(not isinstance(sample, HRSample) for sample in samples):
            raise TypeError("Every streaming sample must be an HRSample")
        if not isinstance(task, dict) or task.get("type") not in SUPPORTED_TASK_TYPES:
            raise ValueError(f"Unsupported task: {task}")
        paths = [sample.image.image_path for sample in samples]
        duplicates = sorted({path for path in paths if paths.count(path) > 1})
        if duplicates:
            raise ValueError(f"Duplicate source image path: {duplicates[0]}")

        self.samples: list[HRSample] = samples
        self.task: TaskDefinition = task
        self.training: bool = training
        self.regions_by_path: dict[str, list[HRImageIndex]] | None = (
            self._validate_regions_by_path(regions_by_path, paths)
        )
        self._entries: list[tuple[int, MultiResolutionIndex | None]] = []
        self.records: list[TaskInputRecord] = []
        # 相邻补丁复用同一解码原图，但绝不跨源图保留多份大图。
        self._cached_path: str | None = None
        self._cached_image: NDArray[np.uint8] | None = None
        self._patch_converter = PatchDataset(
            patches=[],
            training=training,
            task_name=task["name"],
        )
        self._build_index()

    @staticmethod
    def _validate_regions_by_path(
        regions_by_path: dict[str, list[HRImageIndex]] | None,
        paths: list[str],
    ) -> dict[str, list[HRImageIndex]] | None:
        """校验复核区域仅引用当前数据集中的源图。

        Args:
            regions_by_path (dict[str, list[HRImageIndex]] | None): 路径到非空原图
                ``xywh`` 区域列表的映射。
            paths (list[str]): 当前数据集允许引用的源图路径。

        Returns:
            dict[str, list[HRImageIndex]] | None: 复制后的区域映射；未指定时为
            ``None``。

        Raises:
            TypeError: 映射或区域对象类型不正确。
            ValueError: 包含未知源图路径或空区域列表。
        """
        if regions_by_path is None:
            return None
        if not isinstance(regions_by_path, dict):
            raise TypeError("regions_by_path must be a mapping or None")
        unknown_paths = set(regions_by_path) - set(paths)
        if unknown_paths:
            raise ValueError("regions_by_path contains a source image outside this dataset")
        validated: dict[str, list[HRImageIndex]] = {}
        for path, regions in regions_by_path.items():
            if not isinstance(regions, list) or not regions:
                raise ValueError("Each refinement source must have at least one region")
            if any(not isinstance(region, HRImageIndex) for region in regions):
                raise TypeError("Every refinement region must be an HRImageIndex")
            validated[path] = list(regions)
        return validated

    @staticmethod
    def _read_image_size(image_path: str) -> tuple[int, int]:
        """读取图像头并返回正数 ``(width, height)``，不解码像素。

        Args:
            image_path (str): 本地图像文件路径。

        Returns:
            tuple[int, int]: 正数 ``(width, height)``。

        Raises:
            OSError: 文件不存在、不可读或不是 PIL 可识别图像。
            ValueError: 图像头报告非正宽高。
        """
        with Image.open(image_path) as image_file:
            width, height = image_file.size
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid image size for {image_path}: {(width, height)}")
        return int(width), int(height)

    @staticmethod
    def _thumbnail_model_input_size(thumbnail_size: object) -> tuple[int, int]:
        """将缩略图尺寸规范化为正数 ``(width, height)``。

        Args:
            thumbnail_size (object): 正整数边长或两个正整数的宽高序列。

        Returns:
            tuple[int, int]: 规范化的 ``(width, height)``。

        Raises:
            ValueError: 参数不是有效正整数尺寸。
        """
        if isinstance(thumbnail_size, int) and not isinstance(thumbnail_size, bool):
            if thumbnail_size <= 0:
                raise ValueError("thumbnail_size must be positive")
            return thumbnail_size, thumbnail_size
        if (
            isinstance(thumbnail_size, (tuple, list))
            and len(thumbnail_size) == 2
            and all(
                not isinstance(value, bool) and isinstance(value, int) and value > 0
                for value in thumbnail_size
            )
        ):
            return int(thumbnail_size[0]), int(thumbnail_size[1])
        raise ValueError("thumbnail_size must be a positive integer or width-height pair")

    def _build_index(self) -> None:
        """建立数据集条目与可追溯任务输入记录的一一映射。

        滑窗和复核任务记录原图 ``source_xywh`` 以及实际有效的
        ``valid_source_hw``；缩略图任务只记录整图尺寸和模型输入尺寸。
        """
        task = self.task
        task_name = task["name"]
        task_type = task["type"]

        if task_type in {TASK_TYPE_DYNAMIC_PATCH, TASK_TYPE_REFINEMENT_PATCH}:
            patch_task = cast(DynamicPatchTask | RefinementPatchTask, task)
            for sample_index, sample in enumerate(self.samples):
                path = sample.image.image_path
                image_size = self._read_image_size(path)
                image_width, image_height = image_size
                if self.regions_by_path is None:
                    indexes = split_multiresolution_regions(
                        image_size=image_size,
                        patch_size=patch_task["patch_size"],
                        ds_factors=patch_task["ds_factors"],
                        stride=patch_task["stride"],
                    )
                else:
                    indexes = [
                        build_multiresolution_region(
                            image_size,
                            region,
                            patch_task["ds_factors"],
                        )
                        for region in self.regions_by_path.get(path, [])
                    ]
                for region in indexes:
                    source = region.main_index
                    valid_height = min(source.height, image_height - source.y)
                    valid_width = min(source.width, image_width - source.x)
                    self._entries.append((sample_index, region))
                    self.records.append({
                        "task_name": task_name,
                        "task_type": task_type,
                        "image_path": path,
                        "image_size": image_size,
                        "model_input_size": (source.width, source.height),
                        "source_xywh": (source.x, source.y, source.width, source.height),
                        "valid_source_hw": (valid_height, valid_width),
                    })
            return

        thumbnail_task = cast(ThumbnailTask, task)
        model_input_size = self._thumbnail_model_input_size(
            thumbnail_task["thumbnail_size"]
        )
        for sample_index, sample in enumerate(self.samples):
            path = sample.image.image_path
            image_size = self._read_image_size(path)
            self._entries.append((sample_index, None))
            self.records.append({
                "task_name": task_name,
                "task_type": task_type,
                "image_path": path,
                "image_size": image_size,
                "model_input_size": model_input_size,
            })

    def __len__(self) -> int:
        return len(self._entries)

    def source_sample_index(self, index: int) -> int:
        """返回数据集条目所属的源图样本编号。

        Args:
            index (int): 数据集条目索引。

        Returns:
            int: ``samples`` 列表中的源图下标。
        """
        return self._entries[index][0]

    def _load_image(self, sample: HRSample) -> NDArray[np.uint8]:
        """按需解码并缓存一张 HWC ``uint8`` RGB 原图。

        Args:
            sample (HRSample): 需要加载原图的样本。

        Returns:
            NDArray[np.uint8]: ``(height, width, 3)`` RGB 数组。同一路径的相邻
            条目复用同一个只读缓存引用。

        Raises:
            OSError: 图像文件无法读取。
            ValueError: 解码结果不是有限的三通道 ``uint8`` RGB 数组。
        """
        path = sample.image.image_path
        if self._cached_path == path and self._cached_image is not None:
            return self._cached_image
        with Image.open(path) as image_file:
            image_array = np.array(image_file.convert("RGB"), copy=True)
        if (
            not isinstance(image_array, np.ndarray)
            or image_array.dtype != np.uint8
            or image_array.ndim != 3
            or image_array.shape[2] != 3
        ):
            raise ValueError("Source image must be a finite HWC uint8 RGB image")
        self._cached_path = path
        self._cached_image = image_array
        return image_array

    def __getitem__(self, index: int) -> PatchItem:
        sample_index, region = self._entries[index]
        sample = self.samples[sample_index]
        image_array = self._load_image(sample)

        sample.image.image = image_array
        try:
            if sample.mask is not None:
                sample.mask.open()
            if region is None:
                thumbnail_task = cast(ThumbnailTask, self.task)
                patch = sample.down_sampling_to_LR(thumbnail_task["thumbnail_size"])
            else:
                patch = create_dynamic_patch(sample, region)
            return self._patch_converter.transform_patch(patch)
        finally:
            sample.image.image = None
            if sample.mask is not None:
                sample.mask.close()
