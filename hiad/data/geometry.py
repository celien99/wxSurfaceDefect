from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TypeAlias


SizeLike: TypeAlias = int | Sequence[int]


@dataclass
class HRImageIndex:
    """描述原图像素坐标系中的矩形区域。

    坐标采用左上角为原点的 ``(x, y, width, height)`` 格式；``x`` 对应列，
    ``y`` 对应行，宽高均以像素为单位。

    Attributes:
        x (int): 区域左上角的水平像素坐标。
        y (int): 区域左上角的垂直像素坐标。
        width (int): 区域宽度，单位为像素。
        height (int): 区域高度，单位为像素。
    """

    x: int
    y: int
    width: int
    height: int

    def __str__(self) -> str:
        return json.dumps(self.to_dict())

    def __hash__(self) -> int:
        return hash(str(self))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, HRImageIndex) and self.to_dict() == other.to_dict()

    def to_dict(self) -> dict[str, int]:
        """转换为可直接写入 JSON 的坐标字典。

        Returns:
            dict[str, int]: 包含 ``x``、``y``、``width`` 和 ``height`` 的字典。
        """
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
        }

    @staticmethod
    def from_str(value: str) -> HRImageIndex:
        """从 JSON 字符串恢复原图区域。

        Args:
            value (str): 由 :meth:`to_dict` 对应结构序列化得到的 JSON 字符串。

        Returns:
            HRImageIndex: 根据 JSON 中四个字段构造的区域对象；本方法只读取字段，
                不额外校验字段值是否为整数。

        Raises:
            json.JSONDecodeError: ``value`` 不是合法 JSON。
            KeyError: JSON 对象缺少任一必需坐标字段。
        Notes:
            调用方应确保四个字段都是整数像素值；错误字段类型可能在后续坐标
            运算或序列化流程中才暴露。
        """
        data = json.loads(value)
        return HRImageIndex(
            x=data["x"],
            y=data["y"],
            width=data["width"],
            height=data["height"],
        )


class MultiResolutionIndex:
    """关联一个主补丁及其逐级扩大的原图上下文区域。

    Attributes:
        main_index (HRImageIndex): 实际参与局部检测的主补丁区域。
        low_resolution_indexes (list[HRImageIndex] | None): 包含主补丁的上下文
            区域，列表顺序与下采样层级顺序一致；没有上下文时为 ``None``。
    """

    def __init__(
        self,
        main_index: HRImageIndex,
        low_resolution_indexes: list[HRImageIndex] | None = None,
    ) -> None:
        self.main_index: HRImageIndex = main_index
        self.low_resolution_indexes: list[HRImageIndex] | None = low_resolution_indexes

    def add_low_resolution_index(
        self,
        candidate_indexes: Iterable[HRImageIndex],
    ) -> bool:
        """追加第一个能够完整包围主补丁的上下文区域。

        Args:
            candidate_indexes (Iterable[HRImageIndex]): 待检查的原图 ``xywh``
                区域，按调用方期望的优先级排列。

        Returns:
            bool: 找到并追加包围区域时为 ``True``，否则为 ``False``。
        """
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

    def __str__(self) -> str:
        data = {
            "main_index": self.main_index.to_dict(),
            "low_resolution_indexes": (
                [index.to_dict() for index in self.low_resolution_indexes]
                if self.low_resolution_indexes is not None
                else None
            ),
        }
        return json.dumps(data)

    def __hash__(self) -> int:
        return hash(str(self))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, MultiResolutionIndex) and str(self) == str(other)


def split_multiresolution_regions(
    image_size: SizeLike,
    patch_size: SizeLike,
    ds_factors: list[int] | None = None,
    stride: SizeLike | None = None,
) -> list[MultiResolutionIndex]:
    """在原图坐标系中构建覆盖整图的多尺度补丁索引。

    Args:
        image_size (SizeLike): 原图尺寸；整数表示正方形，否则按
            ``(width, height)`` 解释。
        patch_size (SizeLike): 基础模型补丁尺寸，顺序同样为
            ``(width, height)``。
        ds_factors (list[int] | None): 二次幂尺度指数。最小指数对应主补丁，
            后续指数对应需要包围主补丁的上下文区域；默认仅使用指数 ``0``。
        stride (SizeLike | None): 基础滑窗步长；``None`` 表示使用补丁尺寸。

    Returns:
        list[MultiResolutionIndex]: 按先行后列顺序排列的主补丁及上下文索引。

    Raises:
        RuntimeError: 任一主补丁在某个尺度下找不到完整包围它的上下文区域。
    """
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
    image_size: SizeLike,
    main_index: HRImageIndex,
    ds_factors: list[int],
) -> MultiResolutionIndex:
    """围绕指定主补丁构建居中且贴合原图边界的上下文区域。

    Args:
        image_size (SizeLike): 原图 ``(width, height)``；整数表示正方形。
        main_index (HRImageIndex): 需要补充上下文的原图 ``xywh`` 主区域。
        ds_factors (list[int]): 唯一且升序的二次幂尺度指数，必须从 ``0`` 开始。

    Returns:
        MultiResolutionIndex: 主补丁及按 ``ds_factors`` 顺序生成的上下文区域。

    Raises:
        TypeError: ``main_index`` 不是 :class:`HRImageIndex`。
        ValueError: ``ds_factors`` 为空、重复、乱序或不是从 ``0`` 开始。
    """
    if not isinstance(main_index, HRImageIndex):
        raise TypeError("main_index must be an HRImageIndex")
    if not ds_factors or ds_factors[0] != 0 or ds_factors != sorted(set(ds_factors)):
        raise ValueError("ds_factors must be unique, sorted, and start with 0")
    if isinstance(image_size, int):
        image_width = image_height = image_size
    else:
        image_width, image_height = image_size
    contexts: list[HRImageIndex] = []
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
    image_size: SizeLike,
    patch_size: SizeLike,
    stride: SizeLike | None = None,
) -> list[HRImageIndex]:
    """将原图范围切分为覆盖边界的像素坐标补丁。

    Args:
        image_size (SizeLike): 原图尺寸；整数表示正方形，否则为
            ``(width, height)``。
        patch_size (SizeLike): 输出补丁尺寸，按 ``(width, height)`` 解释。
        stride (SizeLike | None): 横纵滑窗步长；``None`` 表示无重叠切分。

    Returns:
        list[HRImageIndex]: 按从上到下、从左到右顺序排列的原图 ``xywh``
        区域。末端窗口向边界回退以确保覆盖整图。

    Notes:
        本函数假设尺寸和步长都是正整数，不会替调用方补做完整参数校验；零或
        负步长、非正补丁尺寸可能触发底层 ``range`` 异常或产生无效区域。
    """

    def extract_starts(axis_size: int, region_size: int, axis_stride: int) -> list[int]:
        """计算单轴上去重且与末端边界对齐的窗口起点。

        Args:
            axis_size (int): 原图当前轴的像素长度。
            region_size (int): 补丁在当前轴的像素长度。
            axis_stride (int): 当前轴的滑窗步长。

        Returns:
            list[int]: 升序起点；区域大于图像时仅返回 ``0``，末个窗口回退到边界。
        """
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


def build_grid_contexts(
    image_size: SizeLike,
    main_indexes: Sequence[HRImageIndex],
    cell_size: int = 1024,
) -> tuple[list[HRImageIndex], list[MultiResolutionIndex]]:
    """为 context 复用构建网格对齐的 cell 布局与主补丁→cell 映射。

    现状每个主补丁独立编码其居中 context，相邻补丁的 context 大量重叠、
    重复编码（spec 2026-08-27 旗舰 B）。本函数把 context 换成覆盖整图的
    网格 cell：每个 cell 只需编码一次，主补丁复用"包含其中心"的 cell 特征
    切片。cell 尺寸对应粗扫 ``ds_factors=[0,1]`` 的 2×patch context 原始
    尺寸（默认 1024）。

    Args:
        image_size (SizeLike): 原图尺寸；整数表示正方形，否则按
            ``(width, height)`` 解释。
        main_indexes (Sequence[HRImageIndex]): 粗扫主补丁区域，与
            ``split_image_regions`` 输出顺序一致。
        cell_size (int): 正方形网格 cell 边长（像素）。

    Returns:
        tuple[list[HRImageIndex], list[MultiResolutionIndex]]: 覆盖整图且
        ``cell_size`` 整数倍对齐、末端回退边界的去重网格 cell 列表，以及与
        ``main_indexes`` 一一对应的 ``MultiResolutionIndex``——每个主补丁的
        ``low_resolution_indexes`` 是包含其中心的那个 cell。

    Raises:
        ValueError: ``cell_size`` 不是正整数。
        RuntimeError: 某个主补丁中心落在所有网格 cell 之外（不应发生，
            cell 布局覆盖整图）。
    """
    if not isinstance(cell_size, int) or cell_size <= 0:
        raise ValueError("cell_size must be a positive integer")
    cells = split_image_regions(image_size, cell_size, stride=cell_size)
    results: list[MultiResolutionIndex] = []
    for main in main_indexes:
        center_x = main.x + main.width // 2
        center_y = main.y + main.height // 2
        enclosing: HRImageIndex | None = None
        for cell in cells:
            if (
                cell.x <= center_x < cell.x + cell.width
                and cell.y <= center_y < cell.y + cell.height
            ):
                enclosing = cell
                break
        if enclosing is None:
            raise RuntimeError(f"No grid cell encloses source index {main}")
        results.append(
            MultiResolutionIndex(main_index=main, low_resolution_indexes=[enclosing])
        )
    return cells, results
