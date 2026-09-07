from __future__ import annotations

import json
from collections.abc import Sequence
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
