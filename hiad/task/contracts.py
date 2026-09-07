from __future__ import annotations

from typing import Literal, TypeAlias, TypedDict


class DynamicPatchTask(TypedDict):
    """原图坐标系下的动态多尺度粗扫任务。

    Attributes:
        name (Literal["dynamic_patch"]): 稳定任务名称。
        type (Literal["dynamic_patch"]): 供数据集分派的任务判别字段。
        patch_size (int): 正方形模型输入边长，单位为像素。
        stride (int | None): 原图滑窗步长；``None`` 表示等于补丁边长。
        ds_factors (list[int]): 从 ``0`` 开始的二次幂上下文尺度指数。
    """

    name: Literal["dynamic_patch"]
    type: Literal["dynamic_patch"]
    patch_size: int
    stride: int | None
    ds_factors: list[int]


class RefinementPatchTask(TypedDict):
    """对粗扫候选区域执行高分辨率复核的任务。

    Attributes:
        name (Literal["refinement_patch"]): 稳定任务名称。
        type (Literal["refinement_patch"]): 供数据集分派的任务判别字段。
        patch_size (int): 正方形复核补丁边长，单位为原图像素。
        stride (int): 复核任务步长，当前与 ``patch_size`` 相同。
        ds_factors (list[int]): 与粗扫一致的多尺度上下文指数。
        refinement_quantile (float): 从路由异常图选择候选像素的分位数。
        refinement_min_area (int): 候选区域必须达到的最小像素数。
        refinement_safety_fraction (float): 确定性安全采样块占全部网格的比例。
    """

    name: Literal["refinement_patch"]
    type: Literal["refinement_patch"]
    patch_size: int
    stride: int
    ds_factors: list[int]
    refinement_quantile: float
    refinement_min_area: int
    refinement_safety_fraction: float


class ThumbnailTask(TypedDict):
    """只提供全局路由与图像级先验的缩略图任务。

    Attributes:
        name (Literal["thumbnail"]): 稳定任务名称。
        type (Literal["thumbnail"]): 供数据集分派的任务判别字段。
        thumbnail_size (int): 正方形整图模型输入边长，单位为像素。
    """

    name: Literal["thumbnail"]
    type: Literal["thumbnail"]
    thumbnail_size: int


TaskDefinition: TypeAlias = DynamicPatchTask | RefinementPatchTask | ThumbnailTask
