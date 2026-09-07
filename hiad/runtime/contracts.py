from __future__ import annotations

from typing import Literal, TypeAlias, TypedDict

import numpy as np
from numpy.typing import NDArray

from hiad.constants import TaskType

FloatMap: TypeAlias = NDArray[np.float32]
ScoreVector: TypeAlias = NDArray[np.float32]
ImageArray: TypeAlias = NDArray[np.uint8]
BinaryMask: TypeAlias = NDArray[np.uint8]
ImageSize: TypeAlias = tuple[int, int]
DecisionState: TypeAlias = Literal["OK", "NG"]


class DetectorPrediction(TypedDict):
    """检测器对单个模型输入返回的异常证据。

    Attributes:
        anomaly_map (FloatMap): 模型输入分辨率下的二维 ``float32`` 异常图。
        score (float): 该输入的图像级异常分数，数值越大表示越异常。
    """

    anomaly_map: FloatMap
    score: float


class TaskInputRecordRequired(TypedDict):
    """每个模型输入都必须携带的任务追溯字段。

    Attributes:
        task_name (str): 产生该输入的任务名称。
        task_type (TaskType): 粗扫补丁、复核补丁或缩略图任务类型。
        image_path (str): 输入所属的原图路径。
        image_size (ImageSize): 原图 ``(width, height)``。
        model_input_size (ImageSize): 模型输入 ``(width, height)``。
    """

    task_name: str
    task_type: TaskType
    image_path: str
    image_size: ImageSize
    model_input_size: ImageSize


class TaskInputRecord(TaskInputRecordRequired, total=False):
    """流式数据集为补丁回填原图保留的可选坐标元数据。

    Attributes:
        source_xywh (tuple[int, int, int, int]): 补丁在原图中的
            ``(x, y, width, height)`` 像素坐标。
        valid_source_hw (tuple[int, int]): 边界填充前有效内容的
            ``(height, width)``。
    """

    source_xywh: tuple[int, int, int, int]
    valid_source_hw: tuple[int, int]


class ImageQualityResult(TypedDict):
    """采集质量门禁的可落盘结果。

    Attributes:
        status (Literal["PASS", "FAIL"]): 图像采集质量是否通过。
        reasons (list[str]): 触发质量失败的稳定机器可读原因码。
        mean_luminance (float): 有效区域内归一化灰度均值。
        clipped_fraction (float): 有效区域内接近全黑或全白的像素比例。
        focus_variance (float): 有效区域内 Laplacian 响应方差。
    """

    status: Literal["PASS", "FAIL"]
    reasons: list[str]
    mean_luminance: float
    clipped_fraction: float
    focus_variance: float


class StrongestComponent(TypedDict):
    """最终判定使用的最强八连通异常区域。

    Attributes:
        area (int): 连通区域像素数。
        area_fraction (float): 区域面积占整张异常图的比例。
        mean_score (float): 区域内异常分数均值。
        max_score (float): 区域内异常分数最大值。
        score (float): 均值与面积奖励组合后的组件分数。
        bbox_xywh (list[int]): 区域外接框 ``[x, y, width, height]``。
    """

    area: int
    area_fraction: float
    mean_score: float
    max_score: float
    score: float
    bbox_xywh: list[int]


class ComponentStatistics(TypedDict):
    """单张异常图的紧凑连通区域摘要。

    Attributes:
        component_count (int): 超过像素阈值的八连通区域数量。
        anomalous_pixel_count (int): 所有异常区域的像素总数。
        largest_component_area (int): 最大连通区域的像素数。
        strongest_component (StrongestComponent | None): 组件分数最高的区域。
    """

    component_count: int
    anomalous_pixel_count: int
    largest_component_area: int
    strongest_component: StrongestComponent | None


class RefinementStatistics(TypedDict):
    """Runtime coverage and cost proxy for one refinement grid.

    Attributes:
        total_tiles (int): Number of native refinement tiles covering the image.
        selected_tiles (int): Number of unique tiles sent to the high-resolution task.
        coverage_ratio (float): ``selected_tiles / total_tiles``.
    """

    total_tiles: int
    selected_tiles: int
    coverage_ratio: float


class InferenceTiming(TypedDict):
    """Wall-clock timings for one inference call, in seconds."""

    quality_seconds: float
    coarse_seconds: float
    routing_seconds: float
    refinement_seconds: float
    postprocess_seconds: float
    total_seconds: float


class CalibrationCategoryRequired(TypedDict):
    """单个业务类别必须具备的正常样本校准字段。

    Attributes:
        normal_image_count (int): 参与该类别校准的正常原图数量。
        threshold (float): 该类别的图像级正常分数阈值。
        pixel_threshold (float): 该类别的像素异常阈值。
    """

    normal_image_count: int
    threshold: float
    pixel_threshold: float


class ScoreCalibrationCategory(CalibrationCategoryRequired, total=False):
    """单个类别的图像、像素及可选连通组件阈值。

    Attributes:
        component_threshold (float): 完成第二阶段校准后使用的连通组件阈值。
    """

    component_threshold: float


class ScoreCalibrationRequired(TypedDict):
    """从正常样本建立的全局与分类别校准契约。

    Attributes:
        percentile (float): 图像分数校准分位数。
        pixel_percentile (float): 单图异常图压缩使用的像素分位数。
        pixel_image_percentile (float): 跨正常图像计算像素阈值的分位数。
        normal_image_count (int): 参与全局校准的正常原图数量。
        global_threshold (float): 未知类别回退使用的图像阈值。
        global_pixel_threshold (float): 未知类别回退使用的像素阈值。
        categories (dict[str, ScoreCalibrationCategory]): 类别名称到分类别阈值的映射。
    """

    percentile: float
    pixel_percentile: float
    pixel_image_percentile: float
    normal_image_count: int
    global_threshold: float
    global_pixel_threshold: float
    categories: dict[str, ScoreCalibrationCategory]


class ScoreCalibration(ScoreCalibrationRequired, total=False):
    """支持第二阶段连通组件阈值的完整校准契约。

    Attributes:
        component_percentile (float): 正常连通组件分数校准分位数。
        global_component_threshold (float): 未知类别回退使用的组件阈值。
    """

    component_percentile: float
    global_component_threshold: float


class InferenceResultRequired(TypedDict):
    """任何一次推理都必须按输入顺序返回的核心字段。

    Attributes:
        image_paths (list[str]): 保持输入顺序的原图路径。
        image_scores (ScoreVector): 与路径同序的一维 ``float32`` 最终图像分数。
        anomaly_maps (list[FloatMap]): 每张原图分辨率的二维 ``float32`` 异常图。
        quality_results (list[ImageQualityResult]): 与路径同序的采集质量结果。
    """

    image_paths: list[str]
    image_scores: ScoreVector
    anomaly_maps: list[FloatMap]
    quality_results: list[ImageQualityResult]


class InferenceResult(InferenceResultRequired, total=False):
    """粗扫、全局路由、复核与校准共同形成的推理结果。

    必需字段和所有逐样本可选字段均与 ``image_paths`` 顺序严格对齐。
    ``anomaly_maps`` 为原图分辨率二维 ``float32`` 数组；显示图为 HWC RGB
    ``uint8``；二值异常图为取值 ``0/1`` 的二维 ``uint8``。

    Attributes:
        display_images (dict[str, ImageArray] | None): 原图路径到固定尺寸 RGB
            ``uint8`` 显示图的映射。
        image_thresholds (ScoreVector): 与最终图像分数比较的分类别阈值。
        decision_thresholds (ScoreVector): 二分类判定实际使用的图像或组件阈值。
        pixel_thresholds (ScoreVector): 生成二值掩码和组件统计的分类别像素阈值。
        component_scores (list[float]): 最强组件与原始全局分数的保守组合。
        raw_image_scores (ScoreVector): 应用组件统计前的粗到细图像分数。
        decisions (list[DecisionState]): ``OK`` 或 ``NG`` 判定。
        decision_reasons (list[str]): 与判定同序的机器可读原因。
        component_summaries (list[ComponentStatistics]): 每张异常图的连通区域摘要。
        is_defect (list[bool]): 与 ``NG`` 判定一一对应的兼容布尔判定。
        binary_anomaly_maps (list[BinaryMask]): 取值 ``0/1`` 的原图二维预测掩码。
        refinement_statistics (list[RefinementStatistics]): 每张图的精修网格覆盖
            和算力代理统计，顺序与 ``image_paths`` 一致。
        inference_timing (InferenceTiming): 当前批次各阶段耗时，用于性能监控，
            不参与模型判定。
    """

    display_images: dict[str, ImageArray] | None
    image_thresholds: ScoreVector
    decision_thresholds: ScoreVector
    pixel_thresholds: ScoreVector
    component_scores: list[float]
    raw_image_scores: ScoreVector
    decisions: list[DecisionState]
    decision_reasons: list[str]
    component_summaries: list[ComponentStatistics]
    is_defect: list[bool]
    binary_anomaly_maps: list[BinaryMask]
    refinement_statistics: list[RefinementStatistics]
    inference_timing: InferenceTiming
