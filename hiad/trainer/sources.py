from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from hiad.data import HRSample


@dataclass(frozen=True)
class TrainingSources:
    """通过统一清单校验的正常训练样本及其类别集合。

    Attributes:
        samples (tuple[HRSample, ...]): 路径唯一、无缺陷掩码且标签为正常的样本。
        categories (tuple[str, ...]): 去重并排序后的非空类别名称。
    """

    samples: tuple[HRSample, ...]
    categories: tuple[str, ...]


def _resolve_training_path(data_root: Path, filename: object) -> Path:
    """把非空相对文件名解析为数据根目录下的绝对路径。

    Args:
        data_root (Path): 已解析的统一数据集根目录。
        filename (object): 清单中的相对图像路径。

    Returns:
        Path: 基于 ``data_root`` 解析后的路径。

    Raises:
        ValueError: 文件名为空、不是字符串或是绝对路径。
    """
    if not isinstance(filename, str) or not filename:
        raise ValueError("Training record filename must be a non-empty string")
    relative_path = Path(filename)
    if relative_path.is_absolute():
        raise ValueError("Training record filename must be relative to data_root")
    return (data_root / relative_path).resolve()


def validate_unified_training_samples(
    samples: Iterable[HRSample],
) -> TrainingSources:
    """验证训练只使用有类别、无缺陷掩码且路径唯一的正常原图。

    Args:
        samples (Iterable[HRSample]): 待验证的延迟加载样本。

    Returns:
        TrainingSources: 固化为元组的样本和排序后的类别集合。

    Raises:
        TypeError: 任一元素不是 :class:`HRSample`。
        ValueError: 样本为空，包含异常标签/掩码、空类别/路径或重复解析路径。
    """
    training_samples = tuple(samples)
    if not training_samples:
        raise ValueError("Training samples must not be empty")

    resolved_paths: list[Path] = []
    categories: set[str] = set()
    for index, sample in enumerate(training_samples):
        if not isinstance(sample, HRSample):
            raise TypeError("Training samples must contain HRSample objects")
        if sample.mask is not None or (
            sample.label is not None
            and (isinstance(sample.label, bool) or sample.label != 0)
        ):
            raise ValueError(
                "Training samples must contain only mask-free normal images; "
                f"invalid sample index: {index}"
            )
        category = sample.clsname
        if not isinstance(category, str) or not category.strip():
            raise ValueError(
                f"Training sample index {index} must have a non-empty clsname"
            )
        image_path = sample.image.image_path
        if not isinstance(image_path, (str, os.PathLike)) or not os.fspath(image_path):
            raise ValueError("Training sample image paths must be non-empty")
        resolved_paths.append(Path(image_path).expanduser().resolve())
        categories.add(category)

    if len(set(resolved_paths)) != len(resolved_paths):
        raise ValueError("Duplicate resolved path in training samples")
    return TrainingSources(
        samples=training_samples,
        categories=tuple(sorted(categories)),
    )


def load_unified_training_samples(
    data_root: str | os.PathLike[str],
) -> tuple[list[HRSample], tuple[str, ...]]:
    """从统一训练清单创建并验证正常样本。

    Args:
        data_root (str | os.PathLike[str]): 包含 ``train_uni.jsonl`` 及其相对资源的
            数据集根目录。

    Returns:
        tuple[list[HRSample], tuple[str, ...]]: 保持清单行序的延迟加载样本，以及
        排序去重后的类别名称。

    Raises:
        FileNotFoundError: 统一训练清单不存在。
        OSError: 清单无法读取。
        json.JSONDecodeError: 任一清单行不是合法 JSON。
        TypeError: 任一记录顶层不是对象。
        ValueError: 清单包含空行、异常/掩码记录、空类别或无效字段类型。
    """
    root = Path(data_root).expanduser().resolve()
    metadata_path = root / "train_uni.jsonl"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Unified training metadata not found: {metadata_path}")

    samples: list[HRSample] = []
    with metadata_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise ValueError(
                    f"Blank metadata record at {metadata_path}:{line_number}"
                )
            record = json.loads(line)
            if not isinstance(record, Mapping):
                raise TypeError(
                    f"Metadata record at {metadata_path}:{line_number} must be a mapping"
                )
            label = record.get("label")
            category = record.get("clsname")
            foreground_name = record.get("foreground")
            label_name = record.get("label_name")
            if (
                isinstance(label, bool)
                or label != 0
                or record.get("mask") is not None
                or not isinstance(category, str)
                or not category.strip()
            ):
                raise ValueError(
                    f"{metadata_path} must contain only normal records with clsname; "
                    f"invalid record at line {line_number}"
                )
            if label_name is not None and not isinstance(label_name, str):
                raise ValueError(
                    f"label_name at {metadata_path}:{line_number} must be a string or null"
                )
            samples.append(
                HRSample(
                    image=os.fspath(_resolve_training_path(root, record.get("filename"))),
                    foreground=(
                        os.fspath(root / foreground_name)
                        if isinstance(foreground_name, str) and foreground_name
                        else None
                    ),
                    clsname=category,
                    label=0,
                    label_name=label_name,
                )
            )

    sources = validate_unified_training_samples(samples)
    return list(sources.samples), sources.categories
