from __future__ import annotations

import argparse
import json
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Literal, TypeAlias, TypedDict

from PIL import Image


Split: TypeAlias = Literal["train", "test"]
Region: TypeAlias = tuple[int, int, int, int]


class DatasetSummary(TypedDict):
    """统一数据集构建完成后的可落盘摘要。

    Attributes:
        source_dir (str): 解析后的源数据根目录。
        manifest (str): 解析后的输入 JSONL 清单路径。
        train (int): 训练记录数量。
        test (int): 测试记录数量。
        categories (list[str]): 排序后的业务类别名称。
        image_sizes (list[list[int]]): 输出图像尺寸列表，每项为
            ``[width, height]``。
    """

    source_dir: str
    manifest: str
    train: int
    test: int
    categories: list[str]
    image_sizes: list[list[int]]


def parse_args() -> argparse.Namespace:
    """解析统一数据集构建脚本的命令行参数。

    Returns:
        argparse.Namespace: 包含源目录、JSONL 清单和输出根目录的参数对象。
    """
    parser = argparse.ArgumentParser(
        description="Build a unified anomaly-detection dataset from an explicit JSONL manifest"
    )
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-root", required=True)
    return parser.parse_args()


def _read_manifest(path: Path) -> list[dict[str, object]]:
    """读取非空 JSONL 清单，忽略空行并要求每条记录为对象。

    Args:
        path (Path): UTF-8 编码的源清单路径。

    Returns:
        list[dict[str, object]]: 按行序排列的清单记录。

    Raises:
        OSError: 清单无法读取。
        json.JSONDecodeError: 任一非空行不是合法 JSON。
        TypeError: 任一记录的顶层值不是对象。
        ValueError: 清单没有有效记录。
    """
    records: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise TypeError(f"Manifest record {line_number} must be a mapping")
            records.append(record)
    if not records:
        raise ValueError("Manifest must contain at least one record")
    return records


def _source_path(source_dir: Path, value: object, field: str) -> Path:
    """解析并限制清单路径必须指向源目录内的现有文件。

    Args:
        source_dir (Path): 已解析的源数据根目录。
        value (object): 清单中的相对路径值。
        field (str): 用于错误消息的字段名称。

    Returns:
        Path: 位于 ``source_dir`` 内的已解析文件路径。

    Raises:
        ValueError: 路径为空或是绝对路径。
        FileNotFoundError: 路径越出源目录或不是现有文件。
    """
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty relative path")
    relative = Path(value)
    if relative.is_absolute():
        raise ValueError(f"{field} must be relative to source_dir")
    path = (source_dir / relative).resolve()
    if source_dir not in path.parents or not path.is_file():
        raise FileNotFoundError(f"{field} not found under source_dir: {value}")
    return path


def _validate_roi(value: object) -> Region | None:
    """校验可选 ROI 为原图像素 ``[x, y, width, height]``。

    Args:
        value (object): 清单中的可选 ROI 值。

    Returns:
        Region | None: ``(x, y, width, height)`` 元组，或未指定时为 ``None``。

    Raises:
        ValueError: 值不是四个整数，坐标为负或宽高不为正。
    """
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError("roi must be [x, y, width, height]")
    x, y, width, height = value
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise ValueError("roi values must be integers")
    if x < 0 or y < 0 or width <= 0 or height <= 0:
        raise ValueError("roi must contain non-negative x/y and positive width/height")
    return x, y, width, height


def _validate_category(value: object) -> str:
    """校验类别是可安全用作单级目录名的非空字符串。

    Args:
        value (object): 清单中的 ``clsname``。

    Returns:
        str: 去除首尾空白的单级类别名。

    Raises:
        ValueError: 类别为空、不是字符串、是点目录或包含路径分隔符。
    """
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Every manifest record must contain clsname")
    value = value.strip()
    if value in {".", ".."} or any(char in value for char in "/\\"):
        raise ValueError("clsname must be a single path component")
    return value


def _copy_image(
    source: Path,
    destination: Path,
    roi: Region | None,
    *,
    is_mask: bool = False,
) -> tuple[int, int]:
    """复制或按原图 ``xywh`` ROI 同步裁剪图像与标注。

    Args:
        source (Path): 源图像或掩码路径。
        destination (Path): 输出路径，父目录会自动创建。
        roi (Region | None): 原图像素 ``(x, y, width, height)``；``None`` 表示
            保留完整文件。
        is_mask (bool): 是否把裁剪结果重新二值化为 ``0/255`` 掩码。

    Returns:
        tuple[int, int]: 输出文件的 ``(width, height)``。

    Raises:
        OSError: 源文件无法读取或目标文件无法写入。
        ValueError: ROI 超出源图边界。
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    if roi is None:
        shutil.copy2(source, destination)
        with Image.open(source) as image:
            return image.size

    with Image.open(source) as image:
        x, y, width, height = roi
        if x + width > image.width or y + height > image.height:
            raise ValueError(f"ROI {roi} exceeds image bounds {image.size} for {source}")
        cropped = image.crop((x, y, x + width, y + height))
        if is_mask:
            cropped = cropped.point(lambda value: 255 if value else 0)
        cropped.save(destination)
        return cropped.size


def _validate_split(value: object) -> Split:
    """将数据划分限制为稳定的 ``train`` 或 ``test`` 值。

    Args:
        value (object): 清单中的 ``split`` 字段。

    Returns:
        Split: 字面量 ``train`` 或 ``test``。

    Raises:
        ValueError: 值不是受支持的数据划分。
    """
    if value == "train":
        return "train"
    if value == "test":
        return "test"
    raise ValueError("Every manifest record must use split 'train' or 'test'")


def _output_relative(
    record: Mapping[str, object],
    source: Path,
    source_dir: Path,
) -> Path:
    """生成保留源目录层级的输出图像相对路径。

    Args:
        record (Mapping[str, object]): 已读取的源清单记录。
        source (Path): 位于 ``source_dir`` 内的源图路径。
        source_dir (Path): 源数据根目录。

    Returns:
        Path: ``images/<split>/<clsname>/...`` 形式的相对路径。
    """
    split = _validate_split(record.get("split"))
    clsname = _validate_category(record.get("clsname"))
    return Path("images") / split / clsname / source.relative_to(source_dir)


def build_dataset(
    source_dir: str | Path,
    manifest: str | Path,
    output_root: str | Path,
) -> DatasetSummary:
    """依据显式清单构建统一数据集，不推断数据划分或缺陷标签。

    Args:
        source_dir (str | Path): 清单中相对图像、前景和掩码路径的根目录。
        manifest (str | Path): 显式描述划分、类别、标签和可选 ROI 的 JSONL。
        output_root (str | Path): 输出根目录；若已存在则必须为空。

    Returns:
        DatasetSummary: 数据来源、训练/测试数量、类别和输出尺寸摘要。

    Raises:
        FileExistsError: 输出目录已经包含文件。
        OSError: 输入无法读取或输出无法创建。
        TypeError: 清单记录不是 JSON 对象。
        ValueError: 路径、ROI、类别、划分或训练标签违反统一数据集约束。
    """
    source_dir = Path(source_dir).resolve()
    manifest = Path(manifest).resolve()
    output_root = Path(output_root).resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_root}")

    source_records = _read_manifest(manifest)
    output_root.mkdir(parents=True, exist_ok=True)
    output_records: dict[Split, list[dict[str, object]]] = {
        "train": [],
        "test": [],
    }
    sizes: set[tuple[int, int]] = set()

    for record in source_records:
        source = _source_path(source_dir, record.get("filename"), "filename")
        roi = _validate_roi(record.get("roi"))
        relative = _output_relative(record, source, source_dir)
        split = _validate_split(record.get("split"))
        label = record.get("label")
        if isinstance(label, bool) or (label is not None and label not in {0, 1}):
            raise ValueError("label must be 0, 1, or omitted")
        if split == "train" and (label != 0 or record.get("mask")):
            raise ValueError("Training records must be normal and mask-free")
        sizes.add(_copy_image(source, output_root / relative, roi))

        output_record: dict[str, object] = {
            "filename": relative.as_posix(),
            "foreground": None,
            "mask": None,
            "clsname": _validate_category(record["clsname"]),
            "label": label,
            "label_name": record.get("label_name"),
        }
        for field, is_mask in (("foreground", True), ("mask", True)):
            if record.get(field):
                auxiliary = _source_path(source_dir, record[field], field)
                auxiliary_relative = Path("annotations") / field / relative
                auxiliary_relative = auxiliary_relative.with_suffix(auxiliary.suffix)
                _copy_image(
                    auxiliary,
                    output_root / auxiliary_relative,
                    roi,
                    is_mask=is_mask,
                )
                output_record[field] = auxiliary_relative.as_posix()
        output_records[split].append(output_record)

    for split, records in output_records.items():
        path = output_root / f"{split}_uni.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=True) + "\n")

    summary: DatasetSummary = {
        "source_dir": str(source_dir),
        "manifest": str(manifest),
        "train": len(output_records["train"]),
        "test": len(output_records["test"]),
        "categories": sorted(
            {_validate_category(record.get("clsname")) for record in source_records}
        ),
        "image_sizes": [list(size) for size in sorted(sizes)],
    }
    with (output_root / "dataset_summary.json").open(
        "w", encoding="utf-8", newline="\n"
    ) as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=True)
        stream.write("\n")
    return summary


def main() -> None:
    """执行数据集构建 CLI，并以 JSON 形式打印最终摘要。"""
    args = parse_args()
    result = build_dataset(args.source_dir, args.manifest, args.output_root)
    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
