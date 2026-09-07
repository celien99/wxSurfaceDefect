from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path


def read_jsonl_records(path: str | os.PathLike[str]) -> list[dict[str, object]]:
    """读取统一 JSONL 清单，并拒绝空行和非对象记录。

    Args:
        path (str | os.PathLike[str]): UTF-8 编码的 JSONL 清单路径。

    Returns:
        list[dict[str, object]]: 按文件行序排列的 JSON 对象副本。

    Raises:
        OSError: 文件不存在、不可读或读取过程中发生 I/O 错误。
        ValueError: 清单包含空行，或某行不是合法 JSON。
        TypeError: 某行 JSON 的顶层值不是对象。
    """
    metadata_path = Path(path)
    records: list[dict[str, object]] = []
    with metadata_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise ValueError(
                    f"Blank metadata record at {metadata_path}:{line_number}"
                )
            record = json.loads(line)
            if not isinstance(record, Mapping):
                raise TypeError(
                    f"Metadata record at {metadata_path}:{line_number} "
                    "must be a mapping"
                )
            records.append(dict(record))
    return records
